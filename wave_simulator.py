import contextlib
import pickle

import numpy as np
import torch
from numba.cuda.random import create_xoroshiro128p_states

from apache_helper import ApachePlatform
from generate_trainingdata import getVAECov, formatTargetClutterData
from models import InfoVAE, BetaVAE, WAE_MMD, Encoder
from simulib.simulation_functions import getElevation, llh2enu, findPowerOf2, db, enu2llh, azelToVec, genPulse
from data_converter.aps_io import loadCorrectionGPSData, loadGPSData, loadGimbalData
from simulib import getMaxThreads, backproject, genRangeProfile, applyRadiationPatternCPU
from simulib.mimo_functions import genChirpAndMatchedFilters, genChannels
from simulib.grid_helper import SDREnvironment
from simulib.platform_helper import SDRPlatform, RadarPlatform
import cupy as cupy
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from tqdm import tqdm
from data_converter.SDRParsing import load
import yaml
from scipy.signal.windows import taylor
from scipy.signal import sawtooth
from scipy.interpolate import RectBivariateSpline
from scipy.linalg import convolution_matrix
from itertools import permutations
import imageio.v2 as imageio
from celluloid import Camera

from waveform_model import GeneratorModel, getTrainTransforms


def getFootprint(pos, az, el, az_bw, el_bw, near_range, far_range):
    footprint = []
    for rng in [near_range, far_range]:
        footprint.extend(
            pos + rng * azelToVec(az + pt[0] * az_bw, el + pt[1] * el_bw)
            for pt in list(set(list(permutations([-1, 1, -1, 1], 2))))
        )
    return np.array(footprint)


def reloadWaveforms(wave_mdl, pulse_data, nr, fft_len, tc_d, rps, cpi_len, bwidth):
    pdata = torch.tensor(formatTargetClutterData(pulse_data, fft_len), device=wave_mdl.device)
    nn_output = wave_mdl([wave_mdl.decoder.encode(pdata).to(wave_mdl.device).unsqueeze(0), tc_d.to(wave_mdl.device).unsqueeze(0),
                          torch.tensor([nr]), torch.tensor([[bwidth]])])
    waves = wave_mdl.getWaveform(nn_output=nn_output).cpu().data.numpy().squeeze(0) * 1e4
    _, chirps, mfilts = genChirpAndMatchedFilters(waves, rps, bwidth, fs, fc, fft_len, cpi_len)
    return chirps, mfilts


# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
GPS_UPDATE_RATE_HZ = 100

# Load all the settings files
with open('./wave_simulator.yaml') as y:
    settings = yaml.safe_load(y.read())
with open('./vae_config.yaml', 'r') as file:
    try:
        wave_config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

nbpj_pts = (int(settings['grid_height'] * settings['pts_per_m']), int(settings['grid_width'] * settings['pts_per_m']))
sim_settings = settings['simulation_params']

print('Loading SDR file...')
# This pickle should have an ASI file attached and use_jump_correction=False
sdr = load(settings['bg_file'], import_pickle=False, export_pickle=False, use_jump_correction=False)

if settings['origin'] is None:
    try:
        settings['origin'] = (sdr.ash['geo']['centerY'], sdr.ash['geo']['centerX'],
                              getElevation(sdr.ash['geo']['centerY'], sdr.ash['geo']['centerX']))
        heading = sdr.gim.initial_course_angle
    except TypeError:
        heading = sdr.gim.initial_course_angle
        pt = (sdr.gps_data['lat'].values[0], sdr.gps_data['lon'].values[0])
        alt = getElevation(*pt)
        nrange = ((sdr[settings['channel']].receive_on_TAC - sdr[settings['channel']].transmit_on_TAC) / TAC -
                  sdr[settings['channel']].pulse_length_S * settings['partial_pulse_percent']) * c0 / 2
        frange = ((sdr[settings['channel']].receive_off_TAC - sdr[settings['channel']].transmit_on_TAC) / TAC -
                  sdr[settings['channel']].pulse_length_S * settings['partial_pulse_percent']) * c0 / 2
        mrange = (nrange + frange) / 2
        settings['origin'] = enu2llh(mrange * np.sin(heading), mrange * np.cos(heading), 0.,
                                     (pt[0], pt[1], alt))

bg = SDREnvironment(sdr)
ref_llh = bg.ref

plane_pos = llh2enu(40.138052, -111.660027, 1365, bg.ref)

# Generate a platform
print('Generating platform...', end='')

if sim_settings['use_sdr_gps']:
    rpi = SDRPlatform(sdr, ref_llh, channel=settings['channel'])
    goff = np.array(
        [sdr.gim.x_offset, sdr.gim.y_offset, sdr.gim.z_offset])
    grot = np.array([sdr.gim.roll * DTR, sdr.gim.pitch * DTR, sdr.gim.yaw * DTR])
    rpi.fc = sdr[settings['channel']].fc
    rpi.bwidth = sdr[settings['channel']].bw
    nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = (
        rpi.getRadarParams(settings['fdelay'], settings['plp'], settings['upsample']))
else:
    # Run directly at the plane from the south
    grid_origin = plane_pos  # llh2enu(*bg.origin, bg.ref)
    full_scan = int(sim_settings['az_bw'] / sim_settings['scan_rate'] * sim_settings['prf'])
    full_scan -= 0 if full_scan % 2 == 0 else 1
    # Get parameters for the Apache specs
    req_slant_range = sim_settings['standoff_range']
    req_alt = wave_config['apache_params']['alt_max']
    ground_range = np.sqrt(req_slant_range ** 2 - req_alt ** 2)
    req_dep_ang = np.arccos(req_alt / req_slant_range) + sim_settings['el_bw'] * DTR
    ngpssam = int(sim_settings['collect_duration'] * GPS_UPDATE_RATE_HZ)
    e = np.linspace(ground_range + .01, ground_range, ngpssam) + grid_origin[0]
    n = np.linspace(0, 0, ngpssam) + grid_origin[1]
    u = np.linspace(req_alt, req_alt, ngpssam) + grid_origin[2]
    r = np.zeros_like(e)
    p = np.zeros_like(e)
    y = np.zeros_like(e) + np.pi / 2
    t = np.arange(ngpssam) / GPS_UPDATE_RATE_HZ
    gim_pan = (sawtooth(np.pi * sim_settings['scan_rate'] / sim_settings['scan_angle'] *
                        np.arange(ngpssam) / GPS_UPDATE_RATE_HZ, .5)
               * sim_settings['scan_angle'] / 2 * DTR)
    gim_el = np.zeros_like(gim_pan) + np.arccos(req_alt / req_slant_range)
    goff = np.array(
        [wave_config['apache_params']['phase_center_offset_m'], 0., wave_config['apache_params']['wheel_height_m']])
    grot = np.array([0., 0., 0.])
    rpi = ApachePlatform(wave_config['apache_params'], e, n, u, r, p, y, t, dep_angle=req_dep_ang / DTR,
                         gimbal=np.array([gim_pan, gim_el]).T, gimbal_rotations=grot, gimbal_offset=goff, fs=2e9)
    rpi.fc = 9.6e9
    rpi.bwidth = 400e6
    nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = (
        rpi.getRadarParams(u.mean(), settings['plp'], settings['upsample']))
    print(f'Plane slant range is {np.linalg.norm(plane_pos - rpi.pos(rpi.gpst), axis=1).mean()}')
wave_fft_len = wave_config['settings']['fft_len']
rbins_gpu = cupy.array(ranges, dtype=np.float64)

plat_e, plat_n, plat_u = rpi.pos(rpi.gpst).T
plat_r, plat_p, plat_y = rpi.att(rpi.gpst).T
gimbal = np.array([rpi.pan(rpi.gpst), rpi.tilt(rpi.gpst)]).T
cpi_len = settings['cpi_len']

rpref, rps, vx_array = genChannels(settings['antenna_params']['n_tx'], settings['antenna_params']['n_rx'],
                                   settings['antenna_params']['tx_pos'], settings['antenna_params']['rx_pos'],
                                   plat_e, plat_n, plat_u, plat_r, plat_p, plat_y, rpi.gpst, gimbal, goff, grot,
                                   rpi.dep_ang, rpi.az_half_bw, rpi.el_half_bw, rpi.fs)

# Get reference data
fs = rpi.fs
bwidth = rpi.bwidth
fc = rpi.fc
print('Done.')

# Generate values needed for backprojection
print('Calculating grid parameters...')

# Chirp and matched filter calculations
bpj_wavelength = c0 / fc

# Run through loop to get data simulated
gap_len = cpi_len
if settings['simulation_params']['use_sdr_gps']:
    data_t = sdr[settings['channel']].pulse_time
    idx_t = sdr[settings['channel']].frame_num
else:
    data_t = rpi.getValidPulseTimings(settings['simulation_params']['prf'], nr / TAC, cpi_len, as_blocks=True)
    data_t = [d for d in data_t if len(d) > 0]
    gap_len = len(data_t[0])

ang_dist_traveled_over_cpi = gap_len / sim_settings['prf'] * sim_settings['scan_rate'] * DTR

# Chirps and Mfilts for each channe
if sim_settings['use_sdr_waveform']:
    waves = np.array([np.fft.fft(genPulse(np.linspace(0, 1, 10),
                                          np.linspace(0, 1, 10), nr, fs, fc, bwidth), fft_len),
                      np.fft.fft(genPulse(np.linspace(0, 1, 10),
                                          np.linspace(1, 0, 10), nr, fs, fc, bwidth), fft_len)]
                     )
    _, chirps, mfilts = genChirpAndMatchedFilters(waves, rps, bwidth, fs, fc, fft_len, cpi_len)
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mfilt = sdr.genMatchedFilter(0, fft_len=fft_len)
    print('Setting up decoder...')
    decoder = Encoder(**wave_config['model_params'], fft_len=wave_config['settings']['fft_len'],
                      params=wave_config['exp_params'])
    with contextlib.suppress(RuntimeError):
        decoder.load_state_dict(torch.load('./model/inference_model.state'))
    decoder.requires_grad = False
    decoder.eval()

    tcdata = torch.tensor(np.fromfile(f'{wave_config["dataset_params"]["data_path"]}/targets.enc', dtype=np.float32).reshape(
        (-1, wave_config['model_params']['latent_dim'])))

    try:
        print(f'Wavemodel save file loading...')
        with open('./model/current_model_params.pic', 'rb') as f:
            generator_params = pickle.load(f)
        wave_mdl = GeneratorModel(**generator_params, decoder=decoder)
        wave_mdl.load_state_dict(torch.load(generator_params['state_file']))
    except RuntimeError as e:
        print(f'Wavemodel save file does not match current structure. Re-running with new structure.\n{e}')
        wave_mdl = GeneratorModel(fft_sz=fft_len,
                                  stft_win_sz=wave_config['settings']['stft_win_sz'],
                                  clutter_latent_size=wave_config['model_params']['latent_dim'],
                                  target_latent_size=wave_config['model_params']['latent_dim'], n_ants=2)
    wave_mdl.eval()

    active_clutter = sdr.getPulses(sdr[0].frame_num[:wave_config['settings']['cpi_len']], 0)[1]
    bandwidth_model = torch.tensor(400e6, device=wave_mdl.device)

    chirps, mfilts = reloadWaveforms(wave_mdl, active_clutter, nr, fft_len,
                                     tcdata[0, ...], rps, cpi_len, bandwidth_model)

# This replaces the ASI background with a custom image
bg.resampleGrid(settings['origin'], settings['grid_width'], settings['grid_height'],
                        *nbpj_pts, bg.heading if settings['rotate_grid'] else 0)
'''bg_image = imageio.imread('/data6/Jeff_Backup/Pictures/josh.png').sum(axis=2)
bg_image = RectBivariateSpline(np.arange(bg_image.shape[0]), np.arange(bg_image.shape[1]), bg_image)(
    np.linspace(0, bg_image.shape[0], bg.refgrid.shape[0]), np.linspace(0, bg_image.shape[1], bg.refgrid.shape[1])) / 750'''
'''bg_image = np.zeros(bg.refgrid.shape)
bg_image[125, 125] = 100
bg_image[150, 150] = 10
bg._refgrid = bg_image'''

# Constant part of the radar equation
receive_power_scale = (settings['antenna_params']['transmit_power'][0] / .01 *
                       (10 ** (settings['antenna_params']['gain'][0] / 20)) ** 2
                       * bpj_wavelength ** 2 / (4 * np.pi) ** 3)
noise_level = 10 ** (settings['noise_level'] / 20) / np.sqrt(2)

# Calculate out points on the ground
gx, gy, gz = bg.getGrid(settings['origin'], settings['grid_width'], settings['grid_height'], *nbpj_pts,
                        bg.heading if settings['rotate_grid'] else 0)
gx_gpu = cupy.array(gx, dtype=np.float64)
gy_gpu = cupy.array(gy, dtype=np.float64)
gz_gpu = cupy.array(gz, dtype=np.float64)
refgrid = bg.getRefGrid(settings['origin'], settings['grid_width'], settings['grid_height'],
                        *nbpj_pts, bg.heading if settings['rotate_grid'] else 0)
refgrid_gpu = cupy.array(refgrid, dtype=np.float64)

if settings['debug']:
    pts_debug = cupy.zeros((3, *gx.shape), dtype=np.float64)
    angs_debug = cupy.zeros((3, *gx.shape), dtype=np.float64)
else:
    pts_debug = cupy.zeros((1, 1), dtype=np.float64)
    angs_debug = cupy.zeros((1, 1), dtype=np.float64)

# Generate range/angle grid for a given position
# First, get the center of the grid for angle calcs
plat_to_grid = rpi.pos(rpi.gpst[0]) - llh2enu(*settings['origin'], bg.ref)
data = rpi.boresight(rpi.gpst)
sorted_data = data[np.lexsort(data.T), :]
row_mask = np.append([True], np.any(np.diff(sorted_data, axis=0), 1))
unique_boresights = sorted_data[row_mask].T
az_spread = np.linspace(rpi.pan(rpi.gpst).min(), rpi.pan(rpi.gpst).max(), nbpj_pts[0])
im_ranges = np.linspace(2181 - 100., 2181 + 100., nbpj_pts[1])
imgrid = np.hstack([azelToVec(az_spread, np.ones_like(az_spread) *
                              np.arcsin(plat_to_grid[2] / np.linalg.norm(plat_to_grid))) * g for g in im_ranges])
im_x = (imgrid[0, :] + rpi.pos(rpi.gpst[0])[0]).reshape((len(az_spread), len(im_ranges)), order='F')
im_y = (imgrid[1, :] + rpi.pos(rpi.gpst[0])[1]).reshape((len(az_spread), len(im_ranges)), order='F')
im_z = (imgrid[2, :] + rpi.pos(rpi.gpst[0])[2]).reshape((len(az_spread), len(im_ranges)), order='F')
imx_gpu = cupy.array(gx, dtype=np.float64)
imy_gpu = cupy.array(gy, dtype=np.float64)
imz_gpu = cupy.array(gz, dtype=np.float64)

# GPU device calculations
threads_per_block = getMaxThreads()
bpg_bpj = (max(1, (nbpj_pts[0]) // threads_per_block[0] + 1), (nbpj_pts[1]) // threads_per_block[1] + 1)

rng_states = create_xoroshiro128p_states(threads_per_block[0] * bpg_bpj[0], seed=1)

# Get pointing vector for MIMO consolidation
ublock = np.array(
    [azelToVec(n, 0) for n in np.linspace(-ang_dist_traveled_over_cpi / 2, ang_dist_traveled_over_cpi / 2, cpi_len)]).T
fine_ucavec = np.exp(-1j * 2 * np.pi * sdr[0].fc / c0 * vx_array.dot(ublock))
array_factor = fine_ucavec.conj().T.dot(np.eye(vx_array.shape[0])).dot(fine_ucavec)[:, 0] / vx_array.shape[0]

test = None
print('Running simulation...')
pulse_pos = 0
# Data blocks for imaging
rbi_image = np.zeros(im_x.shape, dtype=np.complex128)

nz = np.zeros(cpi_len)
tx_pattern = applyRadiationPatternCPU(nz, np.linspace(-ang_dist_traveled_over_cpi / 2, ang_dist_traveled_over_cpi / 2,
                                                      cpi_len),
                                      nz, nz, nz, nz, rpref.az_half_bw, rpref.el_half_bw)
H = convolution_matrix(tx_pattern * array_factor, gap_len, 'valid')

# Truncated SVD for superresolution
U, eig, Vt = np.linalg.svd(H, full_matrices=False)
knee = sum(np.gradient(eig) > np.gradient(eig).mean())
eig[knee:] = 0
eig[:knee] = 1 / eig[:knee]
Hinv = Vt.T.dot(np.diag(eig)).dot(U.T)

# Hinv = np.linalg.pinv(H)

H_w = np.fft.fft(tx_pattern * array_factor, gap_len)
abs_range_min = np.linalg.norm(grid_origin - rpi.pos(rpi.gpst), axis=1).min()
det_pts = list()
ex_chirps = list()

camfig, camax = plt.subplots(1, 1)
cam = Camera(camfig)
for tidx, ts in tqdm(enumerate(data_t), total=len(data_t)):
    tmp_len = len(ts)
    ex_chirps.append(np.array(chirps[0].get()))
    if not sim_settings['use_sdr_waveform'] and tmp_len == gap_len:
        chirps, mfilts = reloadWaveforms(wave_mdl, active_clutter, nr, fft_len,
                                     tcdata[0, ...], rps, cpi_len, bandwidth_model)
    # Pan and Tilt are shared by each channel, antennas are all facing the same way
    panrx = rpi.pan(ts)
    elrx = rpi.tilt(ts)
    panrx_gpu = cupy.array(panrx, dtype=np.float64)
    elrx_gpu = cupy.array(elrx, dtype=np.float64)
    # These are relative to the origin
    posrx = rpref.rxpos(ts)
    postx = rpref.txpos(ts)
    pt_ref = grid_origin - posrx

    beamform_data = cupy.zeros((nsam * settings['upsample'], tmp_len), dtype=np.complex128)

    for ch_idx, rp in enumerate(rps):
        posrx = rp.rxpos(ts)
        postx = rp.txpos(ts)
        posrx_gpu = cupy.array(posrx, dtype=np.float64)
        postx_gpu = cupy.array(postx, dtype=np.float64)

        pd_r = cupy.zeros((nsam, tmp_len), dtype=np.float64)
        pd_i = cupy.zeros((nsam, tmp_len), dtype=np.float64)

        genRangeProfile[bpg_bpj, threads_per_block](gx_gpu, gy_gpu, gz_gpu, refgrid_gpu,
                                                    posrx_gpu, postx_gpu, panrx_gpu, elrx_gpu, panrx_gpu, elrx_gpu,
                                                    pd_r, pd_i, rng_states, pts_debug,
                                                    angs_debug, bpj_wavelength, near_range_s, rpref.fs,
                                                    rpref.az_half_bw, rpref.el_half_bw, receive_power_scale,
                                                    settings['pts_per_tri'],
                                                    settings['debug'])
        pdata = pd_r + 1j * pd_i
        if settings['ngrids'] > 1:
            pass
        rtdata = cupy.fft.fft(pdata, fft_len, axis=0) * chirps[ch_idx][:, None] * mfilts[ch_idx][:, None]
        upsample_data = cupy.array(np.random.normal(0, noise_level, (up_fft_len, tmp_len)) +
                                   1j * np.random.normal(0, noise_level, (up_fft_len, tmp_len)),
                                   dtype=np.complex128)
        upsample_data[:fft_len // 2, :] += rtdata[:fft_len // 2, :]
        upsample_data[-fft_len // 2:, :] += rtdata[-fft_len // 2:, :]
        rtdata = cupy.fft.ifft(upsample_data, axis=0)[:nsam * settings['upsample'], :]
        cupy.cuda.Device().synchronize()

        # This is equivalent to a dot product
        beamform_data += rtdata
    cupy.cuda.Device().synchronize()

    # Real-beam imaging
    range_walk = np.linalg.norm(pt_ref, axis=1) - abs_range_min
    if tmp_len == gap_len:

        rbi_y = beamform_data.get()  # * np.exp(-1j * 2 * np.pi * fc / c0 * range_walk)
        '''g_k = np.zeros((H.shape[1], nsam))
        b_k = np.zeros_like(g_k)
        sig_k = np.linalg.pinv(.01 * H.T.dot(H) + 10 * np.eye(H.shape[1])).dot(.01 * H.T.dot(rbi_y.T) + 10 * (g_k - b_k))'''
        # sig_k = np.fft.ifft(np.fft.fft(rbi_y, axis=1) / H_w)[:, cpi_len // 2:-cpi_len // 2 + 1]
        sig_k = rbi_y.dot(Hinv)
        # Get the angles this scan section covers
        angs = np.logical_and(az_spread <= panrx[gap_len // 2 - H.shape[0] // 2 - 1],
                              az_spread >= panrx[gap_len // 2 + H.shape[0] // 2])
        bpj_grid = cupy.zeros((sum(angs), nbpj_pts[1]), dtype=np.complex128)
        posrx_gpu = cupy.array(rpref.rxpos(ts), dtype=np.float64)
        postx_gpu = cupy.array(rpref.txpos(ts), dtype=np.float64)

        # Backprojection only for beamformed final data
        backproject[bpg_bpj, threads_per_block](postx_gpu, posrx_gpu, imx_gpu[angs, :], imy_gpu[angs, :], imz_gpu[angs, :], rbins_gpu, panrx_gpu,
                                                elrx_gpu, panrx_gpu, elrx_gpu, beamform_data, bpj_grid,
                                                bpj_wavelength, near_range_s, rpref.fs * settings['upsample'], bwidth,
                                                rpref.az_half_bw,
                                                rpref.el_half_bw, settings['poly_num'], pts_debug, angs_debug,
                                                settings['debug'])
        cupy.cuda.Device().synchronize()
        rbi_image[angs, :] += bpj_grid.get()
        camax.imshow(db(np.array(sig_k)))
        plt.axis('tight')
        cam.snap()
        if not sim_settings['use_sdr_waveform']:
            active_clutter = rbi_y.T[:, :32]

del rtdata
del beamform_data
del upsample_data

"""
----------------------------PLOTS-------------------------------
"""
anim = cam.animate()

try:
    plt.figure('IMSHOW truth data')
    climage = db(refgrid)
    clim_image = climage[climage > -299]
    clims = (np.median(clim_image) - 3 * clim_image.std(),
             max(np.median(clim_image) + 3 * clim_image.std(), np.max(clim_image) + 2))
    plt.imshow(np.rot90(db(refgrid)), origin='lower', clim=clims, cmap='gray')
    plt.axis('tight')
    plt.axis('off')
except Exception as e:
    print(f'Error in generating background image: {e}')

plt.figure('Waveform Info')
plt.subplot(2, 1, 1)
plt.title('Spectrum')
plt.plot(db(ex_chirps[0]))
plt.plot(db(ex_chirps[1]))
plt.subplot(2, 1, 2)
plt.title('Time Series')
plt.plot(np.fft.ifft(ex_chirps[0]).real)
plt.plot(np.fft.ifft(ex_chirps[1]).real)

plt.figure('Chirp Changes')
for n in ex_chirps:
    plt.plot(db(n))
plt.show()

rbi_image = np.array(rbi_image)

plane_vec = plane_pos - rpi.pos(rpi.gpst).mean(axis=0)
climage = db(rbi_image)
clim_image = climage[climage > -299]
clims = (np.median(clim_image) - 3 * clim_image.std(),
         max(np.median(clim_image) + 3 * clim_image.std(), np.max(clim_image) + 2))
fig = plt.figure('RBI image')
ax = fig.add_subplot(111, projection='polar')
ax.pcolormesh(az_spread, np.flip(im_ranges), climage.T,
              clim=clims, edgecolors='face')
ax.scatter(np.arctan2(plane_vec[0], plane_vec[1]), np.linalg.norm(plane_vec))
ax.set_ylim(0, ranges[-1])
ax.set_xlim(az_spread.min(), az_spread.max())
ax.grid(False)
plt.axis('tight')

plt.figure('RBI cartesian')
plt.imshow(climage.T, clim=clims,
           extent=(az_spread[0] / DTR, az_spread[-1] / DTR, im_ranges[0], im_ranges[-1]), cmap='gray')
# plt.scatter(np.arctan2(plane_vec[0], plane_vec[1]) / DTR, np.linalg.norm(plane_vec))
plt.axis('tight')

pfig = px.imshow(db(refgrid), x=np.linspace(gx.min(), gx.max(), refgrid.shape[0]),
                 y=np.linspace(gy.min(), gy.max(), refgrid.shape[1]),
                 color_continuous_scale=px.colors.sequential.gray, zmin=130, zmax=180)
if settings['ngrids'] > 1:
    pfig.add_scatter(x=grid_x[1].flatten(), y=grid_y[1].flatten(),
                     marker=dict(color=db(bg_refgrid[1]).flatten(), colorscale=px.colors.sequential.gray, cmin=130,
                                 cmax=180),
                     mode='markers')
pfig.show()

plt.show()

pfig = px.scatter_3d(x=gx.flatten(), y=gy.flatten(), z=gz.flatten())
pfig.add_scatter3d(x=im_x.flatten(), y=im_y.flatten(), z=im_z.flatten())
pfig.show()

'''plt.figure('Target Angles')
plt.plot(az_spread / DTR, db(np.sum(bg_image, axis=1)))
plt.plot(az_spread / DTR, db(np.sum(rbi_image, axis=0)))'''

# px.imshow(abs(rbi_image), zmin=clims.mean() - 3 * clims.std(), zmax=clims.mean() + 3 * clims.std()).show()
