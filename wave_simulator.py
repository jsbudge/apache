import pickle

import numpy as np
import torch

from apache_helper import ApachePlatform
from generate_trainingdata import getVAECov
from models import InfoVAE, BetaVAE, WAE_MMD
from simulib.simulation_functions import getElevation, llh2enu, findPowerOf2, db, enu2llh, azelToVec, genPulse
from data_converter.aps_io import loadCorrectionGPSData, loadGPSData, loadGimbalData
from simulib import getMaxThreads, backproject,
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


def reloadWaveforms(model, device, wave_mdl, pulse_data, mfilt, rollback, nsam, nr, fft_len, tc_d, train_transforms,
                    rps,
                    cpi_len, taytay, bwidth):
    _, cov_dt = getVAECov(pulse_data, mfilt, rollback, nsam, fft_len)
    dt = train_transforms(cov_dt.astype(np.float32)).unsqueeze(0)
    waves = wave_mdl.getWaveform(model.forward(dt.to(device))[2].to(wave_mdl.device),
                                 model.forward(tc_d)[2].to(wave_mdl.device), pulse_length=[nr], bandwidth=bwidth,
                                 scale=True, custom_fft_sz=len(mfilt)).data.numpy().squeeze(0) * 1e4
    chirps = []
    mfilt_jax = []
    for rp in rps:
        chirp = jnp.array(waves[rp.tx_num, :])
        new_mfilt = chirp.conj() * taytay
        chirps.append(jnp.tile(chirp, (cpi_len, 1)).T)
        mfilt_jax.append(np.tile(new_mfilt, (cpi_len, 1)).T)
    return chirps, mfilt_jax


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

nbpj_pts = [(int(settings['grid_height'][i] * settings['pts_per_m'][i]),
             int(settings['grid_width'][i] * settings['pts_per_m'][i])) for i in range(settings['ngrids'])]
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
    nsam = rpi.calcNumSamples(settings['fdelay'], settings['plp'])
    nr = rpi.calcPulseLength(settings['fdelay'], settings['plp'], True)
    ranges = rpi.calcRangeBins(settings['fdelay'], settings['upsample'], settings['plp'])
    ranges_sampled = rpi.calcRangeBins(settings['fdelay'], 1, settings['plp'])
    fft_len = findPowerOf2(nsam + rpi.calcPulseLength(settings['fdelay'], settings['plp'], use_tac=True))
else:
    # Run directly at the plane from the south
    grid_origin = plane_pos  # llh2enu(*bg.origin, bg.ref)
    full_scan = int(sim_settings['az_bw'] / sim_settings['scan_rate'] * sim_settings['prf'])
    full_scan -= 0 if full_scan % 2 == 0 else 1
    # Get parameters for the Apache specs
    req_slant_range = sim_settings['standoff_range']
    req_alt = wave_config['apache_params']['alt_max']
    ground_range = np.sqrt(req_slant_range ** 2 - req_alt ** 2)
    req_dep_ang = np.arccos(req_alt / req_slant_range) + sim_settings['el_bw'] * DTR * 9
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
    ranges = rpi.calcRangeBins(u.mean(), settings['upsample'], settings['plp'], ranges=(1500, 1800))
    ranges_sampled = rpi.calcRangeBins(u.mean(), 1, settings['plp'], ranges=(1500, 1800))
    nsam = len(ranges_sampled)
    nr = rpi.calcPulseLength(u.mean(), settings['plp'], True)
    fft_len = findPowerOf2(nsam + nr)
    print(f'Plane slant range is {np.linalg.norm(plane_pos - rpi.pos(rpi.gpst), axis=1).mean()}')
wave_fft_len = wave_config['generate_data_settings']['fft_sz']

plat_e, plat_n, plat_u = rpi.pos(rpi.gpst).T
plat_r, plat_p, plat_y = rpi.att(rpi.gpst).T
gimbal = np.array([rpi.pan(rpi.gpst), rpi.tilt(rpi.gpst)]).T
cpi_len = settings['cpi_len']

rps = []
vx_array = []
vx_perm = [(n, q) for q in range(settings['antenna_params']['n_tx']) for n in range(settings['antenna_params']['n_rx'])]
for tx, rx in vx_perm:
    txpos = np.array(settings['antenna_params']['tx_pos'][tx])
    rxpos = np.array(settings['antenna_params']['rx_pos'][rx])
    vx_pos = rxpos + txpos
    rps.append(RadarPlatform(plat_e, plat_n, plat_u, plat_r, plat_p, plat_y, rpi.gpst, txpos, rxpos, gimbal, goff,
                             grot, rpi.dep_ang, 0., rpi.az_half_bw * 2 / DTR, rpi.el_half_bw * 2 / DTR,
                             rpi.fs, tx_num=tx, rx_num=rx))
    vx_array.append(rxpos + txpos)
rpref = RadarPlatform(plat_e, plat_n, plat_u, plat_r, plat_p, plat_y, rpi.gpst, np.array([0., 0., 0.]),
                      np.array([0., 0., 0.]), gimbal, goff, grot, rpi.dep_ang, 0.,
                      rpi.az_half_bw * 2 / DTR, rpi.el_half_bw * 2 / DTR, rpi.fs)
vx_array = np.array(vx_array)

# Get reference data
fs = rpi.fs
bwidth = rpi.bwidth
fc = rpi.fc
print('Done.')

# Generate values needed for backprojection
print('Calculating grid parameters...')
near_range_s = ranges[0] / c0
granges = ranges * np.cos(rpi.dep_ang)
up_fft_len = fft_len * settings['upsample']

# Chirp and matched filter calculations
bpj_wavelength = c0 / fc

# Get Taylor window of appropriate length and shift it to the aliased frequency of fc
taywin = int(bwidth / rpi.fs * fft_len)
taywin = taywin + 1 if taywin % 2 != 0 else taywin
taytay = np.zeros(fft_len, dtype=np.complex128)
twin_tmp = taylor(taywin, nbar=10, sll=60)
taytay[:taywin // 2] = twin_tmp[taywin // 2:]
taytay[-taywin // 2:] = twin_tmp[:taywin // 2]

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
    alias_shift = int(fft_len + (fc % (fs / 2) - fs / 2) / fs * fft_len)
    taytay = np.roll(taytay, alias_shift).astype(np.complex128)
    chirps = []
    mfilt_jax = []
    for rp in rps:
        chirp = jnp.array(waves[rp.tx_num, :])
        mfilt = chirp.conj() * taytay
        chirps.append(jnp.tile(chirp, (gap_len, 1)).T)
        mfilt_jax.append(np.tile(mfilt, (gap_len, 1)).T)
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mfilt = sdr.genMatchedFilter(0, fft_len=fft_len)
    rollback = -int(np.round(sdr[0].baseband_fc / (sdr[0].fs / fft_len)))
    # Get the model, experiment, logger set up
    if wave_config['exp_params']['model_type'] == 'InfoVAE':
        model = InfoVAE(**wave_config['model_params'])
    elif wave_config['exp_params']['model_type'] == 'WAE_MMD':
        model = WAE_MMD(**wave_config['model_params'])
    else:
        model = BetaVAE(**wave_config['model_params'])
    train_transforms = getTrainTransforms(wave_config['dataset_params']['var'])
    model.load_state_dict(torch.load('./model/inference_model.state'))
    model.eval()  # Set to inference mode
    model.to(device)

    tcdata = np.fromfile(f'{wave_config["dataset_params"]["data_path"]}/targets.cov', dtype=np.float32).reshape(
        (-1, 32, 32, 2))

    try:
        print(f'Wavemodel save file loading...')
        with open('./model/current_model_params.pic', 'rb') as f:
            generator_params = pickle.load(f)
        wave_mdl = GeneratorModel(**generator_params)
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

    chirps, mfilt_jax = reloadWaveforms(model, device, wave_mdl, active_clutter, mfilt, rollback, nsam, nr, fft_len,
                                        train_transforms(tcdata[0, ...]).unsqueeze(0).to(device), train_transforms,
                                        rps, cpi_len, taytay, bandwidth_model)

# Get all the JAX info ready for random point generation
mapped_rpg = jax.vmap(range_profile_vectorized,
                      in_axes=[None, None, None, None, 0, 0, 0, 0, 0, 0,
                               None, None, None, None, None, None, None, None])
mapped_rbi = jax.vmap(real_beam_image,
                      in_axes=[None, None, None, None, 0, 0, None, None, None, None])

# This replaces the ASI background with a custom image
bg_image = imageio.imread('/data6/Jeff_Backup/Pictures/josh.png').sum(axis=2)
bg_image = RectBivariateSpline(np.arange(bg_image.shape[0]), np.arange(bg_image.shape[1]), bg_image)(
    np.linspace(0, bg_image.shape[0], nbpj_pts[0][0]), np.linspace(0, bg_image.shape[1], nbpj_pts[0][1])) / 750
# bg_image = np.zeros(nbpj_pts[0])
# bg_image[::20, ::20] = 10

# Constant part of the radar equation
receive_power_scale = (settings['antenna_params']['transmit_power'][0] / .01 *
                       (10 ** (settings['antenna_params']['gain'][0] / 20)) ** 2
                       * bpj_wavelength ** 2 / (4 * np.pi) ** 3)
noise_level = 10 ** (settings['noise_level'] / 20) / np.sqrt(2)

# Calculate out points on the ground
grid_z = list()
bg_transforms = list()
bg_refgrid = list()
# These are only used for plotting, transforms are all we need for the actual simulation
grid_y = list()
grid_x = list()
for n in range(settings['ngrids']):
    gx, gy, gz = bg.getGrid(settings['origin'][n], settings['grid_width'][n], settings['grid_height'][n], *nbpj_pts[n],
                            bg.heading if settings['rotate_grid'] else 0)
    grid_z.append(gz)
    grid_x.append(gx)
    grid_y.append(gy)
    bg_transforms.append(bg.getGridParams(settings['origin'][n], settings['grid_width'][n], settings['grid_height'][n],
                                          nbpj_pts[n], bg.heading if settings['rotate_grid'] else 0))
    if n == 0:
        bg_refgrid.append(bg_image)
    else:
        bg_refgrid.append(bg.getRefGrid(settings['origin'][n], settings['grid_width'][n], settings['grid_height'][n],
                                        *nbpj_pts[n], bg.heading if settings['rotate_grid'] else 0))
    bg_refgrid[-1] = (bg_refgrid[-1] - bg_refgrid[-1].mean()) / bg_refgrid[-1].std()
    bg_refgrid[-1] += abs(bg_refgrid[-1].min())
    bg_refgrid[-1] *= 1e7

# Generate range/angle grid for a given position
# First, get the center of the grid for angle calcs
plat_to_grid = rpi.pos(rpi.gpst[0]) - llh2enu(*settings['origin'][0], bg.ref)
data = rpi.boresight(rpi.gpst)
sorted_data = data[np.lexsort(data.T), :]
row_mask = np.append([True], np.any(np.diff(sorted_data, axis=0), 1))
unique_boresights = sorted_data[row_mask].T
az_spread = np.linspace(rpi.pan(rpi.gpst).min(), rpi.pan(rpi.gpst).max(), nbpj_pts[0][0])
im_ranges = np.linspace(2181 - 100., 2181 + 100., nbpj_pts[0][1])
imgrid = np.hstack([azelToVec(az_spread, np.ones_like(az_spread) *
                              np.arcsin(plat_to_grid[2] / np.linalg.norm(plat_to_grid))) * g for g in im_ranges])
im_x = (imgrid[0, :] + rpi.pos(rpi.gpst[0])[0]).reshape((len(az_spread), len(im_ranges)), order='F')
im_y = (imgrid[1, :] + rpi.pos(rpi.gpst[0])[1]).reshape((len(az_spread), len(im_ranges)), order='F')
im_z = (imgrid[2, :] + rpi.pos(rpi.gpst[0])[2]).reshape((len(az_spread), len(im_ranges)), order='F')

# Get pointing vector for MIMO consolidation
ublock = np.array(
    [azelToVec(n, 0) for n in np.linspace(-ang_dist_traveled_over_cpi / 2, ang_dist_traveled_over_cpi / 2, cpi_len)]).T
fine_ucavec = np.exp(-1j * 2 * np.pi * sdr[0].fc / c0 * vx_array.dot(ublock))
array_factor = fine_ucavec.conj().T.dot(np.eye(vx_array.shape[0])).dot(fine_ucavec)[:, 0] / vx_array.shape[0]

test = None
print('Running simulation...')
pulse_pos = 0
# Data blocks for imaging
rbi_image = np.zeros((len(az_spread), len(im_ranges)), dtype=np.complex128)

nz = np.zeros(cpi_len)
tx_pattern = applyRadiationPattern(nz, np.linspace(-ang_dist_traveled_over_cpi / 2, ang_dist_traveled_over_cpi / 2,
                                                   cpi_len),
                                   nz, nz, nz, nz, rpi.az_half_bw, rpi.el_half_bw)
H = convolution_matrix(tx_pattern * array_factor, gap_len, 'valid')

# Truncated SVD for superresolution
U, eig, Vt = np.linalg.svd(H, full_matrices=False)
knee = sum(np.gradient(eig) > np.gradient(eig).mean())
eig[knee:] = 0
eig[:knee] = 1 / eig[:knee]
Hinv = Vt.T.dot(np.diag(eig)).dot(U.T)

Hinv = np.linalg.pinv(H)

H_w = np.fft.fft(tx_pattern * array_factor, gap_len)
abs_range_min = np.linalg.norm(grid_origin - rpi.pos(rpi.gpst), axis=1).min()
det_pts = list()
ex_chirps = list()

camfig, camax = plt.subplots(1, 1)
cam = Camera(camfig)
for tidx, ts in tqdm(enumerate(data_t), total=len(data_t)):
    tmp_len = len(ts)
    ex_chirps.append(np.array(chirps[0][:, 0]))
    if not sim_settings['use_sdr_waveform'] and tmp_len == gap_len:
        chirps, mfilt_jax = reloadWaveforms(model, device, wave_mdl, active_clutter, mfilt, rollback, nsam, nr, fft_len,
                                            train_transforms(tcdata[0, ...]).unsqueeze(0).to(device),
                                            train_transforms, rps, gap_len, taytay, bandwidth_model)
    # Pan and Tilt are shared by each channel, antennas are all facing the same way
    panrx = rpi.pan(ts)
    elrx = rpi.tilt(ts)
    # These are relative to the origin
    posrx = rpref.rxpos(ts)
    postx = rpref.txpos(ts)
    pt_ref = grid_origin - posrx

    beamform_data = cupy.zeros((nsam * settings['upsample'], tmp_len), dtype=np.complex128)

    for ch_idx, rp in enumerate(rps):
        posrx = rp.rxpos(ts)
        postx = rp.txpos(ts)

        pdata = mapped_rpg(bg_transforms[0][0], bg_transforms[0][1], grid_z[0], bg_refgrid[0],
                           postx, posrx, panrx, elrx, panrx, elrx,
                           bpj_wavelength, near_range_s, rp.fs, rp.az_half_bw, rp.el_half_bw,
                           ranges_sampled, settings['pts_per_tri'][0], receive_power_scale)
        if settings['ngrids'] > 1:
            for ng in range(1, settings['ngrids']):
                pdata += mapped_rpg(bg_transforms[ng][0], bg_transforms[ng][1], grid_z[ng], bg_refgrid[ng],
                                    postx, posrx, panrx, elrx, panrx, elrx,
                                    bpj_wavelength, near_range_s, rp.fs, rp.az_half_bw, rp.el_half_bw,
                                    ranges_sampled, settings['pts_per_tri'][ng], receive_power_scale)
        rtdata = cupy.array(np.array(jnp.fft.fft(pdata, fft_len, axis=1).T *
                                     chirps[ch_idx][:, :tmp_len] * mfilt_jax[ch_idx][:, :tmp_len]), dtype=np.complex128)
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
        tmp_im = jnp.sum(mapped_rbi(sig_k, im_x[angs, :], im_y[angs, :], im_z[angs, :],
                                    postx[:2, :],
                                    posrx[:2, :],
                                    panrx[gap_len // 2 - H.shape[0] // 2 - 1:gap_len // 2 + H.shape[0] // 2],
                                    near_range_s, fs * settings['upsample'], 2 * np.pi * (fc / c0)), axis=0)

        rbi_image[angs, :] += tmp_im
        camax.imshow(db(np.array(rbi_y)), clim=[-176, -7])
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
    plt.imshow(db(bg_refgrid[0]), origin='lower')
    plt.axis('tight')
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
clims = (np.median(climage) - 3 * climage.std(),
         max(np.median(climage) + 3 * climage.std(), np.max(climage) + 2))
fig = plt.figure('RBI image')
ax = fig.add_subplot(111, projection='polar')
ax.pcolormesh(az_spread, np.flip(im_ranges), climage,
              clim=clims, edgecolors='face')
ax.scatter(np.arctan2(plane_vec[0], plane_vec[1]), np.linalg.norm(plane_vec))
ax.set_ylim(0, ranges[-1])
ax.set_xlim(az_spread.min(), az_spread.max())
ax.grid(False)
plt.axis('tight')

plt.figure('RBI cartesian')
plt.imshow(climage.T, clim=clims,
           extent=[az_spread[0] / DTR, az_spread[-1] / DTR, im_ranges[0], im_ranges[-1]])
plt.scatter(np.arctan2(plane_vec[0], plane_vec[1]) / DTR, np.linalg.norm(plane_vec))
plt.axis('tight')

pfig = px.imshow(db(bg_refgrid[0]), x=np.linspace(grid_x[0].min(), grid_x[0].max(), bg_refgrid[0].shape[0]),
                 y=np.linspace(grid_y[0].min(), grid_y[0].max(), bg_refgrid[0].shape[1]),
                 color_continuous_scale=px.colors.sequential.gray, zmin=130, zmax=180)
if settings['ngrids'] > 1:
    pfig.add_scatter(x=grid_x[1].flatten(), y=grid_y[1].flatten(),
                     marker=dict(color=db(bg_refgrid[1]).flatten(), colorscale=px.colors.sequential.gray, cmin=130,
                                 cmax=180),
                     mode='markers')
pfig.show()

plt.show()

pfig = px.scatter_3d(x=grid_x[0].flatten(), y=grid_y[0].flatten(), z=grid_z[0].flatten())
pfig.add_scatter3d(x=im_x.flatten(), y=im_y.flatten(), z=im_z.flatten())
pfig.show()

plt.figure('Target Angles')
plt.plot(az_spread / DTR, db(np.sum(bg_image, axis=1)))
plt.plot(az_spread / DTR, db(np.sum(rbi_image, axis=0)))

# px.imshow(abs(rbi_image), zmin=clims.mean() - 3 * clims.std(), zmax=clims.mean() + 3 * clims.std()).show()
