import sys

sys.path.extend(['/home/jeff/repo/apache_mla', '/home/jeff/repo/data_converter', '/home/jeff/repo/simulib'])

import numpy as np
from simulation_functions import getElevation, llh2enu, genPulse, findPowerOf2, db, azelToVec, getElevationMap, enu2llh, \
    loadPostCorrectionsGPSData, loadPreCorrectionsGPSData, loadGPSData, loadGimbalData, GetAdvMatchedFilter, \
    loadMatchedFilter, calcSNR, calcPower
from cuda_kernels import genRangeWithoutIntersection, getMaxThreads, backproject, ambiguity
from grid_helper import MapEnvironment, SDREnvironment
from platform_helper import RadarPlatform, SDRPlatform
from scipy.interpolate import CubicSpline, interpn
from scipy.signal import butter, sosfilt
from PIL import Image
from scipy.signal import medfilt2d
import cupy as cupy
import cupyx.scipy.signal
from numba import cuda, njit
from numba.cuda.random import create_xoroshiro128p_states, init_xoroshiro128p_states
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.signal.windows import taylor
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from celluloid import Camera
from tqdm import tqdm
from SDRParsing import SDRParse, load, getModeValues
from SDRWriting import SDRWrite, SHRT_MAX
import pickle
import json
from itertools import product

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'
mempool = cupy.get_default_memory_pool()


c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254

command_line_args = sys.argv

print('Getting settings from JSON file...')
with open('./settings.json', 'r') as sf:
    settings = json.load(sf)
    grid_center = settings['grid_center']
    nchan = len(settings['tx']) * len(settings['rx'])
    nrx = len(settings['rx'])
    nants = len(settings['antennas'])
    fs = settings['sample_frequency']
    band_frequency = settings['tx'][0]['center_frequency']

if len(command_line_args) > 1:
    bg_file = command_line_args[1]
else:
    bg_file = '/data5/SAR_DATA/2022/03032022/SAR_03032022_155926.sar'
# bg_file = '/data5/SAR_DATA/2022/03282022/SAR_03282022_082824.sar'

print(f'Loading SAR file {bg_file}...')
sdr = load(bg_file)

grid_center[2] = getElevation((grid_center[0], grid_center[1]))

# Generate the background for simulation
print('Generating environment...', end='')
# bg = MapEnvironment(grid_center, extent=(500, 500), npts_background=(150, 150))
bg = SDREnvironment(sdr)

# Set the channel specific data if not using custom params
if settings['use_sdr_waveform']:
    grid_center = bg.origin
    ants = []
    for idx, ant in enumerate(sdr.port):
        ants.append({'offset': [ant.x, ant.y, ant.z], 'az_beamwidth': sdr.ant[ant.assoc_ant].az_bw / DTR,
                     'el_beamwidth': sdr.ant[ant.assoc_ant].el_bw / DTR,
                     'dopp_beamwidth': sdr.ant[ant.assoc_ant].az_bw / DTR})
    settings['antennas'] = ants
    settings['tx'] = [{'pulse_length_percent': sdr[settings['channel']].xml['Pulse_Length_Percent'] / 100,
                           'center_frequency': sdr[settings['channel']].fc,
                       'bandwidth': sdr[settings['channel']].bw}]
    settings['prf'] = sdr[settings['channel']].prf
    nchan = 1
else:
    if np.any([n['custom_waveform'] for n in settings['tx']]):
        # Import all the stuff needed for waveform generation
        import keras.models
        from keras.models import load_model
        from keras.utils import custom_object_scope
        from tensorflow.keras.optimizers import Adam, Adadelta
        from wave_train import genTargetPSD, genModel, opt_loss, compileWaveformData, getWaveFromData

        print('Using custom waveform from DNN model.')
        with open('/home/jeff/repo/apache_mla/model/model_params.pic', 'rb') as f:
            model_params = pickle.load(f)
rx_to_chan = {}
rx2chanconv = 0
for rx in settings['rx']:
    if rx not in rx_to_chan:
        rx_to_chan[rx] = rx2chanconv
        rx2chanconv += 1

print('Done.')

# Generate a platform
print('Generating platform...', end='')
rps = []

settings['nsam'] = 0
rx_array = []
if settings['use_sdr_gps']:
    fdelay = 55
    rps.append(SDRPlatform(sdr, bg.ref, channel=settings['channel']))
    rps[0].rx_num = 0
    rps[0].tx_num = 0
    settings['tx'][0]['nr'] = \
        rps[-1].calcPulseLength(fdelay, settings['tx'][0]['pulse_length_percent'], use_tac=True)
    settings['ranges'] = rps[-1].calcRangeBins(fdelay, settings['upsample'], settings['tx'][0]['pulse_length_percent'])
    settings['nsam'] = rps[0].calcNumSamples(fdelay, settings['tx'][0]['pulse_length_percent'])

else:
    start_loc = llh2enu(sdr.xml['Flight_Line']['Start_Latitude_D'], sdr.xml['Flight_Line']['Start_Longitude_D'],
                        sdr.xml['Flight_Line']['Flight_Line_Altitude_M'], bg.ref)
    stop_loc = llh2enu(sdr.xml['Flight_Line']['Stop_Latitude_D'], sdr.xml['Flight_Line']['Stop_Longitude_D'],
                       sdr.xml['Flight_Line']['Flight_Line_Altitude_M'], bg.ref)
    course_heading = np.arctan2(start_loc[0] - stop_loc[0], start_loc[1] - stop_loc[1]) + np.pi
    gps_times = sdr.gps_data.index.values
    nt = len(gps_times)
    for tx_ant, rx_n in np.dstack(np.meshgrid(*[settings['tx'], settings['rx']], indexing='ij')).reshape(-1, 2):
        rps.append(RadarPlatform(e=np.linspace(start_loc[0], stop_loc[0], nt),
                                 n=np.linspace(start_loc[1], stop_loc[1], nt),
                                 u=np.zeros(nt) + settings['collect_alt'],
                                 r=np.zeros(nt), p=np.zeros(nt), y=np.zeros(nt) + course_heading, t=gps_times,
                                 gimbal=np.zeros((nt, 2)) + np.array(
                                     [sdr.gimbal['pan'].mean(), sdr.gimbal['tilt'].mean()]),
                                 dep_angle=settings['gimbal_dep_angle'],
                                 el_bw=settings['antennas'][tx_ant['antenna']]['el_beamwidth'],
                                 az_bw=settings['antennas'][tx_ant['antenna']]['az_beamwidth'],
                                 gimbal_offset=settings['gimbal_offset'],
                                 gimbal_rotations=settings['gimbal_rotations'],
                                 rx_offset=settings['antennas'][rx_n]['offset'],
                                 tx_offset=settings['antennas'][tx_ant['antenna']]['offset'], tx_num=tx_ant['antenna'],
                                 rx_num=rx_n, wavenumber=tx_ant['antenna']))
        plat_height = rps[-1].pos(rps[-1].gpst)[2, :].mean()
        if tx_ant['custom_waveform']:
            tx_ant['nr'] = model_params['mdl_fft']
        else:
            tx_ant['nr'] = rps[-1].calcPulseLength(plat_height, tx_ant['pulse_length_percent'], use_tac=True)
        settings['nsam'] = max(settings['nsam'], rps[-1].calcNumSamples(plat_height,
                                                                        tx_ant['pulse_length_percent']))
        rx_array.append(np.array(settings['antennas'][tx_ant['antenna']]['offset']) + np.array(settings['antennas'][rx_n]['offset']))
    settings['ranges'] = rps[-1].calcRangeBins(rps[-1].pos(rps[-1].gpst)[2, :].mean(), settings['upsample'],
                                               max([n['pulse_length_percent'] for n in settings['tx']]))
print('Done.')

print('Calculating grid parameters...', end='')
# General calculations for slant ranges, etc.
fft_len = findPowerOf2(settings['nsam'] + max([ch['nr'] for ch in settings['tx']]))
up_fft_len = fft_len * settings['upsample']
up_nsam = settings['nsam'] * settings['upsample']
sim_up_fft_len = fft_len * settings['sim_upsample']
sim_up_nsam = settings['nsam'] * settings['sim_upsample']
sim_bpj_decimation = settings['sim_upsample'] // settings['upsample']

range_to_center = lambda t: np.linalg.norm(rps[0].pos(t) - llh2enu(*grid_center, bg.ref))
print('Done.')

for ch_num, ch in enumerate(settings['tx']):
    if sdr[ch_num].xml['Offset_Video_Enabled'] == 'True' and settings['use_sdr_waveform']:
        offset_hz = sdr[ch_num].xml['DC_Offset_MHz'] * 1e6
        ch['bpj_wavelength'] = c0 / (ch['center_frequency'] - ch['bandwidth'] / 2 - offset_hz)
        # settings['bpj_wavelength'] = c0 / (ch['center_frequency'] - bwidth / 2 - offset_hz)
        # offset_shift = int((offset_hz + bwidth / 2) / (1 / fft_len * fs) * upsample)
    else:
        offset_hz = ch['center_frequency'] - band_frequency - ch['bandwidth'] / 2
        ch['bpj_wavelength'] = c0 / ch['center_frequency']

wf_data = None
if not settings['use_sdr_waveform']:
    if np.any([n['custom_waveform'] for n in settings['tx']]):
        mdl_fft = model_params['mdl_fft']
        n_conv_filters = model_params['n_conv_filters']
        kernel_sz = model_params['kernel_sz']
        mdl_bin_bw = int(model_params['bandwidth'] / (fs / mdl_fft))
        mdl = genModel(mdl_bin_bw, n_conv_filters, kernel_sz, mdl_fft)
        mdl.load_weights('/home/jeff/repo/apache_mla/model/model')

        single_pulse = genPulse(np.linspace(0, 1, 10), np.linspace(0, 1, 10), settings['tx'][0]['nr'], settings['sample_frequency'],
                 settings['tx'][0]['center_frequency'], settings['tx'][0]['bandwidth'])

        plat_height = 0 if settings['use_sdr_gps'] else rps[0].pos(rps[0].gpst)[2].mean()
        midrange = (rps[0].calcRanges(plat_height)[0] + rps[0].calcRanges(plat_height)[1]) / 2

        target_psd = db(np.fft.fft(single_pulse, mdl_fft) * \
                     np.fft.fft(np.exp(1j * 2 * np.pi * settings['tx'][0]['center_frequency'] / c0 *
                                       midrange * 2 * np.arange(mdl_fft) / settings['sample_frequency'])))
        with open('./test_target.pic', 'rb') as f:
            tpsd = pickle.load(f)
        target_psd = tpsd['psd'][:, 0]

        # Generate model data for waveform
        target_data, clutter_data, cd_mu, cd_std, td_mu, td_std = \
            compileWaveformData(sdr, mdl_fft, settings['cpi_len'], settings['tx'][0]['bandwidth'],
                                  settings['tx'][0]['center_frequency'],
                                  settings['ranges'][0],
                                  settings['ranges'][-1], fs, mdl_bin_bw, model_params['td_mu'],
                                  model_params['td_std'], model_params['cd_mu'], model_params['cd_std'],
                                  target_signal=target_psd)

        # Get waveforms
        wf_data = getWaveFromData(mdl, target_data, clutter_data, cd_mu, cd_std,
                                                  2, 2, mdl_fft, mdl_bin_bw)

# Model for waveform simulation
for ch_num, ch in enumerate(settings['tx']):
    if settings['use_sdr_waveform']:
        mchirp = np.mean(sdr.getPulses(sdr[ch_num].cal_num, 0, is_cal=True), axis=1)
        ch['chirp'] = np.fft.fft(mchirp, sim_up_fft_len)
        ch['mchirp'] = mchirp
        waveform = sdr[ch_num].ref_chirp
    else:
        if ch['custom_waveform']:
            mchirp = np.zeros((settings['cpi_len'], settings['nsam']), dtype=np.complex128)
            '''reflection_freq = -(c0 / ch['bpj_wavelength'] - fs * np.round(c0 / ch['bpj_wavelength'] / fs))
            mixup = np.fft.fft(np.exp(-1j * 2 * np.pi * ch['center_frequency'] *
                                                        np.arange(sim_up_fft_len) / (fs / sim_up_fft_len)),
                               ch['nr']) / ch['nr']'''
            wf_ifft = np.fft.ifft(np.fft.fft(np.fft.ifft(wf_data[:, ch_num, :], axis=1), ch['nr'], axis=1), axis=1)
            sp = np.sqrt(np.mean(abs(wf_ifft) ** 2, axis=1) / ch['power'])
            waveform = wf_ifft / sp[:, None]
            mchirp[:, 5:5 + ch['nr']] = waveform
            # Passband filter the thing
            # sos = butter(10, ch['bandwidth'] / 2, 'lowpass', fs=fs, output='sos')
            # mchirp = sosfilt(sos, mchirp)
            '''plt.figure('wfdata'); plt.plot(wf_ifft.real)
            plt.figure('wfspectrum'); plt.plot(np.fft.fftfreq(len(wf_ifft), 1/fs),
                                               db(np.fft.fft(wf_ifft))); plt.plot(np.fft.fftfreq(len(wf_ifft), 1/fs), db(mixup))
            plt.figure(); plt.plot(np.fft.fftfreq(len(mchirp), 1/fs), db(np.fft.fft(mchirp)))
            plt.show()'''
        else:
            mchirp = np.zeros(settings['nsam'], dtype=np.complex128)
            direction = np.linspace(0, 1, 10) if ch_num == 0 else np.linspace(1, 0, 10)
            waveform = genPulse(np.linspace(0, 1, 10), direction, ch['nr'], fs,
                                              ch['center_frequency'], ch['bandwidth']) * np.sqrt(ch['power']) * 1000
            mchirp[5:5 + ch['nr']] = waveform
        ch['chirp'] = np.fft.fft(mchirp, sim_up_fft_len, axis=1)
        ch['mchirp'] = mchirp

# Chirp and matched filter calculations
for ch_num, ch in enumerate(settings['tx']):
    print('Calculating Taylor window.')
    taywin = int(ch['bandwidth'] / fs * sim_up_fft_len)
    taywin = taywin + 1 if taywin % 2 != 0 else taywin
    taytay = taylor(taywin, sll=80)
    tmp = np.zeros(sim_up_fft_len, dtype=np.complex128)
    tmp[:taywin // 2] = taytay[-taywin // 2:]
    tmp[-taywin // 2:] = taytay[:taywin // 2]
    taytay = np.fft.ifft(tmp)
    # tayd = np.fft.fftshift(taylor(cpi_len))
    # taydopp = np.fft.fftshift(np.ones((up_nsam, 1)).dot(tayd.reshape(1, -1)), axes=1)

    print('Calculating matched filter.')
    nco_lambda = c0 / band_frequency
    reflection_freq = c0 / ch['bpj_wavelength'] - fs * np.round(c0 / ch['bpj_wavelength'] / fs)
    if reflection_freq != 0:
        overshot = int((abs(reflection_freq) - ch['bandwidth'] / 2) / (fs / sim_up_fft_len))
        tayshifted = taytay * np.exp(-1j * 2 * np.pi * reflection_freq * np.arange(sim_up_fft_len) * 1 / fs)
        mfilt = ch['chirp'].conj() * np.fft.fft(tayshifted, sim_up_fft_len)[None, :]
        if reflection_freq < 0:
            mfilt[sim_up_fft_len // 2 + overshot:] = 0
        elif reflection_freq > 0:
            mfilt[:sim_up_fft_len // 2 - overshot] = 0
    else:
        mfilt = ch['chirp'].conj() * np.fft.fft(taytay, sim_up_fft_len)[None, :]

    ch['mfilt'] = cupy.array(mfilt[:, ::settings['sim_upsample']].T,
                             dtype=np.complex128)

    ch['chirp'] = cupy.array(ch['chirp'].T, dtype=np.complex128)

dopp_line = ((c0 + rps[0].vel(rps[0].gpst).mean(axis=1).dot(
        azelToVec(rps[0].az_half_bw, rps[0].el_half_bw))) / c0 * settings['tx'][0]['center_frequency'] -
             settings['tx'][0]['center_frequency']) % (settings['prf'] / 2)
dopp_line += -settings['prf'] / 2 if dopp_line > settings['prf'] / 4 else \
    settings['prf'] / 2 if dopp_line < -settings['prf'] / 4 else 0
taywin = int(abs(dopp_line) / settings['prf'] * settings['cpi_len'])
taywin = taywin + 1 if taywin % 2 != 0 else taywin
taywin = min(taywin * 2, settings['cpi_len'])
tayl = taylor(taywin, nbar=10, sll=120)
tayd = np.zeros(settings['cpi_len'])
tayd[:taywin // 2] = tayl[taywin // 2:]
tayd[-taywin // 2:] = tayl[:taywin // 2]
taydopp = np.ones((up_nsam, 1)).dot(tayd.reshape(1, -1))

# Calculate noise for each channel set
noise = []
for rp_idx, rp in enumerate(rps):
    tx = settings['tx'][rp.tx_num]
    tx_ant = settings['antennas'][settings['tx'][rp.tx_num]['antenna']]
    snr_vals = np.array([calcSNR(tx['power'], 10**(tx_ant['gain'] / 20), .1 * DTR, .1 * DTR,
                                 c0 / tx['center_frequency'], tx['nr'] / fs, tx['bandwidth'],
                                 settings['cpi_len'], r) for r in settings['ranges']])
    noise.append(np.array([calcPower(tx['power'], 10**(tx_ant['gain'] / 20), .1 * DTR, .1 * DTR,
                                     c0 / tx['center_frequency'], tx['nr'] / fs, tx['bandwidth'],
                                     settings['cpi_len'], r) for r in settings['ranges']]) / 10**(snr_vals / 20))

bg.resample(grid_center, settings['grid_width'], settings['grid_height'], (settings['nbpj_pts'], settings['nbpj_pts']))
# gx, gy, gz = bg.getGrid(bg.origin, settings['grid_width'], settings['grid_height'], (settings['nbpj_pts'], settings['nbpj_pts']))
gx, gy, gz = bg.getGrid()
ngz_gpu = cupy.array(bg.getGrid()[2], dtype=np.float64)
ng = np.zeros(bg.shape)

ng = Image.open('/home/jeff/Pictures/c130ssar.png').resize((50, 50), Image.ANTIALIAS)
ng = np.linalg.norm(np.array(ng), axis=2) * 1e6
bg._refgrid[20:ng.shape[0] + 20, 20:ng.shape[1] + 20] = ng
bg._refgrid[150:ng.shape[0] + 150, 60:ng.shape[1] + 60] = ng
# bg._refgrid = ng
ref_coef_gpu = cupy.array(bg.refgrid, dtype=np.float64)
rbins_gpu = cupy.array(settings['ranges'], dtype=np.float64)
rmat_gpu = cupy.array(bg.transforms[0], dtype=np.float64)
shift_gpu = cupy.array(bg.transforms[1], dtype=np.float64)

if settings['debug']:
    pts_debug = cupy.zeros((3, *bg.shape), dtype=np.float64)
    angs_debug = cupy.zeros((3, *bg.shape), dtype=np.float64)
    # pts_debug = cupy.zeros((settings['nbpj_pts'], 3), dtype=np.float64)
    # angs_debug = cupy.zeros((settings['nbpj_pts'], 3), dtype=np.float64)
else:
    pts_debug = cupy.zeros((1, 1, 1), dtype=np.float64)
    angs_debug = cupy.zeros((1, 1, 1), dtype=np.float64)
test = None

# GPU device calculations
threads_per_block = getMaxThreads()
bpg_ranges = (bg.refgrid.shape[0] // threads_per_block[0] + 1,
              bg.refgrid.shape[1] // threads_per_block[1] + 1)

rng_states = create_xoroshiro128p_states(bpg_ranges[0] * bpg_ranges[1] * threads_per_block[0] * threads_per_block[1],
                                         seed=10)

fig, ax = plt.subplots(2, 1)

cam = Camera(fig)

# Run through loop to get data simulated
if settings['use_sdr_waveform']:
    data_t = sdr[settings['channel']].pulse_time[0] + np.arange(sdr[settings['channel']].nframes) / settings['prf']
else:
    data_t = np.arange(rps[0].gpst[0], rps[0].gpst[0] + settings['collect_time'], 1 / settings['prf'])
print('Simulating...')
pulse_pos = 0
min_range = settings['nsam']
max_range = 0

a_theta = np.exp(1j * 2 * np.pi * settings['tx'][0]['center_frequency'] / c0 *
                 np.array(rx_array).dot(azelToVec(0, rps[0].dep_ang)))
for tidx in tqdm(np.arange(0, len(data_t), settings['cpi_len'])):
    init_xoroshiro128p_states(rng_states, seed=10)
    ts = data_t[tidx + np.arange(min(settings['cpi_len'], len(data_t) - tidx))]
    tmp_len = len(ts)
    rtdata = cupy.zeros((sim_up_fft_len, tmp_len, nrx), dtype=np.complex128)
    debug_flag = False
    for rp_idx, rp in enumerate(rps):
        debug_flag = settings['debug'] if ts[0] < rp.gpst.mean() <= ts[-1] else False
        panrx_gpu = cupy.array(rp.pan(ts), dtype=np.float64)
        elrx_gpu = cupy.array(rp.tilt(ts), dtype=np.float64)
        posrx_gpu = cupy.array(rp.rxpos(ts), dtype=np.float64)
        postx_gpu = cupy.array(rp.txpos(ts), dtype=np.float64)
        data_r = cupy.zeros((sim_up_nsam, tmp_len), dtype=np.float64)
        data_i = cupy.zeros((sim_up_nsam, tmp_len), dtype=np.float64)
        genRangeWithoutIntersection[bpg_ranges, threads_per_block](rmat_gpu, shift_gpu, ngz_gpu, ref_coef_gpu,
                                                                   postx_gpu, posrx_gpu, panrx_gpu, elrx_gpu,
                                                                   panrx_gpu, elrx_gpu, data_r, data_i, rng_states,
                                                                   pts_debug, angs_debug,
                                                                   settings['tx'][rp.tx_num]['bpj_wavelength'],
                                                                   settings['ranges'][0] / c0,
                                                                   rp.fs * settings['sim_upsample'],
                                                                   rp.az_half_bw,
                                                                   rp.el_half_bw, settings['pts_per_tri'],
                                                                   debug_flag)
        cuda.synchronize()
        data_r[np.isnan(data_r)] = 0
        data_i[np.isnan(data_i)] = 0

        # Create data using chirp
        # Inject noise into the system based on range
        dframe = cupy.fft.fft(data_r + 1j * data_i, sim_up_fft_len, axis=0) * \
                               settings['tx'][rp.tx_num]['chirp'][:, :tmp_len]
        dframe += cupy.fft.fft(cupy.array(np.random.randn(sim_up_nsam, tmp_len) * noise[rp_idx][:, None], dtype=np.float64) + \
                               1j * cupy.array(np.random.randn(sim_up_nsam, tmp_len) * noise[rp_idx][:, None],
                                               dtype=np.float64), sim_up_fft_len, axis=0)
        rtdata[:, :, rx_to_chan[rp.rx_num]] += dframe
        cuda.synchronize()

    rcdata = np.zeros((sim_up_nsam, tmp_len, nchan), dtype=np.complex128)
    for rp_idx, rp in enumerate(rps):
        # Decimate to fit backprojection
        rcdata[:, :, rp_idx] = cupy.fft.ifft(rtdata[:, :, rx_to_chan[rp.rx_num]] *
                               settings['tx'][rp.tx_num]['mfilt'][:, :tmp_len],
                               axis=0)[:sim_up_nsam:sim_bpj_decimation, :].get()
        cuda.synchronize()

    if ts[0] < data_t.mean() < ts[-1]:
        center_map = rcdata.copy()
        pts_calc = pts_debug.get()
        ang_calc = angs_debug.get()
        ts_calc = ts.mean()

    # For multichannel data, do some beamforming
    if rcdata.shape[2] > 1:
        # Get the center of the grid
        gcs = np.array([*bg.getPos(bg.shape[0] // 2, bg.shape[1] // 2), gz.mean()])
        pvec = rps[0].pos(ts[0]) - gcs
        dopp_line = ((c0 + rps[0].vel(ts[0]).dot(
            azelToVec(np.arctan2(pvec[0], pvec[1]), np.arcsin(pvec[2] / np.linalg.norm(pvec))))) / c0 *
                     settings['tx'][0]['center_frequency'] -
                     settings['tx'][0]['center_frequency']) % (settings['prf'] / 2)
        dopp_line += -settings['prf'] / 2 if dopp_line > settings['prf'] / 4 else \
            settings['prf'] / 2 if dopp_line < -settings['prf'] / 4 else 0
        tayshift = np.roll(taydopp, (-int(dopp_line / settings['prf'] * settings['cpi_len']), 0))

        # Capon MVDR beamformer in the center direction
        Rcc = np.linalg.pinv(np.cov(np.mean(rcdata, axis=1).T))
        a_theta = np.exp(1j * 2 * np.pi * settings['tx'][0]['center_frequency'] / c0 *
                         np.array(rx_array).dot(azelToVec(np.arctan2(pvec[0], pvec[1]), np.arcsin(pvec[2] / np.linalg.norm(pvec)))))
        w = Rcc.dot(a_theta) / (a_theta.conj().T.dot(Rcc).dot(a_theta))
        seg_data = rcdata.dot(w)
    else:
        seg_data = rcdata[:, :, 0]

    straight_beam = np.fft.fft(seg_data, axis=1, n=settings['cpi_len'] * 4)

    ax[0].imshow(db(straight_beam[5700:13200]), origin='lower',
                 extent=[-1, 1, settings['ranges'][5700], settings['ranges'][13200]])
    ax[0].axis('tight')
    ax[0].hlines(range_to_center(ts.mean()), -1, 1)
    ax[0].text(0.5, 1.01, f'Frame {tidx:.2f}', transform=ax[0].transAxes)
    ax[1].imshow(db(seg_data[5700:13200]), origin='lower',
                 extent=[-1, 1, settings['ranges'][5700], settings['ranges'][13200]])
    ax[1].axis('tight')
    cam.snap()

    del panrx_gpu
    del postx_gpu
    del posrx_gpu
    del elrx_gpu
    del data_r
    del data_i
    del rtdata
    mempool.free_all_blocks()

del rbins_gpu
del rmat_gpu
del shift_gpu
del ngz_gpu
for tx_it in settings['tx']:
    tx_it['chirp'] = tx_it['chirp'].get()
    tx_it['mfilt'] = tx_it['mfilt'].get()
mempool.free_all_blocks()

if settings['debug']:
    print('Generating debug plots...')
    ngx, ngy, ngz = bg.getGrid()
    flight = rp.pos(rp.gpst)
    fig = px.scatter(x=ngx.flatten(), y=ngy.flatten(), color=db(bg.refgrid.flatten()), range_color=[0, 220])
    fig.add_scatter(x=gx.flatten(), y=gy.flatten(), mode='markers')
    fig.show()

    pts_ext = pts_calc + rps[0].pos(ts_calc)[:, None, None]
    fig = px.scatter(x=ngx.flatten(), y=ngy.flatten(), color=db(bg.refgrid.flatten()), opacity=.1)
    fig.add_scatter(x=pts_ext[0, :, :].flatten(), y=pts_ext[1, :, :].flatten(),
                    marker=dict(color=db(ang_calc[2, ...].flatten() * 1e10)), mode='markers')
    fig.show()

    animation = cam.animate(interval=100)

    if center_map.shape[2] > 1:
        Rcc = np.linalg.pinv(np.cov(np.mean(center_map, axis=1).T))
        w = Rcc.dot(a_theta) / (a_theta.conj().T.dot(Rcc).dot(a_theta))
        seg_data = center_map.dot(w)
    else:
        seg_data = center_map[:, :, 0]
    plt.figure('Centermap')
    plt.imshow(db(np.fft.fftshift(np.fft.fft(seg_data[5700:13200], axis=1), axes=1)))
    plt.axis('tight')

    plt.figure('Waveforms')
    freqs = np.fft.fftshift(np.fft.fftfreq(up_fft_len, 1 / fs))
    for idx, tx in enumerate(settings['tx']):
        plt.subplot(3, 1, idx + 1)
        plt.title(f'Tx_{idx}')
        plt.plot(freqs, np.fft.fftshift(db(tx['chirp'])), c='blue')
        plt.plot(freqs, np.fft.fftshift(db(tx['mfilt'])), c='orange')
        plt.subplot(3, 1, 3)
        plt.plot(np.fft.fftshift(db(np.fft.ifft(tx['chirp'] * tx['mfilt']))))

    if settings['tx'][0]['custom_waveform']:
        plt.figure('Target PSD')
        plt.plot(db(target_psd))

        up_tpsd = np.fft.ifft(tpsd['psd'][:, 0])
        plt.figure('Target Overlay')
        for idx, tx in enumerate(settings['tx']):
            plt.subplot(2, 1, idx + 1)
            plt.title(f'Tx_{idx}')
            plt.plot(freqs, np.fft.fftshift(db(np.fft.fft(up_tpsd, len(tx['chirp'])))), c='orange')
            plt.plot(freqs, np.fft.fftshift(db(tx['chirp'])), c='blue')
    plt.show()

'''test = np.fft.fft(center_map, 4096, axis=0)
with open('test_target.pic', 'wb') as f:
    pickle.dump({'psd': test[:, :, 0]}, f)'''