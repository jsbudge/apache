import pickle
from glob import glob
import sys
import numpy as np
import torch
from apache_helper import ApachePlatform
from models import Encoder
from simulib.simulation_functions import llh2enu, db, azelToVec, genPulse
from simulib.cuda_kernels import getMaxThreads, backproject, applyRadiationPatternCPU
from simulib.cuda_mesh_kernels import genRangeProfileFromMesh
from simulib.mimo_functions import genChirpAndMatchedFilters, genChannels, genSimPulseData
from simulib.grid_helper import SDREnvironment
import cupy as cupy
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from data_converter.SDRParsing import load
import yaml
from scipy.signal import sawtooth
from scipy.linalg import convolution_matrix
from celluloid import Camera
from waveform_model import GeneratorModel
import contextlib


def reloadWaveforms(a_wave_mdl, pulse_data, a_nr, a_fft_len, tc_d, a_rps, a_bwidth, mu, std):
    fn_waves = a_wave_mdl.full_forward(pulse_data, tc_d, a_nr, a_bwidth, mu, std)
    # Calculate out the center frequency for shifting
    wv = np.fft.fftshift(fn_waves, axes=1)
    if a_wave_mdl.fft_sz != a_fft_len:
        new_waves = np.zeros((fn_waves.shape[0], a_fft_len), dtype=np.complex128)
        new_waves[:, :a_wave_mdl.fft_sz // 2] = fn_waves[:, :a_wave_mdl.fft_sz // 2]
        new_waves[:, -a_wave_mdl.fft_sz // 2:] = fn_waves[:, -a_wave_mdl.fft_sz // 2:]
        fn_waves = new_waves
    nfc = 1e9 + (wv.shape[1] // 2 - np.arange(wv.shape[1])[db(wv[0]) > db(wv[0]).max() - 50]).mean() * fs / wv.shape[1]
    _, fn_chirps, fn_mfilts = genChirpAndMatchedFilters(fn_waves, a_rps, a_bwidth, fs, nfc, a_fft_len, use_window=False)
    return fn_chirps, fn_mfilts


pio.renderers.default = 'browser'

c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
GPS_UPDATE_RATE_HZ = 100

if __name__ == '__main__':
    # Load all the settings files
    with open('./wave_simulator.yaml') as y:
        settings = yaml.safe_load(y.read())
    with open('./vae_config.yaml', 'r') as file:
        try:
            wave_config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    nbpj_pts = (
        int(settings['grid_height'] * settings['pts_per_m']), int(settings['grid_width'] * settings['pts_per_m']))
    sim_settings = settings['simulation_params']

    print('Loading SDR file...')
    # This pickle should have an ASI file attached and use_jump_correction=False
    sdr = load(settings['bg_file'], import_pickle=False, export_pickle=False, use_jump_correction=False)
    bg = SDREnvironment(sdr)
    ref_llh = bg.ref

    plane_pos = llh2enu(40.138052, -111.660027, 1365, bg.ref)

    # Generate a platform
    print('Generating platform...', end='')
    # Run directly at the plane from the south
    grid_origin = plane_pos  # llh2enu(*bg.origin, bg.ref)
    # full_scan = int(sim_settings['az_bw'] / sim_settings['scan_rate'] * sim_settings['prf'])
    # full_scan -= 0 if full_scan % 2 == 0 else 1
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
    '''gim_pan = (sawtooth(np.pi * sim_settings['scan_rate'] / sim_settings['scan_angle'] *
                        np.arange(ngpssam) / GPS_UPDATE_RATE_HZ, .5)
               * sim_settings['scan_angle'] / 2 * DTR)'''
    gim_pan = (sawtooth(np.pi * sim_settings['scan_rate'] / sim_settings['scan_angle'] *
                        np.arange(ngpssam) / GPS_UPDATE_RATE_HZ, .5)
               * sim_settings['scan_angle'] / 2 * DTR)
    gim_el = np.zeros_like(gim_pan) + np.arccos(req_alt / req_slant_range)
    goff = np.array(
        [wave_config['apache_params']['phase_center_offset_m'], 0., wave_config['apache_params']['wheel_height_m']])
    grot = np.array([0., 0., 0.])
    wave_fft_len = wave_config['settings']['fft_len']
    cpi_len = settings['cpi_len']

    rpref, rps, vx_array = genChannels(settings['antenna_params']['n_rx'], settings['antenna_params']['n_tx'],
                                       settings['antenna_params']['tx_pos'], settings['antenna_params']['rx_pos'],
                                       e, n, u, r, p, y, t,
                                       np.array([gim_pan, gim_el]).T, goff, grot,
                                       req_dep_ang / DTR, wave_config['apache_params']['az_min_bw'] / 2,
                                       wave_config['apache_params']['el_min_bw'] / 2, 2e9, ApachePlatform,
                                       dict(params=wave_config['apache_params']))

    nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = (
        rpref.getRadarParams(u.mean(), settings['plp'], settings['upsample']))
    print(f'Plane slant range is {np.linalg.norm(plane_pos - rpref.rxpos(rpref.gpst), axis=1).mean()}')
    rbins_gpu = cupy.array(ranges, dtype=np.float64)
    near_range_s = np.float64(near_range_s)

    # Get reference data
    fs = rpref.fs
    bwidth = settings['bandwidth']
    fc = settings['fc']
    print('Done.')

    # Generate values needed for backprojection
    print('Calculating grid parameters...')

    # Chirp and matched filter calculations
    bpj_wavelength = np.float64(c0 / fc)

    # Run through loop to get data simulated
    gap_len = 256
    if settings['simulation_params']['use_sdr_gps']:
        data_t = sdr[settings['channel']].pulse_time
        idx_t = sdr[settings['channel']].frame_num
    else:
        data_t = rpref.getValidPulseTimings(settings['simulation_params']['prf'], nr / TAC, cpi_len, as_blocks=True)
        gap_len = len(data_t[0])
        data_t = np.concatenate(data_t)

    ang_dist_traveled_over_cpi = cpi_len / sim_settings['prf'] * sim_settings['scan_rate'] * DTR

    # Chirps and Mfilts for each channel
    waves = np.array([np.fft.fft(genPulse(np.linspace(0, 1, 10),
                                          np.linspace(0, 1, 10), nr, fs, fc - bwidth / 2, bwidth), fft_len),
                      np.fft.fft(genPulse(np.linspace(0, 1, 10),
                                          np.linspace(1, 0, 10), nr, fs, fc - bwidth / 2, bwidth), fft_len)]
                     ) / 1e5
    _, chirps, mfilts = genChirpAndMatchedFilters(waves, rps, bwidth, fs, fc, fft_len)

    try:
        target_target = int(sys.argv[1])
    except IndexError:
        target_target = sim_settings['target_target']
    if not sim_settings['use_sdr_waveform']:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        mfilt = sdr.genMatchedFilter(0, fft_len=fft_len)
        print('Setting up wavemodel...')
        # Get the model, experiment, logger set up
        decoder = Encoder(**wave_config['exp_params']['model_params'], fft_len=wave_config['settings']['fft_len'],
                          params=wave_config['exp_params'])
        print('Loading decoder...')
        try:
            decoder.load_state_dict(torch.load('./model/inference_model.state'))
        except RuntimeError:
            raise IOError('Decoder save file broken somehow.')
        with open('./model/current_model_params.pic', 'rb') as f:
            generator_params = pickle.load(f)
        wave_mdl = GeneratorModel(**generator_params, decoder=decoder)
        wave_mdl.load_state_dict(torch.load(generator_params['state_file']))
        print('Wavemodel loaded.')

        active_clutter = np.fft.fft(sdr.getPulses(sdr[0].frame_num[:wave_config['settings']['cpi_len']], 0)[1].T,
                                    wave_mdl.fft_sz, axis=1)
        bandwidth_model = torch.tensor(settings['bandwidth'], device=wave_mdl.device)
        tsdata = np.fromfile('/home/jeff/repo/apache/data/targets.enc',
                             dtype=np.float32).reshape((-1, wave_config['target_exp_params']['model_params']['latent_dim'])
                                                       )[target_target, ...]

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

    # Calculate out points on the ground
    gx, gy, gz = bg.getGrid(settings['origin'], settings['grid_width'], settings['grid_height'], *nbpj_pts,
                            bg.heading if settings['rotate_grid'] else 0)
    refgrid = bg.getRefGrid(settings['origin'], settings['grid_width'], settings['grid_height'],
                            *nbpj_pts, bg.heading if settings['rotate_grid'] else 0)

    # Generate range/angle grid for a given position

    imx_gpu = cupy.array(gx, dtype=np.float64)
    imy_gpu = cupy.array(gy, dtype=np.float64)
    imz_gpu = cupy.array(gz, dtype=np.float64)

    # GPU device calculations
    threads_per_block = getMaxThreads()
    bpg_bpj = (max(1, (nbpj_pts[0]) // threads_per_block[0] + 1), (nbpj_pts[1]) // threads_per_block[1] + 1)

    # Get pointing vector for MIMO consolidation
    ublock = np.array(
        [azelToVec(n, 0) for n in
         np.linspace(-ang_dist_traveled_over_cpi / 2, ang_dist_traveled_over_cpi / 2, cpi_len)]).T
    uhat = azelToVec(np.pi / 2, 0)
    avec = np.exp(-1j * 2 * np.pi * sdr[0].fc / c0 * vx_array.dot(uhat))
    fine_ucavec = np.exp(-1j * 2 * np.pi * sdr[0].fc / c0 * vx_array.dot(ublock))
    array_factor = fine_ucavec.conj().T.dot(np.eye(vx_array.shape[0])).dot(fine_ucavec)[:, 0] / vx_array.shape[0]

    test = None
    print('Running simulation...')
    pulse_pos = 0
    # Data blocks for imaging
    rbi_image = np.zeros(gx.shape, dtype=np.complex128)

    nz = np.zeros(cpi_len)
    ang_spread = np.linspace(-ang_dist_traveled_over_cpi / 2, ang_dist_traveled_over_cpi / 2,
                             cpi_len)
    tx_pattern = np.array([applyRadiationPatternCPU(0., a, 0., 0., 0., 0., rpref.az_half_bw, rpref.el_half_bw)
                           for a in ang_spread])
    H = convolution_matrix(tx_pattern * array_factor, gap_len, 'valid')

    # Truncated SVD for superresolution
    U, eig, Vt = np.linalg.svd(H, full_matrices=False)
    knee = 120
    eig[knee:] = 0
    H_hat = U.dot(np.diag(eig)).dot(Vt).T

    # Hinv = np.linalg.pinv(H)

    # H_w = np.fft.fft(tx_pattern * abs(array_factor), gap_len)
    abs_range_min = np.linalg.norm(grid_origin - rpref.rxpos(rpref.gpst), axis=1).min()
    det_pts = []
    ex_chirps = []

    if settings['simulation_params']['load_targets']:
        pass

    data_gen = genSimPulseData(rpref, rps, bg, u.mean(), settings['plp'], settings['upsample'],
                               settings['grid_width'], settings['grid_height'], settings['pts_per_m'],
                               gap_len, chirps, bpj_wavelength, data_t,
                               settings['antenna_params']['transmit_power'][0], settings['antenna_params']['gain'][0],
                               settings['rotate_grid'], settings['debug'], fft_len, settings['noise_level'],
                               settings['origin'])
    next(data_gen)  # Initialize the generator
    if settings['live_figures']:
        plt.ion()
    try:
        with contextlib.suppress(StopIteration):
            idx = 0
            while True:
                pdata = data_gen.send(chirps)
                ex_chirps.append(np.array(chirps[0].get()))
                ts = data_t[idx * gap_len + cpi_len // 2:idx * gap_len + gap_len - cpi_len // 2]
                # ts = data_t[idx * gap_len:idx * gap_len + gap_len]
                ts_hat = ts.min()
                compressed_data = np.zeros((gap_len, nsam * settings['upsample']), dtype=np.complex128)
                for ch_idx, rp in enumerate(rps):
                    try:
                        tmp_data = pdata[rp.rx_num] * mfilts[ch_idx]
                    except TypeError:
                        tmp_data = pdata[rp.rx_num] * mfilts[ch_idx].get()
                    tmp_exp = np.zeros((tmp_data.shape[0], up_fft_len), dtype=np.complex128)
                    tmp_exp[:, :fft_len // 2] = tmp_data[:, :fft_len // 2]
                    tmp_exp[:, -fft_len // 2:] = tmp_data[:, -fft_len // 2:]
                    compressed_data += np.fft.ifft(tmp_exp, axis=1)[:, :nsam * settings['upsample']] * avec[ch_idx]
                if not sim_settings['use_sdr_waveform'] and compressed_data.shape[0] == gap_len:
                    chirps, mfilts = reloadWaveforms(wave_mdl, active_clutter, nr, fft_len,
                                                     tsdata, rps, bandwidth_model,
                                                     wave_config['wave_exp_params']['dataset_params']['mu'],
                                                     wave_config['wave_exp_params']['dataset_params']['var'])

                panrx = rpref.pan(ts)
                elrx = rpref.tilt(ts)
                if compressed_data.shape[0] == gap_len:
                    gcv_k = np.zeros(gap_len)
                    sig_min = abs(compressed_data.T.dot(H_hat))
                    # sig_min = np.fft.ifft(np.fft.fft(compressed_data, axis=0) * H_w[:, None], axis=0).T
                    '''gcv_min = np.inf
                    for n in range(1, gap_len):
                        eig_k = eig + 0.0
                        eig_k[n:] = 0
                        eig_k[:n] = 1 / eig_k[:n]
                        Hinv = Vt.T.dot(np.diag(eig_k)).dot(U.T)
                        sig_k = compressed_data.T.dot(Hinv)
                        gcv_k[n] = np.linalg.norm(H.T.dot(sig_k.T) - compressed_data) / np.trace(np.eye(gap_len) - Hinv.dot(H))
                        if gcv_k[n] < gcv_min:
                            gcv_min = gcv_k[n]
                            sig_min = sig_k + 0.0'''
                    pt_dist = np.array([gx.flatten(), gy.flatten(), gz.flatten()]).T - rpref.rxpos(ts_hat)
                    pans = np.arctan2(pt_dist[:, 0], pt_dist[:, 1])
                    pt_ranges = np.linalg.norm(pt_dist, axis=1)

                    grads = np.sign(np.diff(panrx))
                    if len(np.unique(grads)) == 1:
                        valids = np.logical_and(pans < panrx.max(), pans > panrx.min())
                        panbins_pt = np.digitize(pans[valids], panrx)
                        rbins_pt = np.digitize(pt_ranges[valids], ranges)
                        rbi_image[valids.reshape(rbi_image.shape)] += sig_min[rbins_pt, panbins_pt]
                    if settings['live_figures']:
                        pt_proj = rpref.rxpos(ts_hat)[:, None] + azelToVec(panrx,
                                                                           elrx) * sim_settings['standoff_range']
                        plt.clf()
                        '''plt.subplot(2, 1, 1)
                        plt.plot(db(np.fft.fftshift(np.fft.ifft(chirps[0].get() * mfilts[0].get()))))
                        plt.subplot(2, 1, 2)
                        plt.imshow(db(sig_k))
                        plt.axis('tight')'''
                        plt.subplot(2, 1, 1)
                        plt.title(f'CPI {idx}')
                        plt.imshow(db(sig_min), extent=(panrx.min(), panrx.max(), ranges[0], ranges[-1]))
                        plt.axis('tight')
                        plt.subplot(2, 1, 2)
                        plt.imshow(db(bg.refgrid), extent=(gx.min(), gx.max(), gy.min(), gy.max()))
                        plt.scatter(pt_proj[0], pt_proj[1], s=40, c='red')
                        plt.draw()
                        plt.pause(.1)
                    if not sim_settings['use_sdr_waveform']:
                        active_clutter = np.fft.fft(np.fft.ifft(compressed_data, axis=1),
                                                    wave_config['settings']['fft_len'], axis=1)
                idx += 1
    except ValueError as ev:
        print(ev)

    """
    ----------------------------PLOTS-------------------------------
    """

    if settings['output_figures']:
        plt.figure(f'RBI image for {target_target}')
        plt.imshow(db(rbi_image), clim=[-30, 10])
        plt.axis('tight')
        plt.savefig(f'./data/fig_{target_target}.png', bbox_inches='tight')
    else:
        try:
            plt.figure('IMSHOW truth data')
            climage = db(refgrid)
            clim_image = climage[climage > -299]
            clims = (np.median(clim_image) - 1 * clim_image.std(),
                     max(np.median(clim_image) + 3 * clim_image.std(), np.max(clim_image) + 2))
            plt.imshow(np.rot90(db(refgrid)), origin='lower', clim=clims, cmap='gray')
            plt.axis('tight')
            plt.axis('off')
        except Exception as e:
            print(f'Error in generating background image: {e}')

        plt.figure('Waveform Info')
        plt.subplot(2, 1, 1)
        plt.title('Spectrum')
        for n in range(len(chirps)):
            plt.plot(db(chirps[n].get()))
        plt.subplot(2, 1, 2)
        plt.title('Time Series')
        for n in range(len(chirps)):
            plt.plot(np.fft.ifft(chirps[n].get()).real)

        plt.figure('Matched Filtering')
        plt.subplot(2, 1, 1)
        plt.plot(db(chirps[0].get()))
        plt.plot(db(mfilts[0].get()))
        plt.subplot(2, 1, 2)
        plt.plot(db(np.fft.ifft(chirps[0].get() * mfilts[0].get())))

        plt.figure('Chirp Changes')
        for n in ex_chirps:
            plt.plot(db(n))
        plt.show()

        rbi_image = np.array(rbi_image)

        plt.figure('Comparison')
        plt.subplot(2, 1, 1)
        plt.imshow(db(refgrid), extent=(gx.min(), gx.max(), gy.min(), gy.max()), clim=clims, cmap='gray')
        # plt.scatter(gx.flatten(), gy.flatten(), c=db(refgrid).flatten(), clim=clims)
        plt.scatter([plane_pos[0]], [plane_pos[1]], s=40)
        plt.subplot(2, 1, 2)
        plt.imshow(db(rbi_image), extent=(gx.min(), gx.max(), gy.min(), gy.max()), clim=[-30, 10], cmap='gray')
        plt.scatter([plane_pos[0]], [plane_pos[1]], s=40)

        '''plane_vec = plane_pos - rpref.pos(rpref.gpst).mean(axis=0)
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
        plt.axis('tight')'''

        pfig = px.imshow(db(refgrid), x=np.linspace(gx.min(), gx.max(), refgrid.shape[0]),
                         y=np.linspace(gy.min(), gy.max(), refgrid.shape[1]),
                         color_continuous_scale=px.colors.sequential.gray, zmin=130, zmax=180)
        pfig.show()

        plt.show()

        '''pfig = px.scatter_3d(x=gx.flatten(), y=gy.flatten(), z=gz.flatten())
        pfig.add_scatter3d(x=im_x.flatten(), y=im_y.flatten(), z=im_z.flatten())
        pfig.show()'''

        '''plt.figure('Target Angles')
        plt.plot(az_spread / DTR, db(np.sum(bg_image, axis=1)))
        plt.plot(az_spread / DTR, db(np.sum(rbi_image, axis=0)))'''
