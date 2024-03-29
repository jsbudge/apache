import numpy as np
from simulib.simulation_functions import getElevation, llh2enu, findPowerOf2, db, enu2llh, azelToVec
from data_converter.aps_io import loadCorrectionGPSData, loadGPSData, loadGimbalData
from simulib.cuda_kernels import getMaxThreads, backproject
from simulib.jax_kernels import range_profile_vectorized
import jax.numpy as jnp
import jax
from simulib.grid_helper import SDREnvironment
from simulib.platform_helper import SDRPlatform, APSDebugPlatform
import cupy as cupy
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from tqdm import tqdm
from data_converter.SDRParsing import load, findAllFilenames, findDebugFilenames
import yaml
from generate_trainingdata import getVAECov, formatTargetClutterData
from scipy.interpolate import RectBivariateSpline
from sklearn.preprocessing import QuantileTransformer
import imageio.v2 as imageio

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254

with open('./target_data_generator.yaml') as y:
    settings = yaml.safe_load(y.read())
with open('./vae_config.yaml', 'r') as file:
    try:
        wave_config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
with open('./target_location_list.yaml') as y:
    try:
        target_list = yaml.safe_load(y)
    except yaml.YAMLError as exc:
        print(exc)

for tfile, tloc, tgrid_sz, tname in (
        zip(target_list['target_file'], target_list['target_location'], target_list['target_grid_sz'],
            target_list['target_name'])):
    nbpj_pts = (int(tgrid_sz[0] * settings['pts_per_m']), int(tgrid_sz[1] * settings['pts_per_m']))

    print('Loading SDR file...')
    sdr = load(tfile, export_pickle=False, do_exact_matches=False, use_jump_correction=False)
    settings['origin'] = tloc

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

    # sdr.gimbal['systime'] += TAC * .01

    bg = SDREnvironment(sdr)
    ref_llh = bg.ref

    # Generate a platform
    print('Generating platform...', end='')
    rp = SDRPlatform(sdr, ref_llh, channel=settings['channel'])

    # rp.az_half_bw *= .5
    # rp.el_half_bw *= .5

    # Get reference data
    fs = sdr[settings['channel']].fs
    bwidth = sdr[settings['channel']].bw
    fc = sdr[settings['channel']].fc
    print('Done.')

    # Generate values needed for backprojection
    print('Calculating grid parameters...')
    # General calculations for slant ranges, etc.
    # plat_height = rp.pos(rp.gpst)[2, :].mean()
    nsam = rp.calcNumSamples(settings['fdelay'], settings['plp'])
    ranges = rp.calcRangeBins(settings['fdelay'], settings['upsample'], settings['plp'])
    ranges_sampled = rp.calcRangeBins(settings['fdelay'], 1, settings['plp'])
    near_range_s = ranges[0] / c0
    granges = ranges * np.cos(rp.dep_ang)
    fft_len = findPowerOf2(nsam + rp.calcPulseLength(settings['fdelay'], settings['plp'], use_tac=True))
    up_fft_len = fft_len * settings['upsample']

    # Chirp and matched filter calculations
    try:
        bpj_wavelength = c0 / (fc - bwidth / 2 - sdr[settings['channel']].xml['DC_Offset_MHz'] * 1e6) \
            if sdr[settings['channel']].xml['Offset_Video_Enabled'].lower() == 'true' else c0 / fc
    except KeyError as e:
        f'Could not find {e}'
        bpj_wavelength = c0 / (fc - bwidth / 2 - 5e6)

    mfilt = sdr.genMatchedFilter(settings['channel'], fft_len=fft_len)
    mfilt_gpu = cupy.array(np.tile(mfilt, (settings['cpi_len'], 1)).T, dtype=np.complex128)
    rbins_gpu = cupy.array(ranges, dtype=np.float64)

    noise_level = 0
    if settings['gen_data']:
        # Get all the JAX info ready for random point generation
        chirp = jnp.tile(jnp.fft.fft(sdr[settings['channel']].cal_chirp, fft_len), (settings['cpi_len'], 1)).T
        mfilt_jax = np.tile(mfilt, (settings['cpi_len'], 1)).T
        mapped_rpg = jax.vmap(range_profile_vectorized,
                              in_axes=[None, None, None, None, 0, 0, 0, 0, 0, 0,
                                       None, None, None, None, None, None, None, None])
        bg.resampleGrid(settings['origin'], settings['grid_width'], settings['grid_height'], *nbpj_pts,
                        bg.heading if settings['rotate_grid'] else 0)

        # This replaces the ASI background with a custom image
        '''bg_image = imageio.imread('/data6/Jeff_Backup/Pictures/josh.png').sum(axis=2)
        bg_image = RectBivariateSpline(np.arange(bg_image.shape[0]), np.arange(bg_image.shape[1]), bg_image)(
            np.linspace(0, bg_image.shape[0], nbpj_pts[0]), np.linspace(0, bg_image.shape[1], nbpj_pts[1])) / 750'''
        '''bg_image = np.zeros_like(bg.refgrid)
        bg_image[bg_image.shape[0] // 2, bg_image.shape[1] // 2] = 10'''
        # bg._refgrid = bg_image

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

    if settings['debug']:
        pts_debug = cupy.zeros((3, *gx.shape), dtype=np.float64)
        angs_debug = cupy.zeros((3, *gx.shape), dtype=np.float64)
    else:
        pts_debug = cupy.zeros((1, 1), dtype=np.float64)
        angs_debug = cupy.zeros((1, 1), dtype=np.float64)

    # GPU device calculations
    threads_per_block = getMaxThreads()
    bpg_bpj = (max(1, (nbpj_pts[0]) // threads_per_block[0] + 1), (nbpj_pts[1]) // threads_per_block[1] + 1)

    bin_bw = int(wave_config['settings']['bandwidth'] // (sdr[0].fs / fft_len))
    bin_bw += 1 if bin_bw % 2 != 0 else 0
    rollback = -int(np.round(sdr[0].baseband_fc / (sdr[0].fs / fft_len)))
    target_data_dec_factor = fft_len // wave_config['generate_data_settings']['fft_sz']

    # Run through loop to get data simulated
    data_t = sdr[settings['channel']].pulse_time
    idx_t = sdr[settings['channel']].frame_num
    data_check = None
    clutter_abs = list()
    inp_data = list()
    print('Backprojecting...')
    pulse_pos = 0
    # Data blocks for imaging
    bpj_truedata = np.zeros(nbpj_pts, dtype=np.complex128)
    for tidx, frames in tqdm(
            enumerate(idx_t[pos:pos + settings['cpi_len']] for pos in range(0, len(data_t), settings['cpi_len'])),
            total=len(data_t) // settings['cpi_len'] + 1):
        ts = data_t[tidx * settings['cpi_len'] + np.arange(len(frames))]
        tmp_len = len(ts)
        panrx = rp.pan(ts)
        elrx = rp.tilt(ts)
        posrx = rp.rxpos(ts)
        postx = rp.txpos(ts)

        panrx_gpu = cupy.array(panrx, dtype=np.float64)
        elrx_gpu = cupy.array(elrx, dtype=np.float64)
        posrx_gpu = cupy.array(posrx, dtype=np.float64)
        postx_gpu = cupy.array(postx, dtype=np.float64)
        bpj_grid = cupy.zeros(nbpj_pts, dtype=np.complex128)

        if settings['gen_data']:
            pdata = mapped_rpg(bg.transforms[0], bg.transforms[1], gz, bg.refgrid,
                               postx, posrx, panrx, elrx, panrx, elrx,
                               bpj_wavelength, near_range_s, rp.fs, rp.az_half_bw, rp.el_half_bw,
                               ranges_sampled, settings['pts_per_tri'], receive_power_scale)
            rtdata = cupy.array(np.array(jnp.fft.fft(pdata, fft_len, axis=1).T *
                                         chirp[:, :tmp_len] * mfilt_jax[:, :tmp_len]), dtype=np.complex128)
            if settings['save_as_target']:
                data_target = rtdata.copy()
            upsample_data = cupy.array(np.random.normal(0, noise_level, (up_fft_len, tmp_len)) +
                                       1j * np.random.normal(0, noise_level, (up_fft_len, tmp_len)),
                                       dtype=np.complex128)
        else:
            # Reset the grid for truth data
            rtdata = cupy.fft.fft(cupy.array(sdr.getPulses(frames, settings['channel'])[1],
                                             dtype=np.complex128), fft_len, axis=0)
            if settings['save_as_target']:
                data_target = cupy.fft.ifft(rtdata, axis=0)[:nsam, :].get()
            rtdata = rtdata * mfilt_gpu[:, :tmp_len]

            upsample_data = cupy.zeros((up_fft_len, tmp_len), dtype=np.complex128)
        upsample_data[:fft_len // 2, :] += rtdata[:fft_len // 2, :]
        upsample_data[-fft_len // 2:, :] += rtdata[-fft_len // 2:, :]
        rtdata = cupy.fft.ifft(upsample_data, axis=0)[:nsam * settings['upsample'], :]
        cupy.cuda.Device().synchronize()

        backproject[bpg_bpj, threads_per_block](postx_gpu, posrx_gpu, gx_gpu, gy_gpu, gz_gpu, rbins_gpu, panrx_gpu,
                                                elrx_gpu, panrx_gpu, elrx_gpu, rtdata, bpj_grid,
                                                bpj_wavelength, near_range_s, rp.fs * settings['upsample'], bwidth,
                                                rp.az_half_bw,
                                                rp.el_half_bw, settings['poly_num'], pts_debug, angs_debug,
                                                settings['debug'])
        cupy.cuda.Device().synchronize()

        # If we're halfway through the collect, grab debug data
        postoorig = llh2enu(*settings['origin'], bg.ref) - rp.pos(ts)
        angtoorig = np.arctan2(-postoorig[:, 1], postoorig[:, 0]) + np.pi / 2 - panrx
        if np.any(abs(angtoorig) < .1 * DTR):
            locp = rp.pos(ts[-1]).T
            data_check = rtdata.get()
            angd = angs_debug.get()
            locd = pts_debug.get()
        if settings['save_as_target'] and tmp_len >= 32:
            pmean, cov_dt = getVAECov(data_target[:, :32], rollback=rollback // target_data_dec_factor,
                                      mfilt=mfilt[::target_data_dec_factor],
                                      fft_len=wave_config['generate_data_settings']['fft_sz'],
                                      var=wave_config['dataset_params']['var'])
            clutter_abs.append(pmean)
            inp_data.append(cov_dt)

        bpj_truedata += bpj_grid.get()

    del panrx_gpu
    del postx_gpu
    del posrx_gpu
    del elrx_gpu
    del rtdata
    del upsample_data
    del bpj_grid
    # del shift

    del rbins_gpu
    del gx_gpu
    del gy_gpu
    del gz_gpu

    # Apply range roll-off compensation to final image
    mag_data = np.sqrt(abs(bpj_truedata))
    brightness_raw = np.median(np.sqrt(abs(bpj_truedata)), axis=1)
    brightness_curve = np.polyval(np.polyfit(np.arange(bpj_truedata.shape[0]), brightness_raw, 4),
                                  np.arange(bpj_truedata.shape[1]))
    brightness_curve /= brightness_curve.max()
    brightness_curve = 1. / brightness_curve
    mag_data *= np.outer(np.ones(mag_data.shape[0]), brightness_curve)

    if settings['save_as_target']:
        inp_data = np.array(inp_data, dtype=np.float32)
        clutter_abs = np.array(clutter_abs).astype(np.float32)
        with open(
                f'./data/targets_{tname}.cov', 'ab') as writer:
            inp_data.tofile(writer)
        with open(
                f'./data/targets_{tname}.spec', 'ab') as writer:
            clutter_abs.tofile(writer)

    """
    ----------------------------PLOTS-------------------------------
    """

    if data_check is not None:
        plt.figure(f'Doppler data_{tfile}')
        plt.imshow(np.fft.fftshift(db(np.fft.fft(data_check, axis=1)), axes=1),
                   extent=(-sdr[settings['channel']].prf / 2, sdr[settings['channel']].prf / 2, ranges[-1], ranges[0]))
        plt.axis('tight')

    plt.figure(f'IMSHOW backprojected data_{tfile}')
    plt.imshow(db(mag_data), origin='lower')
    plt.axis('tight')

    try:
        if (nbpj_pts[0] * nbpj_pts[1]) < 400 ** 2:
            cx, cy, cz = bg.getGrid(settings['origin'], width=settings['grid_width'], height=settings['grid_height'],
                                    nrows=nbpj_pts[0], ncols=nbpj_pts[1], az=0)

            fig = px.scatter_3d(x=gx.flatten(), y=gy.flatten(), z=gz.flatten())
            fig.add_scatter3d(x=cx.flatten(), y=cy.flatten(), z=cz.flatten(), mode='markers')
            fig.show()

            pvecs = azelToVec(angd[1, ...].flatten(), angd[0, ...].flatten()) * angd[2, ...].flatten()
            fig = px.scatter_3d(x=gx.flatten(), y=gy.flatten(), z=gz.flatten())
            fig.add_scatter3d(x=locd[0, ...].flatten() + locp[0], y=locd[1, ...].flatten() + locp[1],
                              z=locd[2, ...].flatten() + locp[2], mode='markers')
            fig.add_scatter3d(x=pvecs[0, ...] + locp[0], y=pvecs[1, ...].flatten() + locp[1],
                              z=pvecs[2, ...].flatten() + locp[2], mode='markers')
            fig.show()

        plt.figure(f'IMSHOW truth data: {tfile}')
        plt.imshow(db(bg.refgrid), origin='lower')
        plt.axis('tight')
    except Exception as e:
        print(f'Error in generating background image: {e}')

    # mag_data = np.sqrt(abs(sdr.loadASI(sdr.files['asi'][0])))
    nbits = 256
    plot_data_init = QuantileTransformer(output_distribution='normal').fit(
        mag_data[mag_data > 0].reshape(-1, 1)).transform(mag_data.reshape(-1, 1)).reshape(mag_data.shape)
    plot_data = plot_data_init
    max_bin = 3
    hist_counts, hist_bins = \
        np.histogram(plot_data, bins=np.linspace(-1, max_bin, nbits))
    while hist_counts[-1] == 0:
        max_bin -= .2
        hist_counts, hist_bins = \
            np.histogram(plot_data, bins=np.linspace(-1, max_bin, nbits))
    scaled_data = np.digitize(plot_data_init, hist_bins)

    # px.imshow(db(mag_data), color_continuous_scale=px.colors.sequential.gray).show()
    px.imshow(scaled_data, color_continuous_scale=px.colors.sequential.gray, zmin=0, zmax=nbits,
              origin='lower').show()
    plt.figure(f'Image Histogram_{tfile}')
    plt.plot(hist_bins[1:], hist_counts)

    plt.show()

    if settings['save_as_target']:
        plt.figure(f'Target vs. Clutter power: {tfile}')
        target = np.fft.fft(data_check[::settings['upsample'], 5], fft_len)
        target /= sum(abs(np.sqrt(target * target.conj())))
        clutter = np.fft.fft(sdr.getPulse(1000, 0)[1].flatten(), fft_len) * mfilt
        clutter /= sum(abs(np.sqrt(clutter * clutter.conj())))
        freqs = np.fft.fftshift(np.fft.fftfreq(fft_len, 1 / fs))
        plt.plot(freqs, np.fft.fftshift(db(np.fft.fft(data_check[::settings['upsample'], 5], fft_len))))
        plt.plot(freqs, np.fft.fftshift(db(np.fft.fft(sdr.getPulse(1000, 0)[1].flatten(), fft_len) * mfilt)))
        plt.ylabel('Power (dB)')
        plt.xlabel('Freq (GHz)')
        plt.legend(['Target', 'Clutter'])

        plt.figure(f'Target Covariance_{tfile}')
        plt.imshow(db(inp_data[0, :, :, 0] + 1j * inp_data[0, :, :, 1]))

        plt.figure(f'Target Spec_{tfile}')
        plt.plot(db(pmean[:, 0] + 1j * pmean[:, 1]))


