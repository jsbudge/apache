import sys
import numpy as np
import torch
from numba import cuda
from simulib.backproject_functions import backprojectPulseSet, backprojectPulseStream
from simulib.mesh_objects import Mesh, Scene
from apache_helper import ApachePlatform
from config import get_config
from models import TargetEmbedding
from simulib.simulation_functions import llh2enu, db, azelToVec, genChirp, genTaylorWindow
from simulib.cuda_kernels import applyRadiationPatternCPU
from simulib.mesh_functions import readCombineMeshFile, getRangeProfileFromScene
from simulib.mimo_functions import genChirpAndMatchedFilters, genChannels
from simulib.grid_helper import MapEnvironment
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import yaml
from scipy.signal import sawtooth
from scipy.linalg import convolution_matrix
from waveform_model import GeneratorModel
import matplotlib as mplib
mplib.use('TkAgg')

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
    grid_origin = settings['origin']

    # This is all the constants in the radar equation for this simulation
    fc = settings['fc']
    radar_coeff = (
                c0 ** 2 / fc ** 2 * settings['antenna_params']['transmit_power'][0] * 10 ** ((settings['antenna_params']['gain'][0] + 2.15) / 10) * 10 ** ((settings['antenna_params']['gain'][0] + 2.15) / 10) *
                10 ** ((settings['antenna_params']['rec_gain'][0] + 2.15) / 10) / (4 * np.pi) ** 3)
    noise_power = 10**(sim_settings['noise_power_db'] / 10)

    # Calculate out mesh extent
    bg = MapEnvironment(grid_origin, (settings['grid_width'], settings['grid_height']), background=np.ones(nbpj_pts))

    plane_pos = llh2enu(40.138052, -111.660027, 1365, bg.ref)

    # Generate a platform
    print('Generating platform...', end='')
    # Run directly at the plane from the south
    grid_origin = llh2enu(*bg.origin, bg.ref)
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
    data_t = rpref.getValidPulseTimings(settings['simulation_params']['prf'], nr / TAC, cpi_len, as_blocks=True)
    gap_len = len(data_t[0])
    # data_t = np.concatenate(data_t)

    ang_dist_traveled_over_cpi = cpi_len / sim_settings['prf'] * sim_settings['scan_rate'] * DTR

    try:
        target_target = int(sys.argv[1])
    except IndexError:
        target_target = sim_settings['target_target']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    chirps = [np.fft.fft(genChirp(nr, fs, fc, settings['bandwidth']), fft_len)]
    taytays = [genTaylorWindow(fc % fs, settings['bandwidth'] / 2, fs, fft_len)]
    mfilts = [fft_chirp.conj() * taytay for fft_chirp, taytay in zip(chirps, taytays)]
    pdd = np.fft.fftshift(np.fft.fft(genChirp(nr, fs, fc, settings['bandwidth']), wave_fft_len))
    pulse_data = np.stack([pdd.real, pdd.imag])
    print('Setting up wavemodel...')
    model_config = get_config('wave_exp', './vae_config.yaml')
    wave_mdl = GeneratorModel.load_from_checkpoint(f'{model_config.weights_path}/{model_config.model_name}.ckpt',
                                                   config=model_config, strict=False)
    # wave_mdl.to(device)
    print('Wavemodel loaded.')
    patterns = torch.tensor(torch.load('/home/jeff/repo/apache/data/target_tensors/target_embedding_means.pt')[2],
                            dtype=torch.float32)

    # Calculate out points on the ground
    gx, gy, gz = bg.getGrid(settings['origin'], settings['grid_width'], settings['grid_height'], *nbpj_pts,
                            rpref.heading if settings['rotate_grid'] else 0)
    grid_points = np.array([gx.flatten(), gy.flatten(), gz.flatten()]).T
    flight_path = rpref.txpos(rpref.gpst)

    # GPU device calculations
    streams = [cuda.stream() for _ in rps]

    # Get pointing vector for MIMO consolidation
    ublock = np.array(
        [azelToVec(n, 0) for n in
         np.linspace(-ang_dist_traveled_over_cpi / 2, ang_dist_traveled_over_cpi / 2, cpi_len)]).T
    uhat = azelToVec(np.pi / 2, 0)
    avec = np.exp(-1j * 2 * np.pi * fc / c0 * vx_array.dot(uhat))
    fine_ucavec = np.exp(-1j * 2 * np.pi * fc / c0 * vx_array.dot(ublock))
    array_factor = fine_ucavec.conj().T.dot(np.eye(vx_array.shape[0])).dot(fine_ucavec)[:, 0] / vx_array.shape[0]

    print('Loading mesh and settings...')
    mesh = readCombineMeshFile('/home/jeff/Documents/roman_facade/scene.gltf', points=3000000)
    mesh = mesh.rotate(mesh.get_rotation_matrix_from_xyz(np.array([np.pi / 2, 0, 0])))
    mesh = mesh.translate(llh2enu(*grid_origin, grid_origin), relative=False)

    bgmesh = Mesh(mesh, num_box_levels=settings['nbox_levels'])

    # Add in the target mesh
    mesh = readCombineMeshFile('/home/jeff/Documents/target_meshes/cessna-172-obj.obj', points=3000000)
    mesh = mesh.rotate(mesh.get_rotation_matrix_from_xyz(np.array([np.pi / 2, 0, 0])))
    mesh = mesh.translate(llh2enu(*grid_origin, grid_origin), relative=False)

    planemesh = Mesh(mesh, num_box_levels=settings['nbox_levels'])
    scene = Scene([bgmesh, planemesh])

    test = None
    print('Running simulation...')
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

    sample_points = scene.sample(2**16, view_pos=rpref.txpos(rpref.gpst[np.linspace(0, len(rpref.gpst) - 1, 4).astype(int)]))
    # bpj_grid = np.zeros_like(gx).astype(np.complex128)

    if settings['simulation_params']['load_targets']:
        pass
    ts = data_t[10]
    txposes = [rp.txpos(ts).astype(np.float32) for rp in rps]
    rxposes = [rp.rxpos(ts).astype(np.float32) for rp in rps]
    pans = [rp.pan(ts).astype(np.float32) for rp in rps]
    tilts = [rp.tilt(ts).astype(np.float32) for rp in rps]
    init_rp = getRangeProfileFromScene(scene, sample_points, txposes, rxposes, pans, tilts,
                                         radar_coeff, rpref.az_half_bw, rpref.el_half_bw, nsam, fc, near_range_s,
                                         num_bounces=1, streams=streams)
    single_data = [np.ascontiguousarray(np.fft.fft(srp, fft_len, axis=1) * mfilt * pulse)
                   for srp, mfilt, pulse in zip(init_rp, mfilts, chirps)]
    pdd = np.stack(single_data).swapaxes(0, 1)
    pdd = np.fft.fftshift(np.sum(pdd * avec[None, :, None], axis=1), axes=1)
    pdd = pdd[0, ::2]
    pulse_data = np.stack([pdd.real, pdd.imag])

    wave_mdl.to(device)
    waves = wave_mdl.full_forward(pulse_data, patterns.squeeze(0).to(device), nr, settings['bandwidth'] / fs)
    wave_mdl.to('cpu')
    waves = np.fft.fft(np.fft.ifft(waves, axis=1)[:, :nr], fft_len, axis=1) * 1e6
    chirps = [waves for _ in rps]
    taytays = [genTaylorWindow(fc % fs, settings['bandwidth'] / 2, fs, fft_len)]
    mfilts = [fft_chirp.conj() * taytay for fft_chirp, taytay in zip(chirps, taytays)]

    for idx, ts in enumerate(data_t):
        # Modify the pulse
        ex_chirps.append(chirps)
        wave_mdl.to(device)
        waves = wave_mdl.full_forward(pulse_data, patterns.squeeze(0).to(device), nr, settings['bandwidth'] / fs)
        wave_mdl.to('cpu')
        waves = np.fft.fft(np.fft.ifft(waves, axis=1)[:, :nr], fft_len, axis=1) * 1e6
        chirps = [waves for _ in rps]
        taytays = [genTaylorWindow(fc % fs, settings['bandwidth'] / 2, fs, fft_len)]
        mfilts = [fft_chirp.conj() * taytay for fft_chirp, taytay in zip(chirps, taytays)]
        txposes = [rp.txpos(ts).astype(np.float32) for rp in rps]
        rxposes = [rp.rxpos(ts).astype(np.float32) for rp in rps]
        pans = [rp.pan(ts).astype(np.float32) for rp in rps]
        tilts = [rp.tilt(ts).astype(np.float32) for rp in rps]
        single_rp = getRangeProfileFromScene(scene, sample_points, txposes, rxposes, pans, tilts,
                                      radar_coeff, rpref.az_half_bw, rpref.el_half_bw, nsam, fc, near_range_s,
                                      num_bounces=1, streams=streams)
        if sum(np.sum(abs(s)) for s in single_rp) < 1e-10:
            continue
        single_data = [np.ascontiguousarray(np.fft.ifft(np.fft.fft(srp, fft_len, axis=1) * mfilt * pulse, axis=1)[:, :nsam])
                       for srp, mfilt, pulse in zip(single_rp, mfilts, chirps)]
        compressed_data = np.stack(single_data).swapaxes(0, 1)
        compressed_data = np.sum(compressed_data * avec[None, :, None], axis=1)

        # bpj_grid += backprojectPulseStream([compressed_data.T], pans, tilts, rxposes, txposes, gz.astype(np.float32), c0 / fc, near_range_s, fs * settings['upsample'],
        #                                    rpref.az_half_bw, rpref.el_half_bw, gx=gx.astype(np.float32), gy=gy.astype(np.float32), streams=streams)
        ts_hat = ts.min()
        panrx = rpref.pan(ts[:H_hat.shape[1]])
        elrx = rpref.tilt(ts[:H_hat.shape[1]])
        if compressed_data.shape[0] == gap_len:
            gcv_k = np.zeros(gap_len)
            sig_min = abs(compressed_data.T.dot(H_hat))
            pt_dist = grid_points - rpref.rxpos(ts_hat)
            pans = np.arctan2(pt_dist[:, 0], pt_dist[:, 1])
            pt_ranges = np.linalg.norm(pt_dist, axis=1)

            grads = np.sign(np.diff(panrx))
            if len(np.unique(grads)) == 1:
                valids = np.logical_and(np.logical_and(pans < panrx.max(), pans > panrx.min()),
                                        np.logical_and(pt_ranges < ranges.max(), pt_ranges > ranges.min()))
                panbins_pt = np.digitize(pans[valids], panrx)
                rbins_pt = np.digitize(pt_ranges[valids], ranges)
                rbi_image[valids.reshape(rbi_image.shape)] += sig_min[rbins_pt, panbins_pt]
            if settings['live_figures']:
                pt_proj = rpref.rxpos(ts_hat)[:, None] + azelToVec(panrx, elrx) * sim_settings['standoff_range']
                plt.clf()
                plt.subplot(2, 1, 1)
                plt.title(f'CPI {idx}')
                plt.imshow(db(sig_min), extent=(panrx.min(), panrx.max(), ranges[0], ranges[-1]))
                plt.axis('tight')
                plt.subplot(2, 1, 2)
                plt.imshow(db(rbi_image), extent=(gx.min(), gx.max(), gy.min(), gy.max()), origin='lower')
                plt.scatter(pt_proj[0], pt_proj[1], s=40, c='red')
                plt.draw()
                plt.pause(.1)
            pulse_data = np.fft.fftshift(np.fft.fft(np.fft.ifft(compressed_data, axis=1),
                                        wave_config['settings']['fft_len'], axis=1), axes=1)

    print('Simulation complete. Plotting stuff...')

    """
    ----------------------------PLOTS-------------------------------
    """

    if settings['output_figures']:
        plt.figure(f'RBI image for {target_target}')
        plt.imshow(db(rbi_image))
        plt.axis('tight')
        plt.savefig(f'./data/fig_{target_target}.png', bbox_inches='tight')
    else:

        plt.figure('Waveform Info')
        plt.subplot(2, 1, 1)
        plt.title('Spectrum')
        for ch in chirps:
            plt.plot(db(ch[0]))
        plt.subplot(2, 1, 2)
        plt.title('Time Series')
        for ch in chirps:
            plt.plot(np.fft.ifft(ch[0]).real)

        plt.figure('Matched Filtering')
        plt.subplot(2, 1, 1)
        plt.plot(db(chirps[0].ravel()))
        plt.plot(db(mfilts[0].ravel()))
        plt.subplot(2, 1, 2)
        plt.plot(db(np.fft.ifft(chirps[0].ravel() * mfilts[0].ravel())))

        plt.figure('Chirp Changes')
        for n in ex_chirps:
            plt.plot(db(n[0].flatten()))

        rbi_image = np.array(rbi_image)
        plt.figure(f'RBI image for {target_target}')
        plt.imshow(db(rbi_image))
        plt.axis('tight')


        def getMeshFig(title='Title Goes Here', zrange=100):
            fig = go.Figure(data=[
                go.Mesh3d(
                    x=bgmesh.vertices[:, 0],
                    y=bgmesh.vertices[:, 1],
                    z=bgmesh.vertices[:, 2],
                    # i, j and k give the vertices of triangles
                    i=bgmesh.tri_idx[:, 0],
                    j=bgmesh.tri_idx[:, 1],
                    k=bgmesh.tri_idx[:, 2],
                    # facecolor=triangle_colors,
                    showscale=True
                )
            ])
            fig.update_layout(
                title=title,
                scene=dict(zaxis=dict(range=[-30, zrange])),
            )
            return fig


        fig = getMeshFig('Full Mesh', flight_path[:, 2].mean() + 10)
        fig.add_trace(
            go.Scatter3d(x=flight_path[::100, 0], y=flight_path[::100, 1], z=flight_path[::100, 2], mode='lines'))
        mean_bore = rpref.boresight(rpref.gpst).mean(axis=0)
        fig.add_trace(
            go.Cone(x=[flight_path[:, 0].mean()], y=[flight_path[:, 1].mean()], z=[flight_path[:, 2].mean()], u=[mean_bore[0]], v=[mean_bore[1]], w=[mean_bore[2]], sizeref=40, anchor='tail')
        )
        fig.show()
        plt.show()
