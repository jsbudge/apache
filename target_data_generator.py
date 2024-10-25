from glob import glob
import os
import numpy as np
from pathlib import Path
from simulib.simulation_functions import db, azelToVec, genChirp, getElevation
from simulib.cuda_mesh_kernels import readCombineMeshFile, getBoxesSamplesFromMesh, getRangeProfileFromMesh
from simulib.platform_helper import SDRPlatform
from simulib.cuda_kernels import cpudiff, getMaxThreads
from generate_trainingdata import formatTargetClutterData
from scipy.signal.windows import taylor
import matplotlib.pyplot as plt
import plotly.io as pio
from tqdm import tqdm
import yaml
from data_converter.SDRParsing import load
import torch
import cupy

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
TARGET_PROFILE_MIN_BEAMWIDTH = 0.19634954
fs = 2e9


def readVCS(filepath):
    # Load in the VCS file using the format reader
    with open(filepath, 'r') as f:
        # Total azimuths, elevations, scatterers
        header = [int(k) for k in f.readline().strip().split(' ')]
        scatterers = np.zeros((header[2], 5))
        nblock = 0
        scat_data = []
        angles = []
        while nblock < header[2]:
            subhead = [int(k) for k in f.readline().strip().split(' ')]
            angles.append(subhead[:2])
            for scat in range(subhead[2]):
                scatdata = np.array([float(k) for k in f.readline().strip().split(' ')])
                scatterers[nblock + scat, :] = scatdata[:5]
            blockdata = scatterers[nblock:nblock + scat, :]
            scat_data.append(blockdata[blockdata[:, 3] + blockdata[:, 4] > 1e-1])
            nblock += subhead[2]
    return scat_data, np.array(angles) * DTR


def genProfileFromMesh(obj_path, niters, mf_chirp, nboxes, points_to_sample, scaling, n_tris=20000):
    try:
        mesh = readCombineMeshFile(obj_path, n_tris, scale=1 / scaling)
    except IndexError:
        print(f'{tobj} not found.')

    box_tree, sample_points = getBoxesSamplesFromMesh(mesh, num_boxes=nboxes, sample_points=points_to_sample,
                                                      material_sigmas=[2.])

    # Apply random rotations and scalings for augmenting of training data
    for i in range(niters):
        chirp_idx = np.random.randint(0, len(mf_chirp))
        near_range_s, poses, boresights = calcPosBoresight(standoff + np.random.rand() * 100)

        single_rp = getRangeProfileFromMesh(*box_tree, sample_points, poses, boresights, radar_coeff, 30 * DTR,
                                            30 * DTR,
                                            nsam, fc, near_range_s,
                                            num_bounces=num_bounces,
                                            bounce_rays=nbounce_ray)
        yield np.fft.fft(single_rp, fft_len, axis=1) * mf_chirp[chirp_idx], i


def genProfileFromVCS(obj_path, niters, mf_chirp):
    # Load in the VCS file using the format reader
    scat_data, angles = readVCS(obj_path)

    # Generate target profile on CPU
    for i in range(niters):
        chirp_idx = np.random.randint(0, len(mf_chirp))
        near_range_s, poses, boresights = calcPosBoresight(standoff)
        bore_az = np.arctan2(boresights[:, 0], boresights[:, 1])
        bore_el = -np.arcsin(boresights[:, 2])
        bore_ang = np.array([bore_az, bore_el]).T
        pd = np.zeros((poses.shape[0], nsam), dtype=np.complex128)
        for n in range(poses.shape[0]):
            bl = scat_data[np.argmin(np.linalg.norm(angles - bore_ang[n], axis=1))]
            rvec = bl[:, :3] - poses[n]
            ranges = np.linalg.norm(rvec, axis=1)
            rng_bin = (ranges * 2 / c0 - 2 * near_range_s) * fs
            but = rng_bin.astype(int)
            pd[n, but] += np.exp(-1j * wavenumber * ranges * 2) * np.max(bl[:, 3:], axis=1)
        yield np.fft.fft(pd, fft_len, axis=1) * mf_chirp[chirp_idx], i


def getSpectrumFromTargetProfile(sdr, rp, ts, tcdata, fft_len):
    """
    Given a target profile matrix, generates pulses for a sample set of times and a trajectory.
    :param sdr: SDRParse object for trajectory.
    :param rp: RadarPlatform object to interpolate correct trajectory values.
    :param ts: Array of sample times (GPS times)
    :param tcdata: Target profile matrix.
    :param fft_len: Length of the FFT to return.
    :return: Match filtered, frequency domain data for the target profile.
    """
    pulse = np.fft.fft(sdr[0].cal_chirp, fft_len)
    mfilt = sdr.genMatchedFilter(0, fft_len=fft_len) / 1e4

    # Generate the pans and tilts that correspond to columns in the target profile matrix.
    pans, tilts = np.meshgrid(np.linspace(0, 2 * np.pi, 16, endpoint=False),
                              np.linspace(np.pi / 2 - .1, -np.pi / 2 + .1, 16))
    pan = pans.flatten()
    tilt = tilts.flatten()
    pan[pan == 0] = 1e-9
    pan[pan == 2 * np.pi] = 1e-9
    tilt[tilt == 0] = 1e-9
    tilt[tilt == 2 * np.pi] = 1e-9
    sdr_pan = rp.pan(ts)
    sdr_pan = sdr_pan + 2 * np.pi
    sdr_tilt = rp.tilt(ts)
    sdr_tilt = sdr_tilt + 2 * np.pi

    tpsd = np.array([tcdata[:, np.logical_and(abs(cpudiff(pan, sdr_pan[n])) < TARGET_PROFILE_MIN_BEAMWIDTH,
                                              abs(cpudiff(tilt, sdr_tilt[n])) < TARGET_PROFILE_MIN_BEAMWIDTH)].flatten()
                     for n in range(len(ts))])
    tpsd = np.fft.fft(tpsd, fft_len, axis=1) * 1e12 * pulse * mfilt
    return tpsd


def calcPosBoresight(standoff, pan=None, tilt=None):
    if pan is None:
        # Generate angles for block
        pans, tilts = np.meshgrid(np.linspace(0, 2 * np.pi, 16, endpoint=False),
                                  np.linspace(np.pi / 2 - .1, -np.pi / 2 + .1, 16))
        pan = pans.flatten()
        tilt = tilts.flatten()
    boresights = -azelToVec(pan, tilt).T
    poses = -boresights * standoff
    near_range_s = (standoff - 100.) / c0
    return near_range_s, poses, boresights


if __name__ == '__main__':

    ant_gain = 22  # dB
    ant_transmit_power = 100  # watts
    ant_eff_aperture = 10. * 10.  # m**2
    fc = 9.6e9
    nboxes = 25
    points_to_sample = 2000
    num_bounces = 0
    nbounce_ray = 5
    standoff = 500.
    nsam = 5678
    niters = 10

    with open('./vae_config.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            
    wavenumber = 2 * np.pi * fc / c0
    

    save_path = config['generate_data_settings']['local_path'] if (
        config)['generate_data_settings']['use_local_storage'] else config['dataset_params']['data_path']

    target_obj_files = {'air_balloon.obj': 30., 'black_bird_sr71_obj.obj': .25, 'cessna-172-obj.obj': 41.08,
                        'helic.obj': 1.8,
                        'Humvee.obj': 60, 'Intergalactic_Spaceships_Version_2.obj': 1., 'piper_pa18.obj': 1.,
                        'Porsche_911_GT2.obj': .8, 'Seahawk.obj': 12., 't-90a(Elements_of_war).obj': 1.,
                        'Tiger.obj': 156.25, 'G7_1200.obj': 1., 'x-wing.obj': 1., 'tacoma_VTC.dat': -1.,
                        'NissanSkylineGT-R(R32).obj': 1.25, 'ram1500trx2021.gltf': .0033, 'spider_tank.gltf': 1.,
                        'stug3aufs.gltf': 1., 'b2spirit.gltf': 1.}
    target_obj_files = list(target_obj_files.items())
    target_id_list = []

    clutter_files = glob(f'{save_path}/clutter_*.spec')

    # Standardize the FFT length for training purposes (this may cause data loss)
    fft_len = config['settings']['fft_len']

    # This is all the constants in the radar equation for this simulation
    radar_coeff = ant_transmit_power * 10 ** (ant_gain / 10) * ant_eff_aperture / (4 * np.pi) ** 2

    near_range_s, poses, boresights = calcPosBoresight(standoff)
    plt.ion()
    
    bws = 150e6 + np.random.rand(10) * 1e9
    plens = (1e-7 + np.random.rand(10) * (nsam / 2) / fs) * fs
    chirps = []
    mf_chirp = []
    for bw, plen in zip(bws, plens):
        twin = taylor(int(np.round(bw / fs * fft_len)))
        taytay = np.zeros(fft_len, dtype=np.complex128)
        winloc = int((fc % fs) * fft_len / fs) - len(twin) // 2
        chirps.append(genChirp(int(plen), fs, fc, bw))
        if winloc + len(twin) > fft_len:
            taytay[winloc:fft_len] += twin[:fft_len - winloc]
            taytay[:len(twin) - (fft_len - winloc)] += twin[fft_len - winloc:]
        else:
            taytay[winloc:winloc + len(twin)] += twin
        mf = np.fft.fft(chirps[-1], fft_len) * np.fft.fft(chirps[-1], fft_len).conj() * taytay
        mf_chirp.append(np.roll(mf, fft_len - (winloc + len(twin) // 2)))

    pt_sample = []
    abs_idx = 0

    for tidx, (tobj, scaling) in tqdm(enumerate(target_obj_files)):
        tensor_path = f"{config['target_exp_params']['dataset_params']['data_path']}/target_{tidx}"
        obj_path = f'{config["generate_data_settings"]["obj_path"]}/{tobj}'
        if not Path(tensor_path).exists():
            os.mkdir(tensor_path)
        if scaling > 0:
            gen_iter = iter(genProfileFromMesh(obj_path, niters, mf_chirp, nboxes, points_to_sample, scaling, n_tris=10000))
        else:
            gen_iter = iter(genProfileFromVCS(obj_path, niters, mf_chirp))
            
            
        pt_sample = []
        for pd, i in gen_iter:
            if np.all(pd == 0):
                print(f'Skipping on target {tidx}, pd {i}')
                continue
            pd = pd[np.any(pd, axis=1)]
            if i == 0:
                pt_sample = np.concatenate((pt_sample, pd[pd != 0]))
            pd = pd / np.sqrt(np.sum(pd * pd.conj(), axis=1))[:, None]
            # pd[pd != 0] = ((pd[pd != 0] - config['target_exp_params']['dataset_params']['mu']) /
            #                config['target_exp_params']['dataset_params']['var'])
            pd[pd != 0] *= 1 / pd[pd != 0].std()
            '''plt.gca().cla()
            plt.title(f'Iteration {i}')
            plt.imshow(db(pd))
            plt.draw()
            plt.pause(.1)'''
            pd_cat = np.stack([pd.real, pd.imag]).astype(np.float32).swapaxes(0, 1)

            if config['generate_data_settings']['save_as_target']:
                if i == 0:
                    target_id_list.append(tobj)
                for p in pd_cat:
                    torch.save([torch.tensor(p, dtype=torch.float32), tidx],
                               f"{tensor_path}/target_{tidx}_{abs_idx}.pt")
                    abs_idx += 1
        print(f'Target {tidx} mean of {pt_sample.mean()} and std of {pt_sample.std()}')

        if config['generate_data_settings']['save_as_clutter']:
            print(f'Saving clutter spec for target {tobj}')
            for clut in clutter_files:
                # Load the sar file that the clutter came from
                cfnme_parts = clut.split('_')[1:4]
                clut_name = clut.split('/')[-1][8:-5]
                sdr_ch = load(
                    f'/data6/SAR_DATA/{cfnme_parts[1][4:]}/{cfnme_parts[1]}/SAR_{cfnme_parts[1]}_{cfnme_parts[2][:-5]}.sar',
                    use_jump_correction=False)
                rp = SDRPlatform(sdr_ch)
                nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = (
                    rp.getRadarParams(0., .5, 1))

                bore = rp.boresight(rp.gpst).mean(axis=0)
                heading = np.arctan2(bore[0], bore[1])
                mrange = ranges.mean() * np.cos(sdr_ch.ant[0].dep_ang)

                mpos = np.array([mrange * np.cos(heading), mrange * np.sin(heading), 0.])

                # Locate the extrema to speed up the optimization
                flight_path = rp.txpos(sdr_ch[0].pulse_time)
                vecs = np.array([mpos[0] - flight_path[:, 0], mpos[1] - flight_path[:, 1],
                                 mpos[2] - flight_path[:, 2]]).T
                pt_az = np.arctan2(vecs[:, 0], vecs[:, 1])
                max_pts = sdr_ch[0].frame_num[abs(pt_az - heading) < rp.az_half_bw * 2]
                pulse_lims = [min(max_pts), max(max_pts)]

                # Get pulse data and modify accordingly
                pulse = np.fft.fft(sdr_ch[0].cal_chirp, fft_len)
                mfilt = sdr_ch.genMatchedFilter(0, fft_len=fft_len) / 1e4
                valids = mfilt != 0

                # CHeck to see we're writing to a fresh file
                if Path(f'{save_path}/target_{tidx}_{clut_name}.spec').exists():
                    os.remove(f'{save_path}/target_{tidx}_{clut_name}.spec')
                for n in range(pulse_lims[0], pulse_lims[1], config['settings']['cpi_len']):
                    if n + config['settings']['cpi_len'] > sdr_ch[0].nframes:
                        break
                    ts = sdr_ch[0].pulse_time[n:n + config['settings']['cpi_len']]
                    if scaling > 0:
                        single_rp = getRangeProfileFromMesh(*box_tree, sample_points,
                                                            rp.txpos(ts),
                                                            rp.boresight(ts),
                                                            radar_coeff,
                                                            rp.az_half_bw * 2,
                                                            rp.el_half_bw * 2,
                                                            nsam, fc, near_range_s,
                                                            num_bounces=num_bounces,
                                                            bounce_rays=nbounce_ray)
                        tpsd = np.fft.fft(single_rp, fft_len, axis=1) * mfilt
                    else:
                        # Load in the VCS file using the format reader
                        scat_data, angles = readVCS(obj_path)

                        # Generate target profile on CPU
                        poses = rp.txpos(ts)
                        boresights = rp.boresight(ts)
                        bore_az = np.arctan2(boresights[:, 0], boresights[:, 1])
                        bore_el = -np.arcsin(boresights[:, 2])
                        bore_ang = np.array([bore_az, bore_el]).T
                        pd = np.zeros((poses.shape[0], nsam), dtype=np.complex128)
                        for n in range(poses.shape[0]):
                            bl = scat_data[np.argmin(np.linalg.norm(angles - bore_ang[n], axis=1))]
                            rvec = bl[:, :3] - poses[n]
                            ranges = np.linalg.norm(rvec, axis=1)
                            rng_bin = (ranges * 2 / c0 - 2 * near_range_s) * fs
                            but = rng_bin.astype(int)
                            pd[n, but] += np.exp(-1j * wavenumber * ranges * 2) * np.max(bl[:, 3:], axis=1)
                        tpsd = np.fft.fft(pd, fft_len, axis=1) * mfilt * pulse
                    p_muscale = np.repeat(
                        config['exp_params']['dataset_params']['mu'] / abs(tpsd[:, valids].mean(axis=1)),
                        2).astype(
                        np.float32)
                    p_stdscale = np.repeat(
                        config['exp_params']['dataset_params']['var'] / abs(tpsd[:, valids].std(axis=1)),
                        2).astype(
                        np.float32)
                    tpsd[:, valids] = (tpsd[:, valids] - config['exp_params']['dataset_params']['mu']) / \
                                      config['exp_params']['dataset_params']['var']
                    # Shift the data so it's centered around zero (for the autoencoder)
                    if sdr_ch[0].baseband_fc != 0.:
                        shift_bin = int(sdr_ch[0].baseband_fc / sdr_ch[0].fs * fft_len)
                        pulse_data = np.roll(tpsd, -shift_bin, 1)
                    inp_data = formatTargetClutterData(tpsd, fft_len).astype(np.float32)
                    inp_data = np.concatenate((inp_data, p_muscale.reshape(-1, 2, 1), p_stdscale.reshape(-1, 2, 1)),
                                              axis=2)
                    with open(
                            f'{save_path}/target_{tidx}_{clut_name}.spec', 'ab') as writer:
                        inp_data.tofile(writer)

    with open(f'{save_path}/target_ids.txt', 'w') as f:
        for idx, tid in enumerate(target_id_list):
            f.write(f'{idx}: {tid}\n')