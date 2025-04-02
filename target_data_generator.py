from glob import glob
import os
import numpy as np
from pathlib import Path
from numba import cuda
from simulib.mesh_objects import Scene, Mesh
from simulib.simulation_functions import azelToVec, genChirp
from simulib.mesh_functions import readCombineMeshFile, getRangeProfileFromScene, _float
from simulib.platform_helper import SDRPlatform
from scipy.signal.windows import taylor
import plotly.io as pio
from tqdm import tqdm
import yaml
from sdrparse.SDRParsing import load, loadXMLFile
import torch

from utils import scale_normalize

pio.renderers.default = 'browser'

c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
fs = 2e9


def formatTargetClutterData(data: np.ndarray, bin_bandwidth: int):
    split = np.zeros((data.shape[0], 2, bin_bandwidth), dtype=np.float32)
    split[:, 0, :bin_bandwidth // 2] = data[:, -bin_bandwidth // 2:].real
    split[:, 0, -bin_bandwidth // 2:] = data[:, :bin_bandwidth // 2].real
    split[:, 1, :bin_bandwidth // 2] = data[:, -bin_bandwidth // 2:].imag
    split[:, 1, -bin_bandwidth // 2:] = data[:, :bin_bandwidth // 2].imag
    return split


def readVCS(filepath):
    # Load in the VCS file using the format reader
    with open(filepath, 'r') as f:
        # Total azimuths, elevations, scatterers
        header = [int(k) for k in f.readline().strip().split(' ')]
        scatterers = np.zeros((header[2], 5))
        nblock = 0
        _scat_data = []
        _angles = []
        while nblock < header[2]:
            subhead = [int(k) for k in f.readline().strip().split(' ')]
            _angles.append(subhead[:2])
            for scat in range(subhead[2]):
                scatdata = np.array([float(k) for k in f.readline().strip().split(' ')])
                scatterers[nblock + scat, :] = scatdata[:5]
            blockdata = scatterers[nblock:nblock + scat, :]
            _scat_data.append(blockdata[blockdata[:, 3] + blockdata[:, 4] > 1e-1])
            nblock += subhead[2]
    return _scat_data, np.array(_angles) * DTR


def genProfileFromMesh(a_obj_path, niters, a_mf_chirp, nbox_levels, a_points_to_sample, a_scaling, a_streams,
                       num_bounces=1, n_tris=2000000):
    try:
        m_mesh = readCombineMeshFile(a_obj_path, n_tris, scale=1 / a_scaling)
    except IndexError:
        print(f'{tobj} not found.')

    if np.linalg.norm(m_mesh.get_center()) > 2.:
        m_mesh = m_mesh.translate(np.array([0, 0, 0.]), relative=False)

    _, poses, _, _ = calcPosBoresight(standoff[0])
    view_pos = poses[np.linspace(0, poses.shape[0] - 1, 32).astype(int)]
    m_scene = Scene()
    m_scene.add(Mesh(m_mesh, num_box_levels=nbox_levels))
    m_sample_points = m_scene.sample(a_points_to_sample, view_pos=view_pos[::4])

    # Apply random rotations and scalings for augmenting of training data
    for m_i in range(niters):
        chirp_idx = np.random.randint(0, len(a_mf_chirp))
        block = np.zeros((3, poses.shape[0], fft_len), dtype=np.complex64)
        for k, s in enumerate(standoff):
            m_near_range_s, poses, m_pan, m_tilt = calcPosBoresight(s)
            m_rxposes = poses + 0.0
            m_single_rp = getRangeProfileFromScene(m_scene, m_sample_points, [poses.astype(_float)], [m_rxposes.astype(_float)],
                                                 [m_pan.astype(_float)],
                                                 [m_tilt.astype(_float)],
                                                 radar_coeff,
                                                 12 * DTR, 12 * DTR,
                                                 nsam, cfig_generate['fc'], m_near_range_s,
                                                 num_bounces=num_bounces, streams=a_streams)
            block[k] = np.fft.fft(m_single_rp[0], fft_len, axis=1) * a_mf_chirp[chirp_idx]
        yield block, m_i


def genProfileFromVCS(a_obj_path, a_niters, a_mf_chirp):
    # Load in the VCS file using the format reader
    m_scat_data, m_angles = readVCS(a_obj_path)

    # Generate target profile on CPU
    for m_i in range(a_niters):
        chirp_idx = np.random.randint(0, len(a_mf_chirp))
        block = np.zeros((3, n_az_samples * n_el_samples, fft_len), dtype=np.complex64)
        for k, s in enumerate(standoff):
            m_near_range_s, poses, bore_az, bore_el = calcPosBoresight(s)
            m_bore_ang = np.array([bore_az, bore_el]).T
            m_pd = np.zeros((poses.shape[0], nsam), dtype=np.complex128)
            for n in range(poses.shape[0]):
                m_bl = m_scat_data[np.argmin(np.linalg.norm(m_angles - m_bore_ang[n], axis=1))]
                m_rvec = m_bl[:, :3] - poses[n]
                m_ranges = np.linalg.norm(m_rvec, axis=1)
                m_rng_bin = (m_ranges * 2 / c0 - 2 * m_near_range_s) * fs
                m_but = m_rng_bin.astype(int)
                m_pd[n, m_but] += np.exp(-1j * wavenumber * m_ranges * 2) * np.max(m_bl[:, 3:], axis=1)
            block[k] = np.fft.fft(m_pd, fft_len, axis=1) * a_mf_chirp[chirp_idx]
        yield block, m_i


def calcPosBoresight(a_standoff, a_pan=None, a_tilt=None):
    if a_pan is None:
        # Generate angles for block
        m_pans, m_tilts = np.meshgrid(np.linspace(0, 2 * np.pi, n_az_samples, endpoint=False),
                                  np.linspace(np.pi / 2 - .1, -np.pi / 2 + .1, n_el_samples))
        a_pan = m_pans.flatten()
        a_tilt = m_tilts.flatten()
    boresights = -azelToVec(a_pan, a_tilt).T
    poses = -boresights * a_standoff
    m_near_range_s = (a_standoff - 100.) / c0
    return m_near_range_s, poses, a_pan, a_tilt


if __name__ == '__main__':
    cuda.select_device(1)

    rx_gain = 22  # dB
    tx_gain = 22  # dB
    rec_gain = 100  # dB
    ant_transmit_power = 100  # watts
    points_to_sample = 2**15
    nsam = 5678
    nstreams = 1

    with open('./vae_config.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    cfig_settings = config['settings']
    cfig_generate = config['generate_data_settings']

    standoff = [config['apache_params']['vehicle_slant_range_min'], config['apache_params']['vehicle_slant_range_max']]
    n_az_samples = cfig_generate['n_az_samples']
    n_el_samples = cfig_generate['n_el_samples']
    wavenumber = 2 * np.pi * cfig_generate['fc'] / c0

    save_path = cfig_generate['local_path'] if (
        config)['generate_data_settings']['use_local_storage'] else config['dataset_params']['data_path']

    with open('clutter_files.txt', 'r') as f:
        clt = [c.strip() for c in f.readlines()]

    clutter_files = []
    for clut in clt:
        if int(clut.split('/')[4][:2]) >= 6 and np.any([Path(clut).parts[-1][:-4] in g and
                                                        ('png' in g or 'jpeg' in g)
                                                        for g in glob(f'{str(Path(clut).parent)}/*')]):
            try:
                xml_data = loadXMLFile(f'{clut[:-4]}.xml', True)['SlimSDR_Configuration']
                if (xml_data['SlimSDR_Info']['System_Mode'] == 'SAR'
                        and 8e9 < xml_data['SlimSDR_Info']['Channel_0']['Center_Frequency_Hz'] < 32e9
                        and xml_data['SlimSDR_Info']['Gimbal_Settings']['Gimbal_Depression_Angle_D'] > 20.0):
                    clutter_files.append(clut)
            except FileNotFoundError:
                print(f'{clut} not found.')
                continue
            except Exception:
                print(f'{clut} has broken XML.')
                continue
        else:
            print(f'{clut} does not meet the requirements of the clutter parser.')
            continue

    with open('./target_files.yaml', 'r') as file:
        try:
            target_obj_files = list(yaml.safe_load(file).items())
        except yaml.YAMLError as exc:
            print(exc)
            exit()
    target_id_list = []

    streams = [cuda.stream() for _ in range(nstreams)]

    # Standardize the FFT length for training purposes (this may cause data loss)
    fft_len = cfig_settings['fft_len']

    # This is all the constants in the radar equation for this simulation
    radar_coeff = (
                c0 ** 2 / cfig_generate['fc'] ** 2 * ant_transmit_power * 10 ** ((rx_gain + 2.15) / 10) * 10 ** ((tx_gain + 2.15) / 10) *
                10 ** ((rec_gain + 2.15) / 10) / (4 * np.pi) ** 3)

    # Generate chirps with random bandwidths, pulse lengths
    bws = 150e6 + np.random.rand(10) * 1e9
    plens = (1e-7 + np.random.rand(10) * (nsam / 2) / fs) * fs
    chirps = []

    # Get matched filters for the chirps
    mf_chirp = []
    for bw, plen in zip(bws, plens):
        twin = taylor(int(np.round(bw / fs * fft_len)))
        taytay = np.zeros(fft_len, dtype=np.complex128)
        winloc = int((cfig_generate['fc'] % fs) * fft_len / fs) - len(twin) // 2
        chirps.append(genChirp(int(plen), fs, cfig_generate['fc'], bw))
        if winloc + len(twin) > fft_len:
            taytay[winloc:fft_len] += twin[:fft_len - winloc]
            taytay[:len(twin) - (fft_len - winloc)] += twin[fft_len - winloc:]
        else:
            taytay[winloc:winloc + len(twin)] += twin
        mf_chirp.append(np.fft.fft(chirps[-1], fft_len) * np.fft.fft(chirps[-1], fft_len).conj() * taytay)

    abs_idx = 0
    abs_clutter_idx = 0

    for tidx, (tobj, scaling) in tqdm(enumerate(target_obj_files)):
        obj_path = f'{config["generate_data_settings"]["obj_path"]}/{tobj}'

        # FIRST CODE BLOCK: Generate target representative blocks for an autoencoder to compress
        # Saves them out to a file
        if cfig_generate['save_as_target']:
            tensor_target_path = f"{config['target_exp_params']['dataset_params']['data_path']}/target_{tidx}"
            if not Path(tensor_target_path).exists():
                os.mkdir(tensor_target_path)
            # Generate a Nrange_samples x Nangles x fft_len block for the autoencoder, iterated niters times to get different
            # bandwidths and pulse lengths.
            if scaling > 0:
                gen_iter = iter(genProfileFromMesh(obj_path, cfig_generate['iterations'], mf_chirp, cfig_generate['nbox_levels'], points_to_sample, scaling, streams, cfig_generate['num_bounces']))
            else:
                gen_iter = iter(genProfileFromVCS(obj_path, cfig_generate['iterations'], mf_chirp))


            for pd, i in gen_iter:
                if np.all(pd == 0):
                    print(f'Skipping on target {tidx}, pd {i}')
                    continue
                # We want to normalize and scale this block, then format it to work with the autoencoder
                # Don't scale pulses without anything there, this is a unit energy calculation
                pd_mask = np.sqrt(np.sum(pd * pd.conj(), axis=-1).real) > 0.
                pd[pd_mask] = scale_normalize(pd[pd_mask])
                pd_cat = np.concatenate([formatTargetClutterData(p, fft_len) for p in pd], axis=1)

                if np.all(pd_cat == 0.):
                    continue

                # Append to master target list
                if i == 0:
                    target_id_list.append(tobj)
                # Save the block out to a torch file for the dataloader later
                torch.save([torch.tensor(pd_cat.swapaxes(0, 1), dtype=torch.float32), tidx],
                           f"{tensor_target_path}/target_{tidx}_{abs_idx}.pt")
                abs_idx += 1

        if cfig_generate['save_as_clutter']:
            tensor_clutter_path = f"{config['target_exp_params']['dataset_params']['data_path']}/clutter_tensors"
            if not Path(tensor_clutter_path).exists():
                os.mkdir(tensor_clutter_path)
            print(f'Saving clutter spec for target {tobj}')
            if scaling > 0:
                mesh = readCombineMeshFile(obj_path, points_to_sample, scale=1 / scaling)
            for clut in clutter_files:
                # Load the sar file that the clutter came from
                sdr_ch = load(clut, use_jump_correction=False)
                rp = SDRPlatform(sdr_ch)
                nsam, nr, ranges, ranges_sampled, near_range_s, granges, _, up_fft_len = (
                    rp.getRadarParams(0., .5, 1))

                flight_path = rp.rxpos(sdr_ch[0].pulse_time)
                bore = rp.boresight(rp.gpst).mean(axis=0)
                heading = np.arctan2(bore[0], bore[1])
                # The XML is not guaranteed to have a flight line, so we check for that
                try:
                    flight_alt = sdr_ch.xml['Flight_Line']['Flight_Line_Altitude_M']
                    gnd_exp_alt = flight_path.mean(axis=0)[2] - flight_alt
                    mpos = flight_path.mean(axis=0) + bore * ranges.min()
                    mpos[2] = gnd_exp_alt
                    mrng = np.linalg.norm(mpos)
                    mang = -np.arcsin(mpos[2] / mrng)
                    m_it = 1
                    while not ((ranges.max() > mrng > ranges.min()) and (rp.dep_ang + rp.el_half_bw > mang > rp.dep_ang - rp.el_half_bw)):
                        mpos = flight_path.mean(axis=0) + bore * (ranges.min() + m_it)
                        mpos[2] = gnd_exp_alt
                        mrng = np.linalg.norm(mpos)
                        mang = -np.arcsin(mpos[2] / mrng)
                        m_it += 1
                except KeyError:
                    # We don't know where the ground is, let's just project it into the abyss
                    mpos = flight_path.mean(axis=0) + bore * ranges.mean()

                # Locate the extrema to speed up the optimization
                scene = Scene()
                mesh = mesh.translate(mpos, relative=False)
                scene.add(Mesh(mesh))

                vecs = np.array([mpos[0] - flight_path[:, 0], mpos[1] - flight_path[:, 1],
                                 mpos[2] - flight_path[:, 2]]).T
                pt_az = np.arctan2(vecs[:, 0], vecs[:, 1])
                max_pts = sdr_ch[0].frame_num[abs(pt_az - heading) < rp.az_half_bw]
                pulse_lims = [min(max_pts), max(max_pts)]
                pulse_lims[1] = min(pulse_lims[1], pulse_lims[0] + cfig_settings['cpi_len'] * cfig_generate['iterations'])

                # Get pulse data and modify accordingly
                pulse = np.fft.fft(sdr_ch[0].cal_chirp, fft_len)
                mfilt = sdr_ch.genMatchedFilter(0, fft_len=fft_len)
                valids = mfilt != 0
                sample_points = scene.sample(points_to_sample, vecs[::10])

                for frame in list(zip(*(iter(range(pulse_lims[0], pulse_lims[1] - cfig_settings['cpi_len'], cfig_settings['cpi_len'])),) * (nstreams + 1))):
                    txposes = [rp.txpos(sdr_ch[0].pulse_time[frame[n]:frame[n] + cfig_settings['cpi_len']]).astype(np.float64) for n in
                               range(nstreams)]
                    rxposes = [rp.rxpos(sdr_ch[0].pulse_time[frame[n]:frame[n] + cfig_settings['cpi_len']]).astype(np.float64) for n in
                               range(nstreams)]
                    pans = [rp.pan(sdr_ch[0].pulse_time[frame[n]:frame[n] + cfig_settings['cpi_len']]).astype(np.float64) for n in
                            range(nstreams)]
                    tilts = [rp.tilt(sdr_ch[0].pulse_time[frame[n]:frame[n] + cfig_settings['cpi_len']]).astype(np.float64) for n in
                             range(nstreams)]
                    sdr_data = [sdr_ch.getPulses(sdr_ch[0].frame_num[frame[n]:frame[n] + cfig_settings['cpi_len']], 0)[1] for n in range(nstreams)]
                    if scaling > 0:
                        single_rp = getRangeProfileFromScene(scene, sample_points, txposes, rxposes,
                                                                                      pans,
                                                                                      tilts,
                                                                                      radar_coeff,
                                                                                      rp.az_half_bw, rp.el_half_bw,
                                                                                      nsam, cfig_generate['fc'], near_range_s,
                                                                                      num_bounces=cfig_generate['num_bounces'], streams=streams)
                        tpsds = [np.fft.fft(srp, fft_len, axis=1) * mfilt * pulse for srp in single_rp]
                    else:
                        # Load in the VCS file using the format reader
                        scat_data, angles = readVCS(obj_path)

                        # Generate target profile on CPU
                        tpsds = []
                        for txpos, rxpos, pan, tilt in zip(txposes, rxposes, pans, tilts):
                            bore_ang = np.array([pan, tilt]).T
                            pd = np.zeros((txpos.shape[0], nsam), dtype=np.complex128)
                            for n in range(txpos.shape[0]):
                                bl = scat_data[np.argmin(np.linalg.norm(angles - bore_ang[n], axis=1))]
                                rvec = bl[:, :3] - txpos[n] + mpos
                                ranges = np.linalg.norm(rvec, axis=1)
                                rng_bin = (ranges * 2 / c0 - 2 * near_range_s) * fs
                                but = rng_bin.astype(int)
                                pd[n, but] += np.exp(-1j * wavenumber * ranges * 2) * np.max(bl[:, 3:], axis=1)
                            tpsds.append(np.fft.fft(pd, fft_len, axis=1) * mfilt * pulse)
                    for sdrd, tpsd in zip(sdr_data, tpsds):
                        if len(tpsd[tpsd != 0]) == 0:
                            continue
                        sdata = np.fft.fft(sdrd, fft_len, axis=0).T * mfilt
                        if tpsd[tpsd != 0].std() == 0 or sdata[sdata != 0].std() == 0:
                            continue
                        # Unit energy and scale to have std of one
                        ntpsd = scale_normalize(tpsd)
                        sdata = scale_normalize(sdata)
                        # Shift the data so it's centered around zero (for the autoencoder)
                        if sdr_ch[0].baseband_fc != 0.:
                            shift_bin = int(sdr_ch[0].baseband_fc / sdr_ch[0].fs * fft_len)
                            ntpsd = np.roll(ntpsd, -shift_bin, 1)
                            sdata = np.roll(sdata, -shift_bin, 1)
                        ntpsd = formatTargetClutterData(ntpsd, fft_len)
                        sdata = formatTargetClutterData(sdata, fft_len)
                        for nt, sd in zip(ntpsd, sdata):
                            if not np.any(np.isnan(nt)) and not np.any(np.isnan(sd)):
                                torch.save([torch.tensor(sd, dtype=torch.float32),
                                            torch.tensor(nt, dtype=torch.float32), tidx],
                                           f"{tensor_clutter_path}/tc_{abs_clutter_idx}.pt")
                                abs_clutter_idx += 1

    with open(f'{save_path}/target_ids.txt', 'w') as f:
        for idx, tid in enumerate(target_id_list):
            f.write(f'{idx}: {tid}\n')