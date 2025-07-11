from glob import glob
import os
import numpy as np
from pathlib import Path
from numba import cuda
from simulib.mesh_objects import Scene, Mesh
from simulib.simulation_functions import azelToVec, genChirp
from simulib.mesh_functions import readCombineMeshFile, getRangeProfileFromScene, _float
from simulib.platform_helper import SDRPlatform
from scipy.signal.windows import taylor, tukey
import plotly.io as pio
from tqdm import tqdm
import yaml
from sdrparse.SDRParsing import load, loadXMLFile
import torch
import matplotlib as mplib

from config import load_yaml_config

mplib.use('TkAgg')
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from simulib.simulation_functions import db
import pandas as pd

from utils import scale_normalize, get_radar_coeff, normalize

pio.renderers.default = 'browser'

c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
BASE_BANDWIDTH = 150e6
BANDWIDTH_RANGE = 1e9
BASE_PULSE_LENGTH = 1e-7
MAX_AZ_BEAMWIDTH = 25 * DTR
MAX_EL_BEAMWIDTH = 25 * DTR



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


def genProfileFromMesh(a_obj_path, niters, a_mf_chirp, a_points_to_sample, a_scaling, a_streams,
                       a_standoff, a_fft_len, a_radarc, a_nsam, a_fs, a_fc, num_bounces=1, n_tris=2000000, a_mesh=None,
                       a_naz=8, a_nel=8):
    if a_mesh is not None:
        m_mesh = a_mesh
    else:
        try:
            m_mesh = readCombineMeshFile(a_obj_path, n_tris, scale=1 / a_scaling)
        except IndexError:
            print(f'{a_obj_path} not found.')

    if np.linalg.norm(m_mesh.get_center()) > 2.:
        m_mesh = m_mesh.translate(np.array([0, 0, 0.]), relative=False)

    _, poses, _, _ = calcPosBoresight(a_standoff[0], a_naz=a_naz, a_nel=a_nel)
    view_pos = poses[np.linspace(0, poses.shape[0] - 1, 32).astype(int)]
    m_scene = Scene()
    m_scene.add(Mesh(m_mesh, max_tris_per_split=64))
    m_sample_points = m_scene.sample(a_points_to_sample, view_pos=view_pos[::4], fc=a_fc, fs=a_fs,
                                     near_range_s=a_standoff[0] / c0, radar_equation_constant=a_radarc)

    # Apply random rotations and scalings for augmenting of training data
    for m_i in range(niters):
        chirp_idx = np.random.randint(0, len(a_mf_chirp))
        az_bw = MAX_AZ_BEAMWIDTH * np.random.rand() + DTR
        el_bw = MAX_EL_BEAMWIDTH * np.random.rand() + DTR
        block = np.zeros((len(a_standoff), poses.shape[0], a_fft_len), dtype=np.complex64)
        for k, s in enumerate(a_standoff):
            msrp, block[k] = getTargetProfile(m_scene, m_sample_points, s, a_streams, num_bounces,
                                              a_mf_chirp[chirp_idx], a_naz, a_nel, a_radarc, a_nsam, a_fs, a_fc,
                                              a_fft_len, az_bw, el_bw)
        yield msrp, block, m_i


def genProfileFromVCS(a_obj_path, a_standoff, a_niters, a_mf_chirp):
    # Load in the VCS file using the format reader
    m_scat_data, m_angles = readVCS(a_obj_path)

    # Generate target profile on CPU
    for m_i in range(a_niters):
        chirp_idx = np.random.randint(0, len(a_mf_chirp))
        block = np.zeros((len(a_standoff), n_az_samples * n_el_samples, fft_len), dtype=np.complex64)
        for k, s in enumerate(a_standoff):
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
        yield 7, block, m_i


def calcPosBoresight(a_standoff, a_pan=None, a_tilt=None, a_naz=8, a_nel=8):
    if a_pan is None:
        # Generate angles for block
        m_pans, m_tilts = np.meshgrid(np.linspace(0, 2 * np.pi, a_naz, endpoint=False),
                                  np.linspace(np.pi / 2 - .1, -np.pi / 2 + .1, a_nel))
        a_pan = m_pans.flatten()
        a_tilt = m_tilts.flatten()
    boresights = azelToVec(a_pan, a_tilt).T
    poses = -boresights * a_standoff
    m_near_range_s = (a_standoff - 100.) / c0
    return m_near_range_s, poses, a_pan, a_tilt


def loadClutterFiles():
    with open('clutter_files.txt', 'r') as f:
        clt = [c.strip() for c in f.readlines()]

    m_clutter_files = []
    for m_clut in clt:
        if int(m_clut.split('/')[4][:2]) >= 6 and np.any([Path(m_clut).parts[-1][:-4] in g and
                                                          ('png' in g or 'jpeg' in g)
                                                          for g in glob(f'{str(Path(m_clut).parent)}/*')]):
            try:
                xml_data = loadXMLFile(f'{m_clut[:-4]}.xml').find('SlimSDR_Configuration')
                if (xml_data['SlimSDR_Info']['System_Mode'] == 'SAR'
                        and 8e9 < xml_data['SlimSDR_Info']['Channel_0']['Center_Frequency_Hz'] < 32e9
                        and xml_data['SlimSDR_Info']['Gimbal_Settings']['Gimbal_Depression_Angle_D'] > 20.0):
                    m_clutter_files.append(m_clut)
            except FileNotFoundError:
                print(f'{m_clut} not found.')
                continue
            except Exception:
                print(f'{m_clut} has broken XML.')
                continue
        else:
            print(f'{m_clut} does not meet the requirements of the clutter parser.')
            continue
    return m_clutter_files


def loadClutterTargetSpectrum(a_clut, a_radarc, a_mesh=None, a_scaling=1., fft_len=8192):
    sdr_ch = load(a_clut, use_jump_correction=False)
    rp = SDRPlatform(sdr_ch)
    m_nsam, nr, ranges, ranges_sampled, near_range_s, granges, full_fft_len, up_fft_len = (
        rp.getRadarParams(0., .5, 1))

    downsample_rate = full_fft_len // fft_len
    downsample_fs = fs / downsample_rate
    downsample_nsam = m_nsam // downsample_rate

    flight_path = rp.rxpos(sdr_ch[0].pulse_time)
    bore = rp.boresight(rp.gpst).mean(axis=0)
    heading = np.arctan2(bore[0], bore[1])
    # The XML is not guaranteed to have a flight line, so we check for that
    try:
        flight_alt = sdr_ch.xml.Flight_Line.Flight_Line_Altitude_M
        gnd_exp_alt = flight_path.mean(axis=0)[2] - flight_alt
        mpos = flight_path.mean(axis=0) + bore * ranges.min()
        mpos[2] = gnd_exp_alt
        mrng = np.linalg.norm(mpos)
        mang = -np.arcsin(mpos[2] / mrng)
        m_it = 1
        while not ((ranges.max() > mrng > ranges.min()) and (
                rp.dep_ang + rp.el_half_bw > mang > rp.dep_ang - rp.el_half_bw)):
            mpos = flight_path.mean(axis=0) + bore * (ranges.min() + m_it)
            mpos[2] = gnd_exp_alt
            mrng = np.linalg.norm(mpos)
            mang = -np.arcsin(mpos[2] / mrng)
            m_it += 1
    except KeyError:
        # We don't know where the ground is, let's just project it into the abyss
        mpos = flight_path.mean(axis=0) + bore * ranges.mean()

    # Get pulse data and modify accordingly
    pulse = np.fft.fft(sdr_ch[0].cal_chirp, fft_len)
    mfilt = sdr_ch.genMatchedFilter(0, fft_len=fft_len)

    vecs = np.array([mpos[0] - flight_path[:, 0], mpos[1] - flight_path[:, 1],
                     mpos[2] - flight_path[:, 2]]).T
    pt_az = np.arctan2(vecs[:, 0], vecs[:, 1])
    max_pts = sdr_ch[0].frame_num[abs(pt_az - heading) < rp.az_half_bw]
    pulse_lims = [min(max_pts), max(max_pts)]
    pulse_lims[1] = min(pulse_lims[1],
                        pulse_lims[0] + cfig_settings.cpi_len * cfig_generate.iterations)

    # Locate the extrema to speed up the optimization
    if a_scaling > 0:
        scene = Scene()
        mmesh = a_mesh.translate(mpos, relative=False)
        scene.add(Mesh(mmesh))
        sample_points = scene.sample(points_to_sample,
                                     vecs[np.linspace(0, vecs.shape[0] - 1,
                                                      min(128, vecs.shape[0])).astype(int)])

    for frame in list(
            zip(*(iter(range(pulse_lims[0], pulse_lims[1] - cfig_settings.cpi_len, cfig_settings.cpi_len)),) * (
                    nstreams + 1))):
        txposes = [rp.txpos(sdr_ch[0].pulse_time[frame[n]:frame[n] + cfig_settings.cpi_len]).astype(_float) for n in
                   range(nstreams)]
        rxposes = [rp.rxpos(sdr_ch[0].pulse_time[frame[n]:frame[n] + cfig_settings.cpi_len]).astype(_float) for n in
                   range(nstreams)]
        pans = [rp.pan(sdr_ch[0].pulse_time[frame[n]:frame[n] + cfig_settings.cpi_len]).astype(_float) for n in
                range(nstreams)]
        tilts = [rp.tilt(sdr_ch[0].pulse_time[frame[n]:frame[n] + cfig_settings.cpi_len]).astype(_float) for n in
                 range(nstreams)]
        sdr_data = [sdr_ch.getPulses(sdr_ch[0].frame_num[frame[n]:frame[n] + cfig_settings.cpi_len], 0)[1] for n in
                    range(nstreams)]
        if row['scaling'] > 0:
            single_rp = getRangeProfileFromScene(scene, sample_points, txposes, rxposes, pans, tilts,
                                                 a_radarc, rp.az_half_bw, rp.el_half_bw, downsample_nsam,
                                                 cfig_generate.fc, near_range_s, fs=downsample_fs,
                                                 num_bounces=cfig_generate.num_bounces, streams=streams)
            tpsds = [np.fft.fft(srp, fft_len, axis=1) * mfilt * pulse for srp in single_rp]
        else:
            # Load in the VCS file using the format reader
            scat_data, angles = readVCS(obj_path)

            # Generate target profile on CPU
            tpsds = []
            for txpos, rxpos, pan, tilt in zip(txposes, rxposes, pans, tilts):
                bore_ang = np.array([pan, tilt]).T
                pd = np.zeros((txpos.shape[0], m_nsam), dtype=np.complex128)
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
            m_sdata = np.fft.fft(sdrd, fft_len, axis=0).T * mfilt
            if tpsd[tpsd != 0].std() == 0 or m_sdata[m_sdata != 0].std() == 0:
                continue
            # Unit energy and scale to have std of one
            m_ntpsd = normalize(tpsd) * 1e2
            m_sdata = normalize(m_sdata) * 1e2
            # Shift the data so it's centered around zero (for the autoencoder)
            if sdr_ch[0].baseband_fc != 0.:
                shift_bin = int(sdr_ch[0].baseband_fc / sdr_ch[0].fs * fft_len)
                m_ntpsd = np.roll(m_ntpsd, -shift_bin, 1)
                m_sdata = np.roll(m_sdata, -shift_bin, 1)
            m_ntpsd = formatTargetClutterData(m_ntpsd, fft_len)
            m_sdata = formatTargetClutterData(m_sdata, fft_len)
            yield m_ntpsd, m_sdata


def getTargetProfile(a_scene, a_sample_points, a_standoff, a_streams, a_num_bounces, a_mf_chirp, a_naz, a_nel, a_radarc,
                     a_nsam, a_fs, a_fc, a_fft_len, a_azbw, a_elbw):
    m_near_range_s, poses, m_pan, m_tilt = calcPosBoresight(a_standoff, a_naz=a_naz, a_nel=a_nel)
    m_rxposes = poses + 0.0
    m_single_rp, ray_origins, ray_directions, ray_powers = getRangeProfileFromScene(a_scene, a_sample_points, [poses.astype(_float)],
                                           [m_rxposes.astype(_float)], [m_pan.astype(_float)],
                                           [m_tilt.astype(_float)], a_radarc, a_azbw, a_elbw, a_nsam,
                                           a_fc, m_near_range_s, fs=a_fs, num_bounces=a_num_bounces,
                                           streams=a_streams, debug=True)
    msrp = m_single_rp[
        0]  # + (np.random.randn(*m_single_rp[0].shape) + 1j * np.random.randn(*m_single_rp[0].shape)) * abs(m_single_rp[0][m_single_rp[0] > 0.]).mean() / 10
    return msrp, np.fft.fft(msrp, a_fft_len, axis=1) * a_mf_chirp


def processTargetProfile(a_pd, a_fft_len, a_tsvd):
    pd_mask = np.sqrt(np.sum(a_pd * a_pd.conj(), axis=-1).real) > 0.
    a_pd[pd_mask] = normalize(a_pd[pd_mask]) * 1e2
    ppd_cat = np.concatenate([formatTargetClutterData(p, a_fft_len) for p in a_pd], axis=1)

    if np.all(ppd_cat == 0.):
        return None

    return np.stack([a_tsvd.fit_transform(t) for t in ppd_cat.swapaxes(0, 1)])
    # return np.stack([a_tsvd.fit_transform(t) for t in ppd_cat[:, :, abs(ppd_cat[0, 0]) > 0].swapaxes(0, 1)])


if __name__ == '__main__':
    cuda.select_device(0)

    rx_gain = 40  # dB
    tx_gain = 40  # dB
    rec_gain = 100  # dB
    ant_transmit_power = 100  # watts
    points_to_sample = 2**16
    nsam = 4600
    nstreams = 1
    fs = 2e9
    config = load_yaml_config('./vae_config.yaml')
    cfig_settings = config.settings
    cfig_generate = config.generate_data_settings

    standoff = [config.apache_params.vehicle_slant_range_min, config.apache_params.vehicle_slant_range_max]
    n_az_samples = cfig_generate.n_az_samples
    n_el_samples = cfig_generate.n_el_samples
    wavenumber = 2 * np.pi * cfig_generate.fc / c0

    save_path = cfig_generate.local_path if (
        config).generate_data_settings.use_local_storage else config.dataset_params.data_path


    clutter_files = loadClutterFiles()
    target_info = pd.read_csv('./data/target_info.csv')

    tsvd = TruncatedSVD(n_components=cfig_generate.n_el_samples * cfig_generate.n_az_samples)

    streams = [cuda.stream() for _ in range(nstreams)]

    # Standardize the FFT length for training purposes (this may cause data loss)
    fft_len = cfig_settings.fft_len

    # This is all the constants in the radar equation for this simulation
    radar_coeff = get_radar_coeff(cfig_generate.fc, ant_transmit_power, rx_gain, tx_gain, rec_gain)

    # Generate chirps with random bandwidths, pulse lengths
    bws = BASE_BANDWIDTH + np.random.rand(10) * BANDWIDTH_RANGE
    plens = (BASE_PULSE_LENGTH + np.random.rand(10) * (nsam / 2) / fs) * fs
    chirps = []

    # Get matched filters for the chirps
    mf_chirp = []
    for bw, plen in zip(bws, plens):
        twin = taylor(int(np.round(bw / fs * fft_len)))
        taytay = np.zeros(fft_len, dtype=np.complex128)
        winloc = int((cfig_generate.fc % fs) * fft_len / fs) - len(twin) // 2
        chirps.append(genChirp(int(plen), fs, cfig_generate.fc, bw))
        if winloc + len(twin) > fft_len:
            taytay[winloc:fft_len] += twin[:fft_len - winloc]
            taytay[:len(twin) - (fft_len - winloc)] += twin[fft_len - winloc:]
        else:
            taytay[winloc:winloc + len(twin)] += twin
        mf_chirp.append(np.fft.fft(chirps[-1], fft_len) * taytay / np.fft.fft(chirps[-1], fft_len))

    abs_idx = 0
    abs_clutter_idx = 0

    for tidx, row in target_info.iterrows():
        obj_path = f'{config.generate_data_settings.obj_path}/{row["filename"]}'

        # FIRST CODE BLOCK: Generate target representative blocks for an autoencoder to compress
        # Saves them out to a file
        if cfig_generate.run_target:
            tensor_target_path = f"{config.target_exp_params.dataset_params.data_path}/target_{tidx}"
            if not Path(tensor_target_path).exists():
                os.mkdir(tensor_target_path)
            iterations = cfig_generate.iterations
            if not cfig_generate.overwrite_files:
                # Determine next index
                pre_file_idx = max(
                    int(Path(g).name.split('_')[-1])
                    for g in glob(f'{tensor_target_path}/*.pt')
                )
                if pre_file_idx - abs_idx < iterations:
                    iterations -= pre_file_idx - abs_idx
                    abs_idx = pre_file_idx + 1
                else:
                    # Skip the target if we have enough files for it already
                    abs_idx = pre_file_idx + 1
                    continue

            # Generate a Nrange_samples x Nangles x fft_len block for the autoencoder, iterated niters times to get different
            # bandwidths and pulse lengths.
            if row['scaling'] > 0:
                gen_iter = iter(genProfileFromMesh(obj_path, iterations, mf_chirp,
                                                   points_to_sample, row['scaling'], streams, standoff, fft_len,
                                                   radar_coeff, nsam, fs, cfig_generate.fc,
                                                   cfig_generate.num_bounces, a_naz=n_az_samples, a_nel=n_el_samples))
            else:
                gen_iter = iter(genProfileFromVCS(obj_path, standoff, iterations, mf_chirp))


            pd_cats = []
            for rprof, pd, i in gen_iter:
                if np.all(pd == 0):
                    print(f'Skipping on target {tidx}, pd {i}')
                    continue

                pd_cat = processTargetProfile(pd, fft_len, tsvd)

                # Append to master target list
                if cfig_generate.save_files and pd_cat is not None:
                    # Save the block out to a torch file for the dataloader later
                    torch.save([torch.tensor(pd_cat, dtype=torch.float32), tidx],
                               f"{tensor_target_path}/target_{tidx}_{abs_idx}.pt")
                    abs_idx += 1
                else:
                    pd_cats.append(pd_cat)

        if cfig_generate.run_clutter:
            tensor_clutter_path = f"{config.target_exp_params.dataset_params.data_path}/clutter_tensors"
            if not Path(tensor_clutter_path).exists():
                os.mkdir(tensor_clutter_path)
            print(f'Saving clutter spec for target {row["name"]}')
            if row['scaling'] > 0:
                mesh = readCombineMeshFile(obj_path, points_to_sample, scale=1 / row['scaling'])
            for clut in clutter_files:
                # Load the sar file that the clutter came from
                for ntpsd, sdata in loadClutterTargetSpectrum(clut, radar_coeff, mesh, row['scaling']):
                    if cfig_generate.save_files:
                        for nt, sd in zip(ntpsd, sdata):
                            if not np.any(np.isnan(nt)) and not np.any(np.isnan(sd)):
                                torch.save([torch.tensor(sd, dtype=torch.float32),
                                            torch.tensor(nt, dtype=torch.float32), tidx],
                                           f"{tensor_clutter_path}/tc_{abs_clutter_idx}.pt")
                                abs_clutter_idx += 1
