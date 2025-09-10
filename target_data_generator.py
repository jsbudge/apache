from glob import glob
import os
import numpy as np
from pathlib import Path
from numba import cuda
from simulib.simulation_functions import azelToVec, genChirp
from simulib.mesh_functions import getRangeProfileFromScene
from simulib.platform_helper import SDRPlatform
from simulib.utils import _float, _complex_float
from scipy.signal.windows import taylor, tukey
import plotly.io as pio
from tqdm import tqdm
import yaml
from sdrparse.SDRParsing import load, loadXMLFile
import torch
import matplotlib as mplib
import pickle
from config import load_yaml_config
mplib.use('TkAgg')
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from simulib.simulation_functions import db
import pandas as pd
from utils import scale_normalize, get_radar_coeff, normalize

pio.renderers.default = 'browser'

c0 = 299792458.0
fs = 2e9
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
BASE_BANDWIDTH = 150e6
BANDWIDTH_RANGE = 1e9
BASE_PULSE_LENGTH = 1e-7
MAX_AZ_BEAMWIDTH = 25 * DTR
MAX_EL_BEAMWIDTH = 25 * DTR



def formatTargetClutterData(data: np.ndarray, bin_bandwidth: int):
    split = np.zeros((data.shape[0], 2, bin_bandwidth), dtype=_float)
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


def genProfileFromMesh(a_scene, niters, a_mf_chirp, a_points_to_sample, a_supersamples,
                       a_standoff, a_fft_len, a_radarc, a_nsam, a_fs, a_fc, num_bounces=1,
                       a_naz=8, a_nel=8):

    _, poses, _, _ = calcPosBoresight(a_standoff[0], a_naz=a_naz, a_nel=a_nel)
    m_sample_points = a_scene.sample(a_points_to_sample, np.ascontiguousarray(poses[::10]))

    # Apply random rotations and scalings for augmenting of training data
    for m_i in range(niters):
        chirp_idx = np.random.randint(0, len(a_mf_chirp))
        az_bw = MAX_AZ_BEAMWIDTH * np.random.rand() + DTR
        el_bw = MAX_EL_BEAMWIDTH * np.random.rand() + DTR
        block = np.zeros((len(a_standoff), poses.shape[0], a_fft_len), dtype=_complex_float)
        for k, s in enumerate(a_standoff):
            msrp, block[k] = getTargetProfile(a_scene, m_sample_points, s, a_supersamples, num_bounces,
                                              a_mf_chirp[chirp_idx], a_naz, a_nel, a_radarc, a_nsam, a_fs, a_fc,
                                              a_fft_len, az_bw, el_bw)
        yield msrp, block, m_i


def genProfileFromVCS(a_obj_path, a_standoff, a_niters, a_mf_chirp):
    # Load in the VCS file using the format reader
    m_scat_data, m_angles = readVCS(a_obj_path)

    # Generate target profile on CPU
    for m_i in range(a_niters):
        chirp_idx = np.random.randint(0, len(a_mf_chirp))
        block = np.zeros((len(a_standoff), n_az_samples * n_el_samples, fft_len), dtype=_complex_float)
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
                if (xml_data.SlimSDR_Info.System_Mode == 'SAR'
                        and 8e9 < xml_data.SlimSDR_Info.Channel_0.Center_Frequency_Hz < 32e9
                        and xml_data.SlimSDR_Info.Gimbal_Settings.Gimbal_Depression_Angle_D > 20.0):
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


def loadClutterTargetSpectrum(a_clut, a_radarc, scene, iterations, seq_min, seq_max, a_points_to_sample, fc,
                              num_bounces, fft_len=8192):
    sdr_ch = load(a_clut, import_pickle=True, use_jump_correction=False)
    rp = SDRPlatform(sdr_ch)
    m_nsam, nr, ranges, ranges_sampled, near_range_s, granges, full_fft_len, up_fft_len = (
        rp.getRadarParams(0., .21, 1))
    scene.shift(np.array([0., 0., 0.]), relative=False)

    downsample_rate = full_fft_len // fft_len

    flight_path = rp.rxpos(sdr_ch[0].pulse_time)

    sample_points = scene.sample(a_points_to_sample, a_obs_pts=flight_path[::10])

    # Get pulse data and modify accordingly
    pulse = np.fft.fft(sdr_ch[0].cal_chirp, full_fft_len)
    mfilt = sdr_ch.genMatchedFilter(0, fft_len=full_fft_len)
    for _ in range(iterations):
        pt_pt = np.random.choice(sdr_ch[0].nframes - seq_max)
        frame_idxes = np.arange(pt_pt, pt_pt + np.random.choice(np.arange(seq_min, seq_max)))
        pulse_times = sdr_ch[0].pulse_time[frame_idxes]

        # Place the model in the scene at a random location
        proj_rbin = np.random.choice(np.arange(len(ranges_samples)))
        proj_rng = ranges_samples[proj_rbin]
        r_d = azelToVec(rp.pan(pulse_times).mean() + np.random.normal() * rp.az_half_bw,
                        rp.tilt(pulse_times).mean() + np.random.normal() * rp.el_half_bw)
        floc = rp.rxpos(pulse_times).mean(axis=0)

        mpos = floc + r_d * proj_rng

        txposes = [rp.txpos(pulse_times).astype(_float)]
        rxposes = [rp.rxpos(pulse_times).astype(_float)]
        pans = [rp.pan(pulse_times).astype(_float)]
        tilts = [rp.tilt(pulse_times).astype(_float)]
        sdr_data = [sdr_ch.getPulses(sdr_ch[0].frame_num[frame_idxes], 0)[1]]

        scene.shift(mpos, relative=False)
        shifted_points = [sample_points + scene.center]

        single_rp = getRangeProfileFromScene(scene, shifted_points, txposes, rxposes, pans, tilts,
                                             a_radarc, rp.az_half_bw, rp.el_half_bw, m_nsam,
                                             fc, near_range_s, fs=fs,
                                             num_bounces=num_bounces, supersamples=0)
        tpsds = [[np.fft.fft(s[:, ::downsample_rate], fft_len, axis=1) * mfilt[::downsample_rate] * pulse[::downsample_rate] for s in srp] for srp in single_rp][0]
        for sdrd, tpsd in zip(sdr_data, tpsds):
            if len(tpsd[tpsd != 0]) == 0:
                continue
            m_sdata = np.fft.fft(sdrd, fft_len, axis=0).T * mfilt[::downsample_rate]
            if tpsd[tpsd != 0].std() == 0 or m_sdata[m_sdata != 0].std() == 0:
                continue

            sdtime = np.fft.ifft(m_sdata, axis=1)
            tptime = np.fft.ifft(tpsd, axis=1)
            # dbtime = db(np.fft.ifft(tpsd, axis=1))
            up_fac = (abs(sdtime).mean(axis=1) + abs(sdtime).std(axis=1) * 6) / np.max(abs(tptime), axis=1)
            comb = sdtime + tptime * up_fac[:, None]

            comb = np.fft.fft(comb, axis=1)

            '''plt.figure()
            plt.subplot(3, 1, 1)
            plt.title('sd')
            plt.imshow(db(sdtime))
            plt.axis('tight')
            plt.subplot(3, 1, 2)
            plt.title('sd')
            plt.imshow(db(tptime))
            plt.axis('tight')
            plt.subplot(3, 1, 3)
            plt.title('sd')
            plt.imshow(db(comb))
            plt.axis('tight')'''
            # Unit energy and scale to have std of one based on the original
            norm_factor = np.expand_dims(np.sqrt(np.sum(m_sdata * m_sdata.conj(), axis=-1).real), axis=len(m_sdata.shape) - 1)
            m_ntpsd = comb / norm_factor
            m_sdata = m_sdata / norm_factor
            # Shift the data so it's centered around zero (for the autoencoder)
            if sdr_ch[0].baseband_fc != 0.:
                shift_bin = int(sdr_ch[0].baseband_fc / sdr_ch[0].fs * fft_len)
                m_ntpsd = np.roll(m_ntpsd, -shift_bin, 1)
                m_sdata = np.roll(m_sdata, -shift_bin, 1)
            m_ntpsd = formatTargetClutterData(m_ntpsd, fft_len)
            m_sdata = formatTargetClutterData(m_sdata, fft_len)
            yield m_ntpsd, m_sdata, proj_rbin


def getTargetProfile(a_scene, a_sample_points, a_standoff, a_supersamples, a_num_bounces, a_mf_chirp, a_naz, a_nel, a_radarc,
                     a_nsam, a_fs, a_fc, a_fft_len, a_azbw, a_elbw):
    m_near_range_s, poses, m_pan, m_tilt = calcPosBoresight(a_standoff, a_naz=a_naz, a_nel=a_nel)
    m_rxposes = poses + 0.0
    m_single_rp, ray_origins, ray_directions, ray_powers = getRangeProfileFromScene(a_scene, [a_sample_points], [poses.astype(_float)],
                                           [m_rxposes.astype(_float)], [m_pan.astype(_float)],
                                           [m_tilt.astype(_float)], a_radarc, a_azbw, a_elbw, a_nsam,
                                           a_fc, m_near_range_s, fs=a_fs, num_bounces=a_num_bounces,
                                           supersamples=a_supersamples, debug=True)
    msrp = m_single_rp[0][0]
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
    target_info = glob(f'{cfig_generate.obj_path}/*.model')
    target_csv = pd.DataFrame(columns=['name'])
    # target_info = pd.read_csv('./data/target_info.csv')

    tsvd = TruncatedSVD(n_components=cfig_generate.n_el_samples * cfig_generate.n_az_samples)

    # Standardize the FFT length for training purposes (this may cause data loss)
    fft_len = cfig_settings.fft_len

    # This is all the constants in the radar equation for this simulation
    radar_coeff = get_radar_coeff(cfig_generate.fc, cfig_generate.ant_transmit_power, cfig_generate.rx_gain,
                                  cfig_generate.tx_gain, cfig_generate.rec_gain)

    # Generate chirps with random bandwidths, pulse lengths
    bws = BASE_BANDWIDTH + np.random.rand(10) * BANDWIDTH_RANGE
    plens = (BASE_PULSE_LENGTH + np.random.rand(10) * (cfig_generate.nsam / 2) / fs) * fs
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

    for tidx, tname in enumerate(target_info):
        with open(tname, 'rb') as f:
            model = pickle.load(f)

        # FIRST CODE BLOCK: Generate target representative blocks for an autoencoder to compress
        # Saves them out to a file
        if cfig_generate.run_target:
            print(f'Running target {tname}')
            tensor_target_path = f"{config.target_exp_params.dataset_params.data_path}/target_{tidx}"
            if not Path(tensor_target_path).exists():
                os.mkdir(tensor_target_path)
            target_csv.loc[tidx, 'name'] = tname
            iterations = cfig_generate.iterations
            if not cfig_generate.overwrite_files:
                nfiles_in_path = len(glob(f'{tensor_target_path}/*.pt'))
                iterations -= nfiles_in_path
                abs_idx += nfiles_in_path
                if iterations <= 0:
                    continue

            # Generate a Nrange_samples x Nangles x fft_len block for the autoencoder, iterated niters times to get different
            # bandwidths and pulse lengths.
            if True:
                gen_iter = iter(genProfileFromMesh(model, iterations, mf_chirp,
                                                   2**cfig_generate.num_sample_points_power, cfig_generate.supersamples, standoff, fft_len,
                                                   radar_coeff, cfig_generate.nsam, fs, cfig_generate.fc,
                                                   cfig_generate.num_bounces, a_naz=n_az_samples, a_nel=n_el_samples))
            else:
                gen_iter = iter(genProfileFromVCS(obj_path, standoff, iterations, mf_chirp))


            pd_cats = []
            for rprof, pd, i in tqdm(gen_iter):
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
            print(f'Clutter spec for target {tname}')
            for clut in clutter_files:
                # Load the sar file that the clutter came from
                print(f'{clut}')
                for ntpsd, sdata, rbin in loadClutterTargetSpectrum(clut, radar_coeff, model, cfig_generate.clutter_iterations,
                                                                cfig_generate.seq_min_length, cfig_generate.seq_max_length,
                                                              2**cfig_generate.num_sample_points_power, cfig_generate.fc, cfig_generate.num_bounces):
                    '''plt.figure(f'{clut}')
                    plt.plot(db(ntpsd[0, 0]))
                    plt.plot(db(sdata[0, 0]))'''
                    if cfig_generate.save_files and (not np.any(np.isnan(ntpsd)) and not np.any(np.isnan(sdata))):
                        # Generate a list with sdata (sdr pulses), ntpsd (target profile added to sdr pulse) and tidx (target index)
                        torch.save([torch.tensor(sdata, dtype=torch.float32),
                                    torch.tensor(ntpsd, dtype=torch.float32), rbin, tidx],
                                   f"{tensor_clutter_path}/tc_{abs_clutter_idx}.pt")
                        abs_clutter_idx += 1

    # Grab all the finished target representations and get the mean and std
    targ_files = glob(f"{config.target_exp_params.dataset_params.data_path}/target_*/*.pt", recursive=True)
    ex_mu = torch.tensor(np.zeros(4))
    ex_m2 = torch.tensor(np.zeros(4))
    ex_count = torch.tensor(np.zeros(4))
    for targ in targ_files:
        tdata = torch.load(targ)
        ex, idx = tdata
        ex = ex.reshape((4, -1)).T
        for vec in ex:
            ex_count += 1
            delta = vec - ex_mu
            ex_mu += delta / ex_count
            delta2 = vec - ex_mu
            ex_m2 += delta * delta2
    ex_var = ex_m2 / (ex_count - 1)

