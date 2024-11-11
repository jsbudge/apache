from glob import glob
import os
import numpy as np
from pathlib import Path
from numba import cuda
from simulib import db, azelToVec, genChirp, getElevation, enu2llh
from simulib.mesh_functions import readCombineMeshFile, getBoxesSamplesFromMesh, getRangeProfileFromMesh
from simulib.platform_helper import SDRPlatform
from scipy.signal.windows import taylor
import matplotlib.pyplot as plt
import plotly.io as pio
from tqdm import tqdm
import yaml
from sdrparse.SDRParsing import load, loadXMLFile
import torch

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
TARGET_PROFILE_MIN_BEAMWIDTH = 0.19634954
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


def genProfileFromMesh(obj_path, niters, mf_chirp, nboxes, points_to_sample, scaling, streams, n_tris=20000):
    try:
        mesh = readCombineMeshFile(obj_path, n_tris, scale=1 / scaling)
    except IndexError:
        print(f'{tobj} not found.')

    if np.linalg.norm(mesh.get_center()) > 2.:
        mesh = mesh.translate(np.array([0, 0, 0.]), relative=False)

    _, poses, _, _ = calcPosBoresight(standoff)
    view_pos = poses[np.linspace(0, poses.shape[0] - 1, 32).astype(int)]

    box_tree, sample_points = getBoxesSamplesFromMesh(mesh, num_box_levels=nboxes, sample_points=points_to_sample,
                                                      view_pos=view_pos, use_box_pts=True)

    # Apply random rotations and scalings for augmenting of training data
    for i in range(niters):
        chirp_idx = np.random.randint(0, len(mf_chirp))
        near_range_s, poses, pan, tilt = calcPosBoresight(standoff + np.random.rand() * 100)

        single_rp = getRangeProfileFromMesh(*box_tree, sample_points, [poses], [pan], [tilt], radar_coeff, 30 * DTR,
                                            30 * DTR, nsam, fc, near_range_s, num_bounces=num_bounces, streams=streams)
        yield np.fft.fft(single_rp[0], fft_len, axis=1) * mf_chirp[chirp_idx], i


def genProfileFromVCS(obj_path, niters, mf_chirp):
    # Load in the VCS file using the format reader
    scat_data, angles = readVCS(obj_path)

    # Generate target profile on CPU
    for i in range(niters):
        chirp_idx = np.random.randint(0, len(mf_chirp))
        near_range_s, poses, bore_az, bore_el = calcPosBoresight(standoff)
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
    return near_range_s, poses, pan, tilt


if __name__ == '__main__':
    cuda.select_device(1)

    rx_gain = 22  # dB
    tx_gain = 22  # dB
    rec_gain = 100  # dB
    ant_transmit_power = 100  # watts
    fc = 9.6e9
    nbox_levels = 4
    points_to_sample = 2000
    num_bounces = 1
    nbounce_ray = 2
    standoff = 500.
    nsam = 5678
    niters = 10
    nstreams = 1

    with open('./vae_config.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            
    wavenumber = 2 * np.pi * fc / c0

    save_path = config['generate_data_settings']['local_path'] if (
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
    fft_len = config['settings']['fft_len']

    # This is all the constants in the radar equation for this simulation
    radar_coeff = (
                c0 ** 2 / fc ** 2 * ant_transmit_power * 10 ** ((rx_gain + 2.15) / 10) * 10 ** ((tx_gain + 2.15) / 10) *
                10 ** ((rec_gain + 2.15) / 10) / (4 * np.pi) ** 3)

    near_range_s, poses, pan, tilt = calcPosBoresight(standoff)
    plt.ion()

    # Generate chirps with random bandwidths, pulse lengths
    bws = 150e6 + np.random.rand(10) * 1e9
    plens = (1e-7 + np.random.rand(10) * (nsam / 2) / fs) * fs
    chirps = []

    # Get matched filters for the chirps
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

    abs_idx = 0
    abs_clutter_idx = 0

    for tidx, (tobj, scaling) in tqdm(enumerate(target_obj_files)):
        obj_path = f'{config["generate_data_settings"]["obj_path"]}/{tobj}'
        if config['generate_data_settings']['save_as_target']:
            tensor_target_path = f"{config['target_exp_params']['dataset_params']['data_path']}/target_{tidx}"
            if not Path(tensor_target_path).exists():
                os.mkdir(tensor_target_path)
            if scaling > 0:
                gen_iter = iter(genProfileFromMesh(obj_path, niters, mf_chirp, nbox_levels, points_to_sample, scaling, streams, n_tris=50000))
            else:
                gen_iter = iter(genProfileFromVCS(obj_path, niters, mf_chirp))


            for pd, i in gen_iter:
                if np.all(pd == 0):
                    print(f'Skipping on target {tidx}, pd {i}')
                    continue
                pd = pd[np.any(pd, axis=1)]
                pd = pd / np.sqrt(np.sum(pd * pd.conj(), axis=1))[:, None]
                pd[pd != 0] = (pd[pd != 0] - pd[pd != 0].mean()) / pd[pd != 0].std()
                pd_cat = formatTargetClutterData(pd, fft_len)

                if i == 0:
                    target_id_list.append(tobj)
                for p in pd_cat:
                    torch.save([torch.tensor(p, dtype=torch.float32), tidx],
                               f"{tensor_target_path}/target_{tidx}_{abs_idx}.pt")
                    abs_idx += 1

        if config['generate_data_settings']['save_as_clutter']:
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
                mesh = mesh.translate(mpos, relative=False)
                box_tree, sample_points = getBoxesSamplesFromMesh(mesh, num_box_levels=nbox_levels,
                                                                  sample_points=points_to_sample)

                vecs = np.array([mpos[0] - flight_path[:, 0], mpos[1] - flight_path[:, 1],
                                 mpos[2] - flight_path[:, 2]]).T
                pt_az = np.arctan2(vecs[:, 0], vecs[:, 1])
                max_pts = sdr_ch[0].frame_num[abs(pt_az - heading) < rp.az_half_bw]
                pulse_lims = [min(max_pts), max(max_pts)]
                pulse_lims[1] = min(pulse_lims[1], pulse_lims[0] + config['settings']['cpi_len'] * config['generate_data_settings']['iterations'])

                # Get pulse data and modify accordingly
                pulse = np.fft.fft(sdr_ch[0].cal_chirp, fft_len)
                mfilt = sdr_ch.genMatchedFilter(0, fft_len=fft_len)
                valids = mfilt != 0

                for frame in list(zip(*(iter(range(pulse_lims[0], pulse_lims[1] - config['settings']['cpi_len'], config['settings']['cpi_len'])),) * (nstreams + 1))):
                    txposes = [rp.txpos(sdr_ch[0].pulse_time[frame[n]:frame[n + 1]]).astype(np.float64) for n in
                               range(nstreams)]
                    rxposes = [rp.txpos(sdr_ch[0].pulse_time[frame[n]:frame[n + 1]]).astype(np.float64) for n in
                               range(nstreams)]
                    pans = [rp.pan(sdr_ch[0].pulse_time[frame[n]:frame[n + 1]]).astype(np.float64) for n in
                            range(nstreams)]
                    tilts = [rp.tilt(sdr_ch[0].pulse_time[frame[n]:frame[n + 1]]).astype(np.float64) for n in
                             range(nstreams)]
                    sdr_data = [sdr_ch.getPulses(sdr_ch[0].frame_num[frame[n]:frame[n + 1]], 0)[1] for n in range(nstreams)]
                    if scaling > 0:
                        single_rp = getRangeProfileFromMesh(*box_tree, sample_points, txposes, pans, tilts,
                                                radar_coeff, rp.az_half_bw, rp.el_half_bw, nsam, fc, near_range_s,
                                                num_bounces=num_bounces, streams=streams)
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
                                rvec = bl[:, :3] - txpos[n]
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
                        nt_den = np.sqrt(np.sum(tpsd * tpsd.conj(), axis=1))
                        nt_den[nt_den == 0] = 1
                        ntpsd = tpsd / nt_den[:, None]
                        ntpsd[ntpsd != 0] = (ntpsd[ntpsd != 0] - ntpsd[ntpsd != 0].mean()) / ntpsd[ntpsd != 0].std()
                        sd_den = np.sqrt(np.sum(sdata * sdata.conj(), axis=1))
                        sd_den[sd_den == 0] = 1
                        sdata = sdata / sd_den[:, None]
                        sdata[sdata != 0] = (sdata[sdata != 0] - sdata[sdata != 0].mean()) / sdata[sdata != 0].std()
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