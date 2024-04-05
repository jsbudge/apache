from scipy.stats import rayleigh
import numpy as np
from data_converter.SDRParsing import SDRParse, load, loadXMLFile
from tqdm import tqdm
from glob import glob
from pathlib import Path
import yaml

fs = 2e9
c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
m_to_ft = 3.2808


def genTargetPSD(bw, fc, rng_min, rng_max, spec_sz, fs, sz_m=15, alpha=None):
    """
    Generates a target power spectral density using a bunch of random params
    :param sz_m: (float) Radial size of the target in meters.
    :param alpha: (ndarray) Array of shape parameters. Must be 0, .5, or 1 in each element.
    :return: Normalized power spectral density.
    """

    # Number of bins occupied by target
    M = int(2 * sz_m * bw / c0)
    freqs = np.fft.fftfreq(spec_sz, 1 / fs)
    # Range of target
    rng = np.random.uniform(rng_min, rng_max) + c0 / (2 * bw) * np.arange(M)
    # Shape parameters fo individual scatterers
    alpha = np.random.choice([0, .5, 1], M) if alpha is None else alpha
    # Complex electrical field amplitude
    Am = np.random.rand(M) + 1j * np.random.rand(M)
    # Get a center frequency for the target response
    t_fc = fc + bw / 2 * np.random.uniform(-1, 1)
    # Overall spectrum of target response given the above parameters
    psd = np.sum([Am[n] / rng[n] ** 4 * (1j * freqs / t_fc) ** alpha[n] *
                  np.exp(-1j * 4 * np.pi * freqs / c0 * rng[n]) for n in range(M)], axis=0)
    return psd / np.linalg.norm(psd)


def genTargetPSDSwerling1(bw, fc, rng_min, rng_max, spec_sz, fs, cpi_len, fft_sz, sz_m=15, alpha=None, chirp=None):
    """
    Generates a target power spectral density using a bunch of random params
    :param sz_m: (float) Radial size of the target in meters.
    :param alpha: (ndarray) Array of shape parameters. Must be 0, .5, or 1 in each element.
    :return: Normalized power spectral density.
    """
    # Generate over a cpi of pulses
    psd = np.zeros((cpi_len, fft_sz), dtype=np.complex128)
    # Number of bins occupied by target
    M = int(2 * sz_m * bw / c0)
    freqs = np.fft.fftfreq(spec_sz, 1 / fs)
    norm_energy = 0.
    for n in range(cpi_len):
        # Range of target
        rng = np.random.uniform(rng_min, rng_max) + c0 / (2 * bw) * np.arange(M)
        # Shape parameters fo individual scatterers
        alpha = np.random.choice([0, .5, 1], M) if alpha is None else alpha
        # Complex electrical field amplitude
        Am = rayleigh.rvs(size=M) * (np.random.rand(M) + 1j * np.random.rand(M))
        # Get a center frequency for the target response
        t_fc = fc + bw / 2 * np.random.uniform(-1, 1)
        # Overall spectrum of target response given the above parameters
        psd[n, :] = np.sum([Am[n] / rng[n] ** 4 * (1j * freqs / t_fc) ** alpha[n] *
                            np.exp(-1j * 4 * np.pi * freqs / c0 * rng[n]) for n in range(M)], axis=0)
        if chirp is not None:
            psd[n, :] *= chirp
        # Normalize based on the first pulse in the spectrum
        if n == 0:
            norm_energy = np.sqrt(np.sum(abs(psd[n, :] * psd[n, :].conj())))
        psd[n, :] /= norm_energy
    return psd


def formatTargetClutterData(data: np.ndarray, bin_bandwidth: int):
    split = np.zeros((data.shape[0], bin_bandwidth, 2), dtype=np.float64)
    split[:, :bin_bandwidth // 2, :] = data[:, -bin_bandwidth // 2:, :]
    split[:, -bin_bandwidth // 2:, :] = data[:, :bin_bandwidth // 2, :]
    return split


def getVAECov(data: np.ndarray, mfilt: np.ndarray = None, rollback: int = 0,
              nsam: int = 0, fft_len: int = 32768, mu: float = 0., var: float = 1., apply_mfilt: bool = True):
    if apply_mfilt:
        pulses = np.fft.fft(data, fft_len, axis=0) * mfilt[:, None]
        # If the pulses are offset video, shift to be centered around zero
        pulses = np.roll(pulses, rollback, axis=0)
    else:
        pulses = np.fft.fft(data, fft_len, axis=0)
    norm_energy = np.sqrt(np.sum(abs(pulses[:, 0] * pulses[:, 0].conj())))
    pulses /= norm_energy  # Normalize everything to the first pulse
    pmean = pulses.mean(axis=1)
    cov_dt = np.corrcoef(pulses.T)
    return (np.stack((pmean.real, pmean.imag), axis=1),
            np.stack(((cov_dt.real - mu) / var, (cov_dt.imag - mu) / var), axis=2))


sdr_fnmes = ['/data6/SAR_DATA/2023/06072023/SAR_06072023_111559.sar',
             '/data6/SAR_DATA/2023/06072023/SAR_06072023_154802.sar',
             '/data6/SAR_DATA/2023/06202023/SAR_06202023_135617.sar',
             '/data6/SAR_DATA/2023/06202023/SAR_06202023_140503.sar',
             '/data6/SAR_DATA/2023/07112023/SAR_07112023_164706.sar',
             '/data6/SAR_DATA/2023/07122023/SAR_07122023_104842.sar',
             '/data6/SAR_DATA/2023/07122023/SAR_07122023_105424.sar',
             '/data6/SAR_DATA/2023/08092023/SAR_08092023_115234.sar',
             '/data6/SAR_DATA/2023/08092023/SAR_08092023_143927.sar',
             '/data6/SAR_DATA/2023/08092023/SAR_08092023_144029.sar',
             '/data6/SAR_DATA/2023/08092023/SAR_08092023_144437.sar',
             '/data6/SAR_DATA/2023/08102023/SAR_08102023_150617.sar',
             '/data6/SAR_DATA/2023/08222023/SAR_08222023_083700.sar',
             '/data6/SAR_DATA/2023/08222023/SAR_08222023_084158.sar',
             '/data6/SAR_DATA/2023/08222023/SAR_08222023_090159.sar',
             '/data6/SAR_DATA/2023/08222023/SAR_08222023_121923.sar',
             '/data6/SAR_DATA/2023/08222023/SAR_08222023_122248.sar',
             '/data6/SAR_DATA/2023/08232023/SAR_08232023_091003.sar',
             '/data6/SAR_DATA/2023/08232023/SAR_08232023_110946.sar',
             '/data6/SAR_DATA/2023/08232023/SAR_08232023_112652.sar',
             '/data6/SAR_DATA/2023/08232023/SAR_08232023_115317.sar',
             '/data6/SAR_DATA/2023/08232023/SAR_08232023_150257.sar',
             '/data6/SAR_DATA/2023/09112023/SAR_09112023_151139.sar',
             '/data6/SAR_DATA/2023/09112023/SAR_09112023_151257.sar',
             '/data6/SAR_DATA/2023/09122023/SAR_09122023_152050.sar',
             '/data6/SAR_DATA/2023/09122023/SAR_09122023_152903.sar',
             '/data6/SAR_DATA/2023/09122023/SAR_09122023_153015.sar',
             '/data6/SAR_DATA/2023/09132023/SAR_09132023_115432.sar',
             '/data6/SAR_DATA/2023/10042023/SAR_10042023_135851.sar',
             '/data6/SAR_DATA/2023/10062023/SAR_10062023_112753.sar',
             '/data6/SAR_DATA/2023/10232023/SAR_10232023_093009.sar',
             '/data6/SAR_DATA/2023/10242023/SAR_10242023_113149.sar']


if __name__ == '__main__':
    with open('./vae_config.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # sdr_fnmes = glob('/data6/SAR_DATA/2023/**/*.sar')
    sdr_file = []
    save_path = config['generate_data_settings']['local_path'] if (
        config)['generate_data_settings']['use_local_storage'] else config['dataset_params']['data_path']
    for s in sdr_fnmes:
        if Path(f'{save_path}/clutter_{s.split("/")[-1].split(".")[0]}.cov').exists():
            print(f'{s} already has a .cov file.')
            continue
        if int(s.split('/')[4][:2]) >= 6 and np.any([Path(s).parts[-1][:-4] in g and
                                                     ('png' in g or 'jpeg' in g)
                                                     for g in glob(f'{str(Path(s).parent)}/*')]):
            try:
                xml_data = loadXMLFile(f'{s[:-4]}.xml', True)['SlimSDR_Configuration']
                if (xml_data['SlimSDR_Info']['System_Mode'] == 'SAR' 
                        and 9e9 < xml_data['SlimSDR_Info']['Channel_0']['Center_Frequency_Hz'] < 32e9 
                        and xml_data['SlimSDR_Info']['Gimbal_Settings']['Gimbal_Depression_Angle_D'] > 20.0):
                    sdr_file.append(s)
            except FileNotFoundError:
                print(f'{s} not found.')
            except Exception:
                print(f'{s} has broken XML.')

    franges = np.linspace(config['perf_params']['vehicle_slant_range_min'],
                          config['perf_params']['vehicle_slant_range_max'], 1000) * 2 / c0
    nrange = franges[0]
    pulse_length = (nrange - 1 / TAC) * config['settings']['plp']
    duty_cycle_time_s = pulse_length + franges
    nr = int(pulse_length * fs)

    # Standardize the FFT length for training purposes (this may cause data loss)
    fft_len = config['generate_data_settings']['fft_sz']
    bin_bw = int(config['settings']['bandwidth'] // (fs / fft_len))
    bin_bw += 1 if bin_bw % 2 != 0 else 0

    if config['generate_data_settings']['run_clutter']:
        print('Running clutter data...')

        for fn in sdr_file:
            try:
                sdr_f = load(fn, progress_tracker=True)
            except (ModuleNotFoundError, TypeError):
                print(f'Out of date pickle for {fn}')
                sdr_f = load(fn, import_pickle=False, progress_tracker=True)
            if sdr_f[0].fs != fs:
                continue  # I'll work on this later
            mfilt = sdr_f.genMatchedFilter(0, fft_len=fft_len)
            rollback = -int(np.round(sdr_f[0].baseband_fc / (fs / fft_len)))
            print('Matched filter loaded.')

            print(f'File is {fn}')
            used_pts = []
            for _ in range(config['generate_data_settings']['iterations']):
                check_pts = np.random.choice(list(set(sdr_f[0].frame_num[:-config['settings']['batch_sz']]).difference(
                    used_pts)),
                    config['generate_data_settings']['iterations'])
                used_pts.extend(check_pts)
                inp_data = []
                clutter_abs = []
                for n in tqdm(check_pts):
                    if n + config['settings']['cpi_len'] > sdr_f[0].nframes:
                        break
                    pnums = sdr_f[0].frame_num[n:n + config['settings']['cpi_len']]
                    if np.diff(pnums).max() > 1:
                        continue
                    pulse_data = sdr_f.getPulses(pnums, 0)[1]
                    pmean, cov_dt = getVAECov(pulse_data, mfilt, rollback, sdr_f[0].nsam, fft_len)
                    clutter_abs.append(pmean)
                    inp_data.append(cov_dt)
                if not inp_data:
                    break
                inp_data = np.array(inp_data, dtype=np.float32)
                clutter_abs = np.array(clutter_abs)  #formatTargetClutterData(np.array(clutter_abs), bin_bw).astype(np.float32)
                with open(
                        f'{save_path}/clutter_{fn.split("/")[-1].split(".")[0]}.cov', 'ab') as writer:
                    inp_data.tofile(writer)
                with open(
                        f'{save_path}/clutter_{fn.split("/")[-1].split(".")[0]}.spec', 'ab') as writer:
                    clutter_abs.tofile(writer)

    if config['generate_data_settings']['run_targets']:
        bin_bw = int(config['settings']['bandwidth'] // (fs / config['generate_data_settings']['fft_sz']))
        bin_bw += 1 if bin_bw % 2 != 0 else 0
        print('Running targets...')
        targs = []
        targ_abs = []
        for _ in tqdm(range(128)):
            tpsd = genTargetPSDSwerling1(config['settings']['bandwidth'], config['settings']['fc'],
                                         config['perf_params']['vehicle_slant_range_min'],
                                         config['perf_params']['vehicle_slant_range_max'],
                                         config['generate_data_settings']['fft_sz'], fs, 32,
                                         config['generate_data_settings']['fft_sz'])
            tp_mean = tpsd.mean(axis=0)
            targ_abs.append(np.stack((tp_mean.real, tp_mean.imag), axis=1))
            cov_dt = np.cov(tpsd)
            targs.append(np.stack((cov_dt.real, cov_dt.imag), axis=2))
        targs = np.array(targs).astype(np.float32)
        targ_abs = formatTargetClutterData(np.array(targ_abs), bin_bw).astype(np.float32)
        with open(f'{save_path}/targets.cov', 'ab') as writer:
            targs.tofile(writer)
        with open(f'{save_path}/targets.spec', 'ab') as writer:
            targ_abs.tofile(writer)
