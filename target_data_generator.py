import contextlib
import numpy as np

from simulib.simulation_functions import llh2enu, db
from simulib import genSimPulseData, getRadarAndEnvironment
from generate_trainingdata import formatTargetClutterData
from models import Encoder
import torch
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from tqdm import tqdm
import yaml
from data_converter.SDRParsing import load

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254

if __name__ == '__main__':

    with open('./vae_config.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    # This is the file used to backproject data
    # bg_file = '/data5/SAR_DATA/2021/05052021/SAR_05052021_112647.sar'
    # bg_file = '/data5/SAR_DATA/2022/09082022/SAR_09082022_131237.sar'
    # bg_file = '/data5/SAR_DATA/2022/Redstone/SAR_08122022_170753.sar'
    # bg_file = '/data6/SAR_DATA/2023/06202023/SAR_06202023_135617.sar'
    # bg_file = '/data6/Tower_Redo_Again/tower_redo_SAR_03292023_120731.sar'
    # bg_file = '/data5/SAR_DATA/2022/09272022/SAR_09272022_103053.sar'
    # bg_file = '/data5/SAR_DATA/2019/08072019/SAR_08072019_100120.sar'
    bg_file = '/data6/SAR_DATA/2023/07132023/SAR_07132023_122801.sar'
    # bg_file = '/data6/SAR_DATA/2023/07132023/SAR_07132023_122801.sar'
    # bg_file = '/data6/SAR_DATA/2023/07132023/SAR_07132023_123050.sar'
    upsample = 1
    poly_num = 1
    rotate_grid = True
    use_ecef = True
    ipr_mode = False
    cpi_len = 256
    plp = 0
    partial_pulse_percent = .2
    debug = True
    pts_per_m = 20
    grid_width = 20
    grid_height = 20
    channel = 0
    fdelay = 5.8
    origin = (40.138018, -111.660087, 1382)

    save_path = config['generate_data_settings']['local_path'] if (
        config)['generate_data_settings']['use_local_storage'] else config['dataset_params']['data_path']

    # Standardize the FFT length for training purposes (this may cause data loss)
    fft_len = config['settings']['fft_len']

    print('Setting up decoder...')
    decoder = Encoder(**config['model_params'], fft_len=config['settings']['fft_len'], params=config['exp_params'])
    with contextlib.suppress(RuntimeError):
        decoder.load_state_dict(torch.load('./model/inference_model.state'))
    decoder.requires_grad = False

    print('Loading SDR file...')
    sdr = load(bg_file, progress_tracker=True, use_jump_correction=False)
    print('Generating platform...', end='')
    bg, rp = getRadarAndEnvironment(sdr, channel)
    print('Done.')

    min_poss = -config['exp_params']['dataset_params']['mu'] / config['exp_params']['dataset_params']['var']

    # cust_grid = np.zeros_like(bg.refgrid)
    # cust_grid[100, 100] = 10
    # bg._refgrid = cust_grid
    nsam, nr, ranges, ranges_sampled, near_range_s, granges, _, up_fft_len = (
        rp.getRadarParams(fdelay, plp, upsample))
    pulse = np.fft.fft(sdr[0].cal_chirp, fft_len)
    mfilt = sdr.genMatchedFilter(0, fft_len=fft_len)
    for tpsd in genSimPulseData(rp, bg, fdelay, plp, upsample, grid_width, grid_height,
                                pts_per_m, cpi_len, a_chirp=pulse, a_sdr=sdr, a_noise_level=-300,
                                a_rotate_grid=rotate_grid, a_fft_len=fft_len, a_debug=debug, a_origin=origin):
        tpsd = tpsd * mfilt[:, None]
        # tpsd *= 4000.
        p_muscale = np.repeat(config['exp_params']['dataset_params']['mu'] / abs(tpsd.mean(axis=0)), 2).astype(
            np.float32)
        p_stdscale = np.repeat(config['exp_params']['dataset_params']['var'] / abs(tpsd.std(axis=0)), 2).astype(
            np.float32)
        tpsd = (tpsd - config['exp_params']['dataset_params']['mu']) / config['exp_params']['dataset_params']['var']
        inp_data = formatTargetClutterData(tpsd.T, fft_len).astype(np.float32)
        if p_muscale.mean() < 2.:
            enc_data = decoder.encode(torch.tensor(inp_data).to(decoder.device)).cpu().data.numpy()
            inp_data = np.concatenate((inp_data, p_muscale.reshape(-1, 2, 1), p_stdscale.reshape(-1, 2, 1)), axis=2)

            with open(
                    f'{save_path}/targets.spec', 'ab') as writer:
                inp_data.tofile(writer)
            with open(
                    f'{save_path}/targets.enc', 'ab') as writer:
                enc_data.tofile(writer)
        else:
            print(f'{inp_data.mean()}')
