import contextlib
from glob import glob
import os
import numpy as np
from pathlib import Path
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
    bg_file = '/data6/SAR_DATA/2023/07132023/SAR_07132023_122801.sar'
    upsample = 1
    poly_num = 1
    rotate_grid = True
    use_ecef = True
    ipr_mode = False
    cpi_len = 64
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

    clutter_files = glob(f'{save_path}/clutter_*.spec')

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
    mumax = 0
    stdmax = 0

    # cust_grid = np.zeros_like(bg.refgrid)
    # cust_grid[100, 100] = 10
    # bg._refgrid = cust_grid
    nsam, nr, ranges, ranges_sampled, near_range_s, granges, _, up_fft_len = (
        rp.getRadarParams(fdelay, plp, upsample))
    for clut in clutter_files:
        # Load the sar file that the clutter came from
        cfnme_parts = clut.split('_')[1:4]
        clut_name = clut.split('/')[-1][8:-5]
        sdr_ch = load(f'/data6/SAR_DATA/{cfnme_parts[1][4:]}/{cfnme_parts[1]}/SAR_{cfnme_parts[1]}_{cfnme_parts[2][:-5]}.sar',
                      use_jump_correction=False)

        # Get pulse data and modify accordingly
        pulse = np.fft.fft(sdr_ch[0].cal_chirp, fft_len)
        mfilt = sdr_ch.genMatchedFilter(0, fft_len=fft_len) / 1e4
        valids = mfilt != 0
        if sdr_ch[0].baseband_fc != 0:
            valids = np.roll(valids, -int(sdr_ch[0].baseband_fc / rp.fs * fft_len), axis=0)

        # CHeck to see we're writing to a fresh file
        if Path(f'{save_path}/target_{clut_name}.spec').exists():
            os.remove(f'{save_path}/target_{clut_name}.spec')
            os.remove(f'{save_path}/target_{clut_name}.enc')
        for tpsd in genSimPulseData(rp, bg, fdelay, plp, upsample, grid_width, grid_height,
                                    pts_per_m, cpi_len, a_chirp=pulse,
                                    a_bpj_wavelength=sdr_ch.getBackprojectWavelength(0),
                                    a_pulse_times=sdr[0].pulse_time, a_noise_level=-300,
                                    a_rotate_grid=rotate_grid, a_fft_len=fft_len, a_debug=debug, a_origin=origin):
            tpsd = tpsd * mfilt[:, None]
            if sdr_ch[0].baseband_fc != 0:
                tpsd = np.roll(tpsd, -int(sdr_ch[0].baseband_fc / rp.fs * fft_len), axis=0)
            # tpsd /= 12.5
            tmp_mu = abs(tpsd[valids, :].mean(axis=0)).max()
            tmp_std = abs(tpsd[valids, :].std(axis=0)).max()
            if tmp_mu > mumax:
                print(f'MU: {tmp_mu}')
                mumax = tmp_mu + 0.
            if tmp_std > stdmax:
                print(f'STD:{tmp_std}')
                stdmax = tmp_std + 0.
            p_muscale = np.repeat(config['exp_params']['dataset_params']['mu'] / abs(tpsd[valids, :].mean(axis=0)),
                                  2).astype(
                np.float32)
            p_stdscale = np.repeat(config['exp_params']['dataset_params']['var'] / abs(tpsd[valids, :].std(axis=0)),
                                   2).astype(
                np.float32)
            tpsd[valids, :] = (tpsd[valids, :] - config['exp_params']['dataset_params']['mu']) / \
                              config['exp_params']['dataset_params']['var']
            inp_data = formatTargetClutterData(tpsd.T, fft_len).astype(np.float32)
            if p_muscale.min() < 2.:
                enc_data = decoder.encode(torch.tensor(inp_data).to(decoder.device)).cpu().data.numpy()
                inp_data = np.concatenate((inp_data, p_muscale.reshape(-1, 2, 1), p_stdscale.reshape(-1, 2, 1)), axis=2)
                with open(
                        f'{save_path}/target_{clut_name}.spec', 'ab') as writer:
                    inp_data.tofile(writer)
                with open(
                        f'{save_path}/target_{clut_name}.enc', 'ab') as writer:
                    enc_data.tofile(writer)
