import contextlib
import copy
from glob import glob
import os
import numpy as np
from pathlib import Path
from simulib.simulation_functions import llh2enu, db, azelToVec
from simulib.cuda_mesh_kernels import readCombineMeshFile, genRangeProfileFromMesh
from simulib.platform_helper import SDRPlatform
from simulib.cuda_kernels import cpudiff, getMaxThreads
from generate_trainingdata import formatTargetClutterData
from models import Encoder
import torch
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from tqdm import tqdm
import yaml
from data_converter.SDRParsing import load
import cupy

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254


def getSpectrumFromTargetProfile(sdr, rp, ts, tcdata, fft_len):
    pulse = np.fft.fft(sdr[0].cal_chirp, fft_len)
    mfilt = sdr.genMatchedFilter(0, fft_len=fft_len) / 1e4
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
    tpsd = np.array([tcdata[:, np.logical_and(abs(cpudiff(pan, sdr_pan[n])) < 0.19634954,
                                              abs(cpudiff(tilt, sdr_tilt[n])) < 0.19634954)].flatten() for n in
                     range(len(ts))])
    tpsd = np.fft.fft(tpsd, fft_len, axis=1) * 1e12 * pulse * mfilt
    if sdr[0].baseband_fc != 0:
        tpsd = np.roll(tpsd, -int(sdr[0].baseband_fc / rp.fs * fft_len), axis=1)
    return tpsd


if __name__ == '__main__':

    with open('./vae_config.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    standoff = 500.

    save_path = config['generate_data_settings']['local_path'] if (
        config)['generate_data_settings']['use_local_storage'] else config['dataset_params']['data_path']

    target_obj_files = {'air_balloon.obj': 30., 'black_bird_sr71_obj.obj': .25, 'cessna-172-obj.obj': 41.08, 'helic.obj': 1.8, 
                        'Humvee.obj': 60, 'Intergalactic_Spaceships_Version_2.obj': 1., 'piper_pa18.obj': 1., 
                        'Porsche_911_GT2.obj': .8, 'Seahawk.obj': 12., 't-90a(Elements_of_war).obj': 1.,
                        'Tiger.obj': 156.25, 'G7_1200.obj': 1., 'x-wing.obj': 1., 'tacoma_VTC.dat': -1.,
                        }
    target_obj_files = list(target_obj_files.items())

    clutter_files = glob(f'{save_path}/clutter_*.spec')

    # Standardize the FFT length for training purposes (this may cause data loss)
    fft_len = config['settings']['fft_len']

    # Generate angles for block
    pans, tilts = np.meshgrid(np.linspace(0, 2 * np.pi, 16, endpoint=False),
                              np.linspace(np.pi / 2 - .1, -np.pi / 2 + .1, 16))
    pan = pans.flatten()
    tilt = tilts.flatten()
    poses = azelToVec(pan, tilt).T * standoff
    pan[pan == 0] = 1e-9
    pan[pan == 2 * np.pi] = 1e-9
    tilt[tilt == 0] = 1e-9
    tilt[tilt == 2 * np.pi] = 1e-9

    poses_gpu = cupy.array(poses, dtype=np.float32)
    pan_gpu = cupy.array(pan, dtype=np.float32)
    tilt_gpu = cupy.array(tilt, dtype=np.float32)
    plt.ion()

    for tidx, (tobj, scaling) in tqdm(enumerate(target_obj_files)):
        if scaling > 0:
            try:
                mesh = readCombineMeshFile(f'{config["generate_data_settings"]["obj_path"]}/{tobj}')
            except IndexError:
                print(f'{tobj} not found.')
                continue
            mesh.scale(1 / scaling, center=(0, 0, 0))
            pt_sample = []
            # Apply random rotations and scalings for augmenting of training data
            for i in range(300):
                if i != 0:
                    mesh.rotate(mesh.get_rotation_matrix_from_xyz((np.random.rand() * 2 * np.pi, np.random.rand() * 2 * np.pi,
                                                                     np.random.rand() * 2 * np.pi)), center=(0, 0, 0))
                # sample_points = nmesh.sample_points_poisson_disk(30000)
                face_centers = np.asarray(mesh.vertices)
                face_normals = np.asarray(mesh.vertex_normals)

                pd_r = cupy.zeros((256, len(pan)), dtype=np.float32)
                pd_i = cupy.zeros((256, len(pan)), dtype=np.float32)
                near_range_s = (standoff - 10) / c0

                face_centers_gpu = cupy.array(face_centers, dtype=np.float32)
                face_normals_gpu = cupy.array(face_normals, dtype=np.float32)
                reflectivity_gpu = cupy.array(np.ones(face_centers.shape[0]), dtype=np.float32)

                threads_per_block = getMaxThreads()
                bpg_bpj = (max(1, face_centers.shape[0] // threads_per_block[0] + 1), len(pan) // threads_per_block[1] + 1)
                genRangeProfileFromMesh[bpg_bpj, threads_per_block](face_centers_gpu, face_normals_gpu, reflectivity_gpu,
                                                                    poses_gpu,
                                                                    poses_gpu,
                                                                    pan_gpu, tilt_gpu, pan_gpu, tilt_gpu, pd_r, pd_i,
                                                                    c0 / 9.6e9,
                                                                    near_range_s, 2e9, 10 * DTR,
                                                                    10 * DTR, 1., False, False)

                pd = pd_r.get() + 1j * pd_i.get()
                if i == 0:
                    pt_sample = np.concatenate((pt_sample, pd[pd != 0]))
                pd[pd != 0] = ((pd[pd != 0] - config['target_exp_params']['dataset_params']['mu']) /
                      config['target_exp_params']['dataset_params']['var'])
                plt.gca().cla()
                plt.imshow(db(pd))
                plt.draw()
                plt.pause(.1)
                pd_cat = np.stack([pd.real, pd.imag]).astype(np.float32)

                if i == 0:
                    with open(f'{save_path}/targetpatterns.dat', 'ab') as w:
                        pd_cat.tofile(w)
                with open(f'{save_path}/targetprofiles.dat', 'ab') as w:
                    pd_cat.tofile(w)
        else:
            pass
            '''with open(f'{config["generate_data_settings"]["obj_path"]}/{tobj}', 'r') as f:
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

                poses_gpu = cupy.array(poses, dtype=np.float32)
                pan_gpu = cupy.array(pan, dtype=np.float32)
                tilt_gpu = cupy.array(pan, dtype=np.float32)'''

        if config['settings']['save_as_target']:
            for clut in clutter_files:
                # Load the sar file that the clutter came from
                cfnme_parts = clut.split('_')[1:4]
                clut_name = clut.split('/')[-1][8:-5]
                sdr_ch = load(f'/data6/SAR_DATA/{cfnme_parts[1][4:]}/{cfnme_parts[1]}/SAR_{cfnme_parts[1]}_{cfnme_parts[2][:-5]}.sar',
                              use_jump_correction=False)
                rp = SDRPlatform(sdr_ch)

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
                for n in range(config['generate_data_settings']['iterations']):
                    if n + config['settings']['cpi_len'] > sdr_ch[0].nframes:
                        break
                    ts = sdr_ch[0].pulse_time[n:n + config['settings']['cpi_len']]
                    # Get the appropriate pan and tilt values from the profile
                    tpsd = getSpectrumFromTargetProfile(sdr_ch, rp, ts, pd, fft_len)
                    tmp_mu = abs(tpsd[:, valids].mean(axis=0)).max()
                    tmp_std = abs(tpsd[:, valids].std(axis=0)).max()
                    p_muscale = np.repeat(config['exp_params']['dataset_params']['mu'] / abs(tpsd[:, valids].mean(axis=1)),
                                          2).astype(
                        np.float32)
                    p_stdscale = np.repeat(config['exp_params']['dataset_params']['var'] / abs(tpsd[:, valids].std(axis=1)),
                                           2).astype(
                        np.float32)
                    tpsd[:, valids] = (tpsd[:, valids] - config['exp_params']['dataset_params']['mu']) / \
                                      config['exp_params']['dataset_params']['var']
                    inp_data = formatTargetClutterData(tpsd, fft_len).astype(np.float32)
                    inp_data = np.concatenate((inp_data, p_muscale.reshape(-1, 2, 1), p_stdscale.reshape(-1, 2, 1)), axis=2)
                    with open(
                            f'{save_path}/target_{tidx}_{clut_name}.spec', 'ab') as writer:
                        inp_data.tofile(writer)
