import time
import timeit

import torch
import numpy as np
import matplotlib.pyplot as plt
from dataloaders import CovarianceDataset, PulseDataset
from models import InfoVAE, BetaVAE, WAE_MMD
from glob import glob
from torchvision import transforms
from simulib.simulation_functions import db, GetAdvMatchedFilter
import yaml
from scipy.spatial.distance import pdist, squareform
from torch.utils.data import DataLoader, Dataset
from data_converter.SDRParsing import loadASIFile
from celluloid import Camera
import cv2
from data_converter.SDRParsing import SDRParse, load
from tqdm import tqdm
from jax.numpy import fft as jaxfft

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open('./vae_config.yaml') as y:
    param_dict = yaml.safe_load(y.read())

batch_sz = param_dict['settings']['batch_sz']

# Get the model, experiment, logger set up
if param_dict['exp_params']['model_type'] == 'InfoVAE':
    model = InfoVAE(**param_dict['model_params'])
elif param_dict['exp_params']['model_type'] == 'WAE_MMD':
    model = WAE_MMD(**param_dict['model_params'])
else:
    model = BetaVAE(**param_dict['model_params'])
model.load_state_dict(torch.load('./model/inference_model.state'))
model.eval()  # Set to inference mode
model.to(device)

sdr_file_bgtype = [('SAR_06072023_154802', 'river'),
                   ('SAR_06062023_125944', 'suburb'),
                   ('SAR_08092023_143927', 'farmland'),
                   ('SAR_08092023_112016', 'airport'),
                   ('SAR_08092023_144437', 'highschool'),
                   ('SAR_08232023_114640', 'farmcornmaze'),
                   ('SAR_08232023_144235', 'sportsparksuburb'),
                   ('SAR_08102023_110807', 'river'),
                   ('SAR_09132023_114021', 'airportfield'),
                   ('SAR_09122023_115704', 'ruralstreet'),
                   ('SAR_09122023_151902', 'sportspark'),
                   ('SAR_09122023_152050', 'orchard'),
                   ('SAR_09122023_152903', 'lakeleft'),
                   ('SAR_09122023_153015', 'lakeorchard')]

clutter_files = glob(f'./data/clutter_*.cov')
spec_files = glob(f'./data/clutter_*.spec')
image_files = glob('/data6/SAR_DATA/2023/**/*.jpeg')
latent_reps = []
images = []

train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.034028642, 0.04619637), 0.4151423),
            ]
        )

fft_len = 32768
sdr_file = ['/data6/SAR_DATA/2023/06072023/SAR_06072023_154802.sar',
            '/data6/SAR_DATA/2023/06062023/SAR_06062023_125944.sar',
            '/data6/SAR_DATA/2023/08092023/SAR_08092023_143927.sar',
            '/data6/SAR_DATA/2023/08092023/SAR_08092023_112016.sar',
            '/data6/SAR_DATA/2023/08092023/SAR_08092023_144437.sar',
            '/data6/SAR_DATA/2023/08232023/SAR_08232023_114640.sar',
            '/data6/SAR_DATA/2023/08232023/SAR_08232023_144235.sar',
            '/data6/SAR_DATA/2023/08232023/SAR_08232023_091003.sar',
            '/data6/SAR_DATA/2023/08232023/SAR_08232023_090943.sar',
            '/data6/SAR_DATA/2023/08102023/SAR_08102023_110807.sar',
            '/data6/SAR_DATA/2023/09132023/SAR_09132023_114021.sar',
            '/data6/SAR_DATA/2023/09122023/SAR_09122023_115704.sar',
            '/data6/SAR_DATA/2023/09122023/SAR_09122023_151902.sar',
            '/data6/SAR_DATA/2023/09122023/SAR_09122023_152050.sar',
            '/data6/SAR_DATA/2023/09122023/SAR_09122023_152903.sar',
            '/data6/SAR_DATA/2023/09122023/SAR_09122023_153015.sar']
for s in sdr_file_bgtype:
    sdr_f = load([c for c in sdr_file if s[0] in c][0])
    mfilt = GetAdvMatchedFilter(sdr_f[0], fft_len=fft_len)
    dataset = CovarianceDataset([c for c in clutter_files if s[0] in c][0], transform=train_transforms)
    ims = []
    latent_z = []
    samples = []
    for i, batch in tqdm(enumerate(DataLoader(dataset, shuffle=False, batch_size=batch_sz)), total=len(dataset) // batch_sz):
        latent_z.append(model.forward(batch[0].to(device))[2].cpu().data.numpy())
        samples.append(batch[0].data.numpy())
        pulse_fft = jaxfft.fft(sdr_f.getPulses(sdr_f[0].frame_num[i:i + param_dict['settings']['cpi_len'] * 2], 0)[1],
                               fft_len, axis=0) * mfilt[:, None]
        ims.append(db(np.fft.fftshift(jaxfft.fft(jaxfft.ifft(pulse_fft, axis=0)[:sdr_f[0].nsam, :], axis=1), axes=1)))
    ims = np.stack(ims)
    latent_z = np.concatenate(latent_z)
    samples = np.concatenate(samples)

    latent_reps.append(latent_z)
    images.append(ims)

    plt.figure(f'{s[0]} Information')
    plt.subplot(1, 3, 1)
    plt.imshow(db(samples[0, 0, ...] + 1j * samples[0, 1, ...]))
    plt.subplot(1, 3, 2)
    plt.scatter(np.arange(latent_z.shape[1]), latent_z[0, ...], c='blue')
    plt.subplot(1, 3, 3)
    plt.imshow(ims[0, ...])
    plt.axis('tight')

    # images.append(cv2.imread([c for c in image_files if s[0] in c][0]))
    '''sfig, ax = plt.subplots(1, 3)
    cam = Camera(sfig)
    for d in tqdm(range(0, len(dataset), 64)):
        ax[0].imshow(db(samples[d, 0, ...] + 1j * samples[d, 1, ...]))
        ax[1].scatter(np.arange(latent_z.shape[1]), latent_z[d, ...], c='blue')
        ax[1].set_title(f'Set {d}')
        ax[2].imshow(np.fft.fftshift(db(ims[d // 64, ...]), axes=1))
        plt.axis('tight')
        cam.snap()
    anim = cam.animate(interval=120)
    plt.show()'''

fig, axes = plt.subplots(3, 5)
cam = Camera(fig)
dist_lim = 0
for didx, d in tqdm(enumerate(range(0, min([f.shape[0] for f in latent_reps]), batch_sz))):
    lat_rep = np.array([f[d, ...] for f in latent_reps])
    dist_mat = squareform(pdist(lat_rep))
    if dist_lim == 0:
        dist_lim = dist_mat.max()
    axes[1, 2].imshow(dist_mat, clim=[0, dist_lim])
    axes[1, 2].set_xticks(np.arange(dist_mat.shape[0]), [s[1] for s in sdr_file_bgtype], rotation=45)
    axes[1, 2].set_yticks(np.arange(dist_mat.shape[0]), [s[1] for s in sdr_file_bgtype], rotation=45)
    idxes = np.arange(len(images))
    idxes[idxes > 6] += 1
    for idx, im in enumerate(images):
        axes[idxes[idx] // 5, idxes[idx] % 5].imshow(im[didx, ...])
        axes[idxes[idx] // 5, idxes[idx] % 5].axis('tight')
        axes[idxes[idx] // 5, idxes[idx] % 5].set_title(f'{sdr_file_bgtype[idx][1]}')
    cam.snap()

print('Saving visualization...')
anim = cam.animate()
anim.save('./visualization.mp4', fps=10)
plt.show()

'''ax = plt.figure().add_subplot(projection='3d')
for l in latent_reps:
    ax.scatter(l[:, 0], l[:, 1], l[:, 2])
plt.show()'''

