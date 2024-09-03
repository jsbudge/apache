import torch
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from torchvision import transforms
from simulib.simulation_functions import db
import yaml
from celluloid import Camera
from data_converter.SDRParsing import load
from tqdm import tqdm
from sklearn.decomposition import KernelPCA
import matplotlib.animation as animation

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open('./vae_config.yaml') as y:
    param_dict = yaml.safe_load(y.read())

batch_sz = param_dict['settings']['batch_sz']
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
                   ('SAR_06072023_111506', 'airportfield'),
                   ('SAR_09122023_115704', 'ruralstreet'),
                   ('SAR_09122023_151902', 'sportspark'),
                   ('SAR_09122023_152050', 'orchard'),
                   ('SAR_09122023_152903', 'lakeleft'),
                   ('SAR_09122023_153015', 'lakeorchard')]

clutter_files = glob('./data/clutter_*.cov')
spec_files = glob('./data/clutter_*.spec')
image_files = glob('/data6/SAR_DATA/2023/**/*.jpeg')
latent_reps = []
images = []

train_transforms = getTrainTransforms(param_dict['dataset_params']['var'])

iters = 10

fft_len = param_dict['generate_data_settings']['fft_sz']
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
            '/data6/SAR_DATA/2023/06072023/SAR_06072023_111506.sar',
            '/data6/SAR_DATA/2023/09122023/SAR_09122023_115704.sar',
            '/data6/SAR_DATA/2023/09122023/SAR_09122023_151902.sar',
            '/data6/SAR_DATA/2023/09122023/SAR_09122023_152050.sar',
            '/data6/SAR_DATA/2023/09122023/SAR_09122023_152903.sar',
            '/data6/SAR_DATA/2023/09122023/SAR_09122023_153015.sar']
for s in sdr_file_bgtype:
    try:
        sdr_f = load([c for c in sdr_file if s[0] in c][0])
    except IndexError:
        print(f'{s[0]} not found.')
        continue
    except ModuleNotFoundError:
        sdr_f = load([c for c in sdr_file if s[0] in c][0], import_pickle=False, progress_tracker=True)
    mfilt = sdr_f.genMatchedFilter(0, fft_len=fft_len)
    rollback = -int(np.round(sdr_f[0].baseband_fc / (sdr_f[0].fs / fft_len)))
    ims = []
    latent_z = []
    samples = []
    for i, batch_idx in tqdm(enumerate(range(0, len(sdr_f[0].frame_num) - param_dict['settings']['cpi_len'],
                                             len(sdr_f[0].frame_num) // iters))):
        pulse_data = sdr_f.getPulses(sdr_f[0].frame_num[batch_idx:batch_idx + param_dict['settings']['cpi_len']], 0)[1]
        _, cov_dt = getVAECov(pulse_data, mfilt, rollback, sdr_f[0].nsam, fft_len)
        dt = train_transforms(cov_dt.astype(np.float32)).unsqueeze(0)
        latent_z.append(model.forward(dt.to(device))[2].cpu().data.numpy())
        samples.append(dt.data.numpy())
        pulse_fft = jaxfft.fft(pulse_data, fft_len, axis=0) * mfilt[:, None]
        ims.append(db(np.fft.fftshift(jaxfft.fft(jaxfft.ifft(pulse_fft, axis=0)[:sdr_f[0].nsam, :], axis=1), axes=1)))
    ims = np.stack(ims)
    latent_z = np.concatenate(latent_z)
    samples = np.concatenate(samples)

    latent_reps.append(latent_z)
    images.append(ims)

    dec = model.decode(torch.tensor(latent_z[0, :]).to(device)).cpu().data.numpy()
    dec = (dec[0, 0, ...] + 1j * dec[0, 1, ...]) * param_dict['dataset_params']['var']

    plt.figure(f'{s[0]} Information')
    plt.subplot(2, 3, 1)
    plt.title('True Covariance')
    plt.imshow(db(samples[0, 0, ...] + 1j * samples[0, 1, ...]))
    plt.axis('off')
    plt.subplot(2, 3, 4)
    plt.title('Generated Covariance')
    plt.imshow(db(dec))
    plt.axis('off')
    plt.subplot(2, 3, (2, 5))
    plt.scatter(np.arange(latent_z.shape[1]), latent_z[0, ...], c='blue')
    plt.subplot(2, 3, (3, 6))
    clim_mu = np.mean(ims[0, ...])
    clim_std = np.std(ims[0, ...])
    plt.imshow(ims[0, ...], cmap='gray', clim=[clim_mu - clim_std * 3, clim_mu + clim_std + 3])
    plt.axis('off')
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

'''fig, axes = plt.subplots(3, 5)
cam = Camera(fig)
dist_lim = 0
for didx, d in tqdm(enumerate(range(0, min([f.shape[0] for f in latent_reps])))):
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
anim = cam.animate(interval=1000)
# anim.save('./visualization.mp4', fps=10)
plt.show()'''

'''ax = plt.figure().add_subplot(projection='3d')
for l in latent_reps:
    ax.scatter(l[:, 0], l[:, 1], l[:, 2])
plt.show()'''

lr = np.vstack(latent_reps)
kpca = KernelPCA(3)

lr_kpca = kpca.fit(lr)

lr_river = kpca.transform(latent_reps[7])
lr_sppark = kpca.transform(latent_reps[10])
fig = plt.figure()
ax0 = fig.add_subplot(1, 3, 1)
ax1 = fig.add_subplot(1, 3, 2, projection='3d')
axtitle = ax0.text(x=0.5, y=0.85, s="", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                   transform=ax0.transAxes, ha="center")
ax2 = fig.add_subplot(1, 3, 3)
cam = Camera(fig)
for idx in range(iters):
    ax0.imshow(images[7][idx, ...], cmap='gray',
               clim=[images[7].mean() - images[7].std() * 3, images[7].mean() + images[7].std() * 3])
    ax0.axis('tight')
    ax0.axis('off')
    ax1.scatter(lr_river[idx, 0], lr_river[idx, 1], lr_river[idx, 2], c='blue')
    ax1.scatter(lr_sppark[idx, 0], lr_sppark[idx, 1], lr_sppark[idx, 2], c='red')
    ax0.title = ax0.text(x=0.5, y=0.85, s=f'CPI {idx}',
                         transform=ax0.transAxes, ha="center")
    ax2.imshow(images[10][idx, ...], cmap='gray',
               clim=[images[10].mean() - images[10].std() * 3, images[10].mean() + images[10].std() * 3])
    ax2.axis('tight')
    ax2.axis('off')
    cam.snap()
anim = cam.animate(interval=100)
pilwriter = animation.PillowWriter(fps=10)
anim.save('./visualization.gif', writer=pilwriter)
plt.show()
