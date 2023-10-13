import torch
import numpy as np
import matplotlib.pyplot as plt
from dataloaders import CovarianceDataset
from models import WAE_MMD
from glob import glob
from torchvision import transforms
from simulib.simulation_functions import db
import yaml
from scipy.spatial.distance import pdist, squareform


with open('./vae_config.yaml') as y:
    param_dict = yaml.safe_load(y.read())

model = WAE_MMD(**param_dict['model_params'])
model.load_state_dict(torch.load('./model/inference_model.state'))
model.eval()  # Set to inference mode

sdr_file_bgtype = [('SAR_06072023_154802', 'river'),
                   ('SAR_06062023_125944', 'suburb'),
                   ('SAR_08092023_143927', 'farmland'),
                   ('SAR_08092023_112016', 'airport'),
                   ('SAR_08092023_144437', 'highschool'),
                   ('SAR_08232023_114640', 'stuff'),
                   ('SAR_08232023_144235', 'stuff'),
                   ('SAR_08102023_110807', 'river'),
                   ('SAR_09132023_114021', 'airportfield'),
                   ('SAR_09122023_115704', 'ruralstreet'),
                   ('SAR_09122023_151902', 'sportspark'),
                   ('SAR_09122023_152050', 'orchard'),
                   ('SAR_09122023_152903', 'lakeleft'),
                   ('SAR_09122023_153015', 'lakeorchard')]

clutter_files = glob(f'./data/clutter_*.cov')
latent_reps = []

train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.034028642, 0.04619637), 0.4151423),
            ]
        )

for s in sdr_file_bgtype:
    dataset = CovarianceDataset([c for c in clutter_files if s[0] in c][0], transform=train_transforms)
    data_vec = dataset[0][0].unsqueeze(0)
    recons = model.generate(data_vec).squeeze(0).data.numpy()
    latent_z = model.encode(data_vec)[0].squeeze(0).data.numpy()
    sample = data_vec.squeeze(0).data.numpy()

    latent_reps.append(latent_z.flatten())

    plt.figure(s[1])
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(db(sample[0, ...] + 1j * sample[1, ...]))
    plt.subplot(1, 2, 2)
    plt.title('Reconstruction')
    plt.imshow(db(recons[0, ...] + 1j * recons[1, ...]))

latent_reps = np.array(latent_reps)
dist_mat = squareform(pdist(latent_reps))

plt.figure()
plt.imshow(dist_mat)
plt.xticks(np.arange(dist_mat.shape[0]), [s[1] for s in sdr_file_bgtype], rotation=45)
plt.yticks(np.arange(dist_mat.shape[0]), [s[1] for s in sdr_file_bgtype], rotation=45)




