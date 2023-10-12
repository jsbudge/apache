import torch
import torch.nn as nn
from pytorch_lightning import Trainer, loggers
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from simulib.simulation_functions import genPulse, findPowerOf2, db, GetAdvMatchedFilter
from dataloaders import CovarianceDataset, DataModule
from experiment import VAExperiment
from models import BetaVAE

print(f'Cuda is avaliable? {torch.cuda.is_available()}')

with open('./vae_config.yaml') as y:
    param_dict = yaml.safe_load(y.read())
# hparams = make_dataclass('hparams', param_dict.items())(**param_dict)

data = DataModule(**param_dict["dataset_params"])
data.setup()

model = BetaVAE(**param_dict['model_params'])
experiment = VAExperiment(model, param_dict['exp_params'])
logger = loggers.TensorBoardLogger(param_dict['train_params']['log_dir'], name=f"BetaVAE")
trainer = Trainer(logger=logger, max_epochs=param_dict['train_params']['max_epochs'], log_every_n_steps=1)

Path(f"{logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)
# trainer.test(model, train_loader, verbose=True)

print(f"======= Training =======")
trainer.fit(experiment, datamodule=data)

sample = data.val_dataset[0][0].data.numpy()
recon = model(data.val_dataset[0][0].unsqueeze(0))[0].squeeze(0).data.numpy()

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(db(sample[0, ...] + 1j * sample[1, ...]))
plt.subplot(1, 2, 2)
plt.imshow(db(recon[0, ...] + 1j * recon[1, ...]))
plt.show()