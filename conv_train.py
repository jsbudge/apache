import torch
from pytorch_lightning import Trainer, loggers
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from simulib.simulation_functions import db
from dataloaders import CovDataModule
from experiment import AExperiment
from models import ConvAE
print(f'Cuda is available? {torch.cuda.is_available()}')

with open('./vae_config.yaml') as y:
    param_dict = yaml.safe_load(y.read())
# hparams = make_dataclass('hparams', param_dict.items())(**param_dict)

data = CovDataModule(**param_dict["dataset_params"])
data.setup()

model = ConvAE(**param_dict['compression_params'])
experiment = AExperiment(model, param_dict['exp_params'])
logger = loggers.TensorBoardLogger(param_dict['train_params']['log_dir'],
                                   name=f"ConvAE")
trainer = Trainer(logger=logger, max_epochs=param_dict['train_params']['max_epochs'], log_every_n_steps=1)

print(f"======= Training =======")
trainer.fit(experiment, datamodule=data)

model.eval()
sample = data.train_dataset[0][0].data.numpy()
recon = model(data.train_dataset[0][0].unsqueeze(0))[1].squeeze(0).data.numpy()

try:
    torch.save(model.state_dict(), './model/compression_model.state')
    print('Model saved to disk.')
except:
    print('Model not saved.')


plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(db(sample[0, ...] + 1j * sample[1, ...]))
plt.subplot(1, 2, 2)
plt.imshow(db(recon[0, ...] + 1j * recon[1, ...]))
plt.show()

