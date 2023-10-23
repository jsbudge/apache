import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from dataloaders import DataModule
from experiment import VAExperiment
from models import BetaVAE, InfoVAE, WAE_MMD
import argparse

'''parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--local_rank", type=int,
                    help="Local rank. Necessary for using the torch.distributed.launch utility.")
argv = parser.parse_args()

local_rank = argv.local_rank'''

strat = DDPStrategy(process_group_backend='nccl')
print(f'Cuda is available? {torch.cuda.is_available()}')

with open('./vae_config.yaml') as y:
    param_dict = yaml.safe_load(y.read())
# hparams = make_dataclass('hparams', param_dict.items())(**param_dict)

data = DataModule(**param_dict["dataset_params"])
data.setup()

# Get the model, experiment, logger set up
if param_dict['exp_params']['model_type'] == 'InfoVAE':
    model = InfoVAE(**param_dict['model_params'])
elif param_dict['exp_params']['model_type'] == 'WAE_MMD':
    model = WAE_MMD(**param_dict['model_params'])
else:
    model = BetaVAE(**param_dict['model_params'])
experiment = VAExperiment(model, param_dict['exp_params'])
logger = loggers.TensorBoardLogger(param_dict['train_params']['log_dir'],
                                   name=f"{param_dict['exp_params']['model_type']}")
trainer = Trainer(logger=logger, max_epochs=param_dict['train_params']['max_epochs'], log_every_n_steps=50,
                  strategy=strat, devices=2, num_nodes=4)

# Generate filepaths for sample and reconstruction images
Path(f"{logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)
# trainer.test(model, train_loader, verbose=True)

print(f"======= Training =======")
trainer.fit(experiment, datamodule=data)

try:
    torch.save(model.state_dict(), './model/inference_model.state')
    print('Model saved to disk.')
except:
    print('Model not saved.')

