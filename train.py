import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
import yaml
from pathlib import Path
from dataloaders import CovDataModule
from experiment import VAExperiment
from models import BetaVAE, InfoVAE, WAE_MMD, init_weights
import matplotlib.pyplot as plt
import numpy as np

torch.set_float32_matmul_precision('medium')
print(f'Cuda is available? {torch.cuda.is_available()}')
seed_everything(70, workers=True)

with open('./vae_config.yaml') as y:
    param_dict = yaml.safe_load(y.read())
# hparams = make_dataclass('hparams', param_dict.items())(**param_dict)

data = CovDataModule(**param_dict["dataset_params"])
data.setup()

# Get the model, experiment, logger set up
if param_dict['exp_params']['model_type'] == 'InfoVAE':
    model = InfoVAE(**param_dict['model_params'])
elif param_dict['exp_params']['model_type'] == 'WAE_MMD':
    model = WAE_MMD(**param_dict['model_params'])
else:
    model = BetaVAE(**param_dict['model_params'])
print('Setting up model...')
if param_dict['settings']['warm_start']:
    print('Model loaded from save state.')
    try:
        model.load_state_dict(torch.load('./model/inference_model.state'))
    except RuntimeError:
        print('Model save file does not match current structure. Re-running with new structure.')
        model.apply(init_weights)
else:
    print('Initializing new model...')
    model.apply(init_weights)

# ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
experiment = VAExperiment(model, param_dict['exp_params'])
logger = loggers.TensorBoardLogger(param_dict['train_params']['log_dir'],
                                   name=f"{param_dict['exp_params']['model_type']}")
trainer = Trainer(logger=logger, max_epochs=param_dict['train_params']['max_epochs'],
                  log_every_n_steps=param_dict['exp_params']['log_epoch'],
                  strategy='ddp', deterministic=True, callbacks=
                  [EarlyStopping(monitor='loss', patience=param_dict['exp_params']['patience'],
                                 check_finite=True)])
# trainer.test(model, train_loader, verbose=True)

print("======= Training =======")
trainer.fit(experiment, datamodule=data)

model.eval()
sample = data.val_dataset[0][0].data.numpy()
recon = model(data.val_dataset[0][0].unsqueeze(0))[0].squeeze(0).data.numpy()

if trainer.is_global_zero:
    if param_dict['exp_params']['save_model']:
        try:
            torch.save(model.state_dict(), './model/inference_model.state')
            print('Model saved to disk.')
        except Exception as e:
            print(f'Model not saved: {e}')
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title('Sample Real')
    plt.imshow(sample[0, ...])
    plt.subplot(2, 2, 2)
    plt.title('Sample Imag')
    plt.imshow(sample[1, ...])
    plt.subplot(2, 2, 3)
    plt.title('Recon Real')
    plt.imshow(recon[0, ...])
    plt.subplot(2, 2, 4)
    plt.title('Recon Imag')
    plt.imshow(recon[1, ...])
    plt.show()

