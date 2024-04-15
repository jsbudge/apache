from glob import glob

import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
import yaml
from pathlib import Path
from dataloaders import EncoderModule
from experiment import VAExperiment
from models import BetaVAE, InfoVAE, WAE_MMD, init_weights
import matplotlib.pyplot as plt
import numpy as np

torch.set_float32_matmul_precision('medium')
print(f'Cuda is available? {torch.cuda.is_available()}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed_everything(np.random.randint(1, 2048), workers=True)

with open('./vae_config.yaml') as y:
    param_dict = yaml.safe_load(y.read())

exp_params = param_dict['exp_params']
# hparams = make_dataclass('hparams', param_dict.items())(**param_dict)

data = EncoderModule(fft_len=param_dict['settings']['fft_len'], **param_dict["dataset_params"])
data.setup()

# Get the model, experiment, logger set up
if exp_params['model_type'] == 'InfoVAE':
    model = InfoVAE(**param_dict['model_params'])
elif exp_params['model_type'] == 'WAE_MMD':
    model = WAE_MMD(fft_len=param_dict['settings']['fft_len'], **param_dict['model_params'])
else:
    model = BetaVAE(**param_dict['model_params'])
print('Setting up model...')
if exp_params['warm_start']:
    print('Model loaded from save state.')
    try:
        model.load_state_dict(torch.load('./model/inference_model.state'))
    except RuntimeError:
        print('Model save file does not match current structure. Re-running with new structure.')
        model.apply(init_weights)
else:
    print('Initializing new model...')
    model.apply(init_weights)

experiment = VAExperiment(model, exp_params)
logger = loggers.TensorBoardLogger(param_dict['train_params']['log_dir'],
                                   name=f"{exp_params['model_type']}")
expected_lr = max((exp_params['LR'] *
                   exp_params['scheduler_gamma'] ** (exp_params['max_epochs'] *
                                                     exp_params['swa_start'])), 1e-9)
trainer = Trainer(logger=logger, max_epochs=exp_params['max_epochs'],
                  log_every_n_steps=exp_params['log_epoch'],
                  strategy='ddp', deterministic=True, devices=2, callbacks=
                  [EarlyStopping(monitor='loss', patience=exp_params['patience'],
                                 check_finite=True),
                   StochasticWeightAveraging(swa_lrs=expected_lr, swa_epoch_start=exp_params['swa_start'])])
# trainer.test(model, train_loader, verbose=True)

print("======= Training =======")
trainer.fit(experiment, datamodule=data)

model.eval()
sample = data.val_dataset[0][0].data.numpy()
recon = model(data.val_dataset[0][0].unsqueeze(0))[0].squeeze(0).data.numpy()

if trainer.is_global_zero:
    if exp_params['save_model']:
        try:
            torch.save(model.state_dict(), './model/inference_model.state')
            print('Model saved to disk.')
        except Exception as e:
            print(f'Model not saved: {e}')
    print('Plotting outputs...')
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title('Sample Real')
    plt.plot(sample[0, ...])
    plt.subplot(2, 2, 2)
    plt.title('Sample Imag')
    plt.plot(sample[1, ...])
    plt.subplot(2, 2, 3)
    plt.title('Recon Real')
    plt.plot(recon[0, ...])
    plt.subplot(2, 2, 4)
    plt.title('Recon Imag')
    plt.plot(recon[1, ...])
    plt.show()

    if exp_params['transform_data']:
        print('Running data transformation of files...')
        fnmes, fdata = data.train_dataset.get_filedata(concat=False)
        save_path = param_dict['generate_data_settings']['local_path'] if (
            param_dict)['generate_data_settings']['use_local_storage'] else param_dict['dataset_params']['data_path']
        for fn, dt in zip(fnmes, fdata):
            out_data = model.encode(torch.tensor(dt, dtype=torch.float32)).data.numpy()
            with open(
                    f'{save_path}/{fn.split("/")[-1].split(".")[0]}.enc', 'ab') as writer:
                out_data.tofile(writer)
        target_spec_files = glob(f'{save_path}/targets.spec')[0]
        out_data = model.encode(torch.tensor(
            np.fromfile(target_spec_files, dtype=np.float32).reshape((-1, 2, param_dict['settings']['fft_len'])))).data.numpy()
        with open(
                f'{save_path}/targets.enc', 'ab') as writer:
            out_data.tofile(writer)
