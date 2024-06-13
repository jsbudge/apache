from glob import glob
from clearml import Task
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
import yaml
from dataloaders import TargetEncoderModule
from models import init_weights, TargetEncoder
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

torch.set_float32_matmul_precision('medium')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# seed_everything(np.random.randint(1, 2048), workers=True)
seed_everything(43, workers=True)

with open('./vae_config.yaml') as y:
    param_dict = yaml.safe_load(y.read())

exp_params = param_dict['exp_params']
# hparams = make_dataclass('hparams', param_dict.items())(**param_dict)

data = TargetEncoderModule(**exp_params["dataset_params"])
data.setup()

# Get the model, experiment, logger set up
model = TargetEncoder(**param_dict['model_params'], fft_len=param_dict['settings']['fft_len'], params=exp_params)
print('Setting up model...')
tag_warm = 'new_model'
if exp_params['warm_start']:
    print('Model loaded from save state.')
    try:
        model.load_state_dict(torch.load('./model/inference_model.state'))
        tag_warm = 'warm_start'
    except RuntimeError:
        print('Model save file does not match current structure. Re-running with new structure.')
        model.apply(init_weights)
else:
    print('Initializing new model...')
    model.apply(init_weights)

task = Task.init(project_name='TargetEncoder', task_name=param_dict['exp_params']['exp_name'])

logger = loggers.TensorBoardLogger(param_dict['train_params']['log_dir'], name="TargetEncoder")
expected_lr = max((exp_params['LR'] *
                   exp_params['scheduler_gamma'] ** (exp_params['max_epochs'] *
                                                     exp_params['swa_start'])), 1e-9)
trainer = Trainer(logger=logger, max_epochs=exp_params['max_epochs'],
                  log_every_n_steps=exp_params['log_epoch'],
                  strategy='ddp', devices=1, callbacks=
                  [EarlyStopping(monitor='val_loss', patience=exp_params['patience'],
                                 check_finite=True),
                   StochasticWeightAveraging(swa_lrs=expected_lr, swa_epoch_start=exp_params['swa_start'])])
# trainer.test(model, train_loader, verbose=True)

print("======= Training =======")
try:
    trainer.fit(model, datamodule=data)
except KeyboardInterrupt:
    if trainer.is_global_zero:
        print('Training interrupted.')
    else:
        print('adios!')
        exit(0)

model.to('cpu')
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
        batch_sz = exp_params['dataset_params']['train_batch_size']
        fnmes, fdata = data.train_dataset.get_filedata(concat=False)
        save_path = param_dict['generate_data_settings']['local_path'] if (
            param_dict)['generate_data_settings']['use_local_storage'] else exp_params['dataset_params']['data_path']
        for fn, dt in zip(fnmes, fdata):
            enc_fnme = f'{save_path}/{fn.split("/")[-1].split(".")[0]}.enc'
            chunk_start = 0
            if Path(enc_fnme).exists():
                chunk_start = 1
                with open(enc_fnme, 'wb') as f:
                    out_data = model.encode(
                        torch.tensor(dt[:batch_sz, :, :-2], dtype=torch.float32)).data.numpy()
                    out_data.tofile(f)
            with open(
                    f'{save_path}/{fn.split("/")[-1].split(".")[0]}.enc', 'ab') as writer:
                for chunk in np.arange(chunk_start, dt.shape[0], batch_sz):
                    out_data = model.encode(torch.tensor(dt[chunk:chunk + batch_sz, :, :-2], dtype=torch.float32)).data.numpy()
                    out_data.tofile(writer)
        target_spec_files = glob(f'{save_path}/targets.spec')[0]
        target_data = np.fromfile(target_spec_files, dtype=np.float32).reshape((-1, 2, param_dict['settings']['fft_len'] + 2))[:, :, :-2]
        with open(
                f'{save_path}/targets.enc', 'ab') as writer:
            for td in np.arange(0, target_data.shape[0], batch_sz):
                out_data = model.encode(torch.tensor(target_data[td:td + batch_sz, ...])).data.numpy()
                out_data.tofile(writer)
task.close()
