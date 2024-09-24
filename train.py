from clearml import Task
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
import yaml
from dataloaders import EncoderModule
from models import init_weights, Encoder
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def getModelTrainerDataModule(a_params):
    ep = a_params['exp_params']

    a_data = EncoderModule(fft_len=a_params['settings']['fft_len'], **ep["dataset_params"])
    a_data.setup()

    # Get the model, experiment, logger set up
    a_model = Encoder(**ep['model_params'], fft_len=a_params['settings']['fft_len'], params=ep)
    print('Setting up model...')
    if ep['warm_start']:
        print('Model loaded from save state.')
        try:
            a_model.load_state_dict(torch.load('./model/inference_model.state'))
        except RuntimeError:
            print('Model save file does not match current structure. Re-running with new structure.')
            a_model.apply(init_weights)
    else:
        print('Initializing new model...')
        a_model.apply(init_weights)

    logger = loggers.TensorBoardLogger(a_params['train_params']['log_dir'], name="Encoder")
    expected_lr = max((ep['LR'] *
                       ep['scheduler_gamma'] ** (ep['max_epochs'] *
                                                         ep['swa_start'])), 1e-9)
    a_trainer = Trainer(logger=logger, max_epochs=ep['max_epochs'],
                      log_every_n_steps=ep['log_epoch'], devices=[0], callbacks=
                      [EarlyStopping(monitor='val_loss', patience=ep['patience'],
                                     check_finite=True),
                       StochasticWeightAveraging(swa_lrs=expected_lr, swa_epoch_start=ep['swa_start'])])
    return a_model, a_data, a_trainer

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(np.random.randint(1, 2048), workers=True)
    # seed_everything(44, workers=True)

    with open('./vae_config.yaml') as y:
        param_dict = yaml.safe_load(y.read())

    exp_params = param_dict['exp_params']

    model, data, trainer = getModelTrainerDataModule(param_dict)
    if exp_params['init_task']:
        task = Task.init(project_name='Encoder', task_name=exp_params['exp_name'])

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
            model.to('cuda:0')
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
                            torch.tensor(dt[:batch_sz, :, :-2], dtype=torch.float32, device=model.device)).cpu().data.numpy()
                        out_data.tofile(f)
                with open(
                        f'{save_path}/{fn.split("/")[-1].split(".")[0]}.enc', 'ab') as writer:
                    for chunk in np.arange(chunk_start, dt.shape[0], batch_sz):
                        out_data = model.encode(torch.tensor(dt[chunk:chunk + batch_sz, :, :-2], dtype=torch.float32, device=model.device)).cpu().data.numpy()
                        out_data.tofile(writer)
        if exp_params['init_task']:
            task.close()
