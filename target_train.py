from glob import glob
from clearml import Task
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
from simulib.simulation_functions import db
import yaml
from dataloaders import TargetEncoderModule
from models import init_weights, TargetEncoder
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # seed_everything(np.random.randint(1, 2048), workers=True)
    seed_everything(43, workers=True)
    
    with open('./vae_config.yaml') as y:
        param_dict = yaml.safe_load(y.read())
    
    exp_params = param_dict['target_exp_params']
    # hparams = make_dataclass('hparams', param_dict.items())(**param_dict)
    
    data = TargetEncoderModule(**exp_params["dataset_params"])
    data.setup()
    
    # Get the model, experiment, logger set up
    model = TargetEncoder(**exp_params['model_params'], params=exp_params)
    print('Setting up model...')
    tag_warm = 'new_model'
    if exp_params['warm_start']:
        print('Model loaded from save state.')
        try:
            model.load_state_dict(torch.load('./model/target_model.state'))
            tag_warm = 'warm_start'
        except RuntimeError:
            print('Model save file does not match current structure. Re-running with new structure.')
            model.apply(init_weights)
    else:
        print('Initializing new model...')
        model.apply(init_weights)
    if exp_params['init_task']:
        task = Task.init(project_name='TargetEncoder', task_name=exp_params['exp_name'])
    
    logger = loggers.TensorBoardLogger(param_dict['train_params']['log_dir'], name="TargetEncoder")
    expected_lr = max((exp_params['LR'] *
                       exp_params['scheduler_gamma'] ** (exp_params['max_epochs'] *
                                                         exp_params['swa_start'])), 1e-9)
    trainer = Trainer(logger=logger, max_epochs=exp_params['max_epochs'],
                      log_every_n_steps=exp_params['log_epoch'], devices=1, callbacks=
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
                torch.save(model.state_dict(), './model/target_model.state')
                print('Model saved to disk.')
            except Exception as e:
                print(f'Model not saved: {e}')
        print('Plotting outputs...')
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.title('Sample')
        plt.imshow(db(sample[0] + 1j * sample[1]))
        plt.subplot(2, 1, 2)
        plt.title('Recon')
        plt.imshow(db(recon[0] + 1j * recon[1]))
        plt.show()
    
        if exp_params['transform_data']:
            print('Running data transformation of files...')
            save_path = param_dict['generate_data_settings']['local_path'] if (
                param_dict)['generate_data_settings']['use_local_storage'] else exp_params['dataset_params']['data_path']
            target_data = np.fromfile(f'{save_path}/targetpatterns.dat', np.float32).reshape((-1, 2, 256, 256))
            with open(
                    f'{save_path}/targets.enc', 'wb') as writer:
                out_data = model.encode(torch.tensor(target_data)).data.numpy()
                out_data.tofile(writer)
    if exp_params['init_task']:
        task.close()
