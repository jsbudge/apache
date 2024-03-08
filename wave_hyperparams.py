import pickle

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
import yaml
from dataloaders import WaveDataModule
from experiment import GeneratorExperiment
from waveform_model import GeneratorModel
import optuna
import numpy as np
import sys

fs = 2e9
c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
m_to_ft = 3.2808

print(f'Cuda is available? {torch.cuda.is_available()}')
try:
    pref_device = int(sys.argv[1])
    opt_study = sys.argv[2]
except Exception:
    pref_device = 0
    opt_study = './logs/opt0.txt'

with open('./vae_config.yaml') as y:
    config = yaml.safe_load(y.read())

fft_len = config['generate_data_settings']['fft_sz']
config['wave_exp_params']['nr'] = 5000
config['dataset_params']['max_pulse_length'] = 5000
config['dataset_params']['min_pulse_length'] = 1000


def log_callback(curr_study: optuna.Study, trial: optuna.Trial):
    if curr_study.best_trial.number == trial.number:
        params = trial.params
        param_str = ' '.join([f'{key} {val:.03f}' for key, val in params.items()])
        with open(opt_study, 'a') as f:
            f.write(f'Trial {trial.number}: {param_str} ({curr_study.best_value})\n')


def objective(trial: optuna.Trial):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    weight_decay = trial.suggest_float('weight_decay', 0.0, .99, step=.01)
    lr = trial.suggest_categorical('lr', [.001, .0001, .00001, .00005, .000001, .000005, .0000001])
    gamma = trial.suggest_float('scheduler_gamma', .01, .99, step=.01)
    step_size = trial.suggest_int('step_size', 1, 10)
    beta0 = trial.suggest_float('beta0', 0.01, .99, step=.01)
    beta1 = trial.suggest_float('beta1', 0.01, .99, step=.01)

    config['wave_exp_params']['weight_decay'] = weight_decay
    config['wave_exp_params']['LR'] = lr
    config['wave_exp_params']['scheduler_gamma'] = gamma
    config['wave_exp_params']['step_size'] = step_size
    config['wave_exp_params']['betas'] = [beta0, beta1]

    wave_mdl = GeneratorModel(fft_sz=fft_len,
                              stft_win_sz=config['settings']['stft_win_sz'],
                              clutter_latent_size=config['model_params']['latent_dim'],
                              target_latent_size=config['model_params']['latent_dim'], n_ants=2)

    data = WaveDataModule(latent_dim=config['model_params']['latent_dim'], device=device, **config["dataset_params"])
    data.setup()

    print('Setting up experiment...')
    config['wave_exp_params']['is_tuning'] = True

    experiment = GeneratorExperiment(wave_mdl, config['wave_exp_params'])
    expected_lr = (config['wave_exp_params']['LR'] *
                   config['wave_exp_params']['scheduler_gamma'] ** (config['train_params']['max_epochs'] * .8))
    trainer = Trainer(logger=False, max_epochs=config['train_params']['max_epochs'], enable_checkpointing=False,
                      devices=[pref_device], accelerator='gpu',
                      callbacks=[EarlyStopping(patience=15, monitor='loss', check_finite=True),
                                 StochasticWeightAveraging(swa_lrs=expected_lr)])
    trainer.fit(experiment, datamodule=data)

    return trainer.callback_metrics['loss'].item()


study = optuna.create_study()
study.optimize(objective, n_trials=500, callbacks=[log_callback])

print(study.best_params)

optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_contour(study).show()
optuna.visualization.plot_parallel_coordinate(study).show()
