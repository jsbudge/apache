import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
import yaml

from config import get_config
from dataloaders import EncoderModule
from models import init_weights, Encoder
import optuna
import sys

from target_train import setupTrainer

print(f'Cuda is available? {torch.cuda.is_available()}')
try:
    pref_device = int(sys.argv[1])
    opt_study = sys.argv[2]
except Exception:
    pref_device = 0
    opt_study = './logs/opt0.txt'


def log_callback(curr_study: optuna.Study, trial: optuna.Trial):
    if curr_study.best_trial.number == trial.number:
        params = trial.params
        param_str = ' '.join([f'{key} {val}' for key, val in params.items()])
        with open(opt_study, 'a') as f:
            f.write(f'Trial {trial.number}: {param_str} ({curr_study.best_value})\n')


with open('./vae_config.yaml') as y:
    param_dict = yaml.safe_load(y.read())


def objective(trial: optuna.Trial):
    target_config = get_config('target_exp', './vae_config.yaml')

    weight_decay = trial.suggest_float('weight_decay', 0.0, .99, step=.01)
    lr = trial.suggest_categorical('lr', [.00000001, .001, .0001, .01, .00001, .000001, .0000001])
    scheduler_gamma = trial.suggest_float('scheduler_gamma', .1, .99, step=.05)
    beta0 = trial.suggest_float('beta0', .1, .99, step=.05)
    beta1 = trial.suggest_float('beta1', .1, .99, step=.05)
    latent_dim = trial.suggest_int('latent_dim', 10, 2048, 32)

    target_config.weight_decay = weight_decay
    target_config.lr = lr
    target_config.scheduler_gamma = scheduler_gamma
    target_config.betas = [beta0, beta1]
    target_config.latent_dim = latent_dim

    target_config.max_epochs = 5
    trainer, model, data = setupTrainer(pref_device, target_config, do_logs=False, trainer_args={'enable_checkpointing': False})
    trainer.fit(model, datamodule=data)

    return trainer.callback_metrics['val_loss'].item()


study = optuna.create_study()
study.optimize(objective, n_trials=20, callbacks=[log_callback])

print(study.best_params)

optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_contour(study).show()
optuna.visualization.plot_parallel_coordinate(study).show()