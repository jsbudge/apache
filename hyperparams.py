import torch
from pytorch_lightning import Trainer, loggers
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from simulib.simulation_functions import db
from dataloaders import DataModule
from experiment import VAExperiment
from models import BetaVAE, InfoVAE, WAE_MMD
import optuna

print(f'Cuda is available? {torch.cuda.is_available()}')

with open('./vae_config.yaml') as y:
    param_dict = yaml.safe_load(y.read())


def objective(trial: optuna.Trial):

    lr = trial.suggest_uniform('learning_rate', 1e-5, 5e-3)
    batch_sz = trial.suggest_categorical('batch_size', [32, 64])
    latent_dim = trial.suggest_int('latent_dim', 3, 128)
    reg_weight = trial.suggest_int('reg_weight', 110, 5000, step=200)
    kernel_type = trial.suggest_categorical('kernel', ['rbf', 'imq'])

    param_dict['dataset_params']['train_batch_size'] = batch_sz
    param_dict['dataset_params']['val_batch_size'] = batch_sz

    data = DataModule(**param_dict['dataset_params'])
    data.setup()

    param_dict['model_params']['kernel_type'] = kernel_type
    param_dict['model_params']['reg_weight'] = reg_weight
    param_dict['model_params']['latent_dim'] = latent_dim

    # Get the model, experiment, logger set up
    model = WAE_MMD(**param_dict['model_params'])

    param_dict['exp_params']['LR'] = lr
    experiment = VAExperiment(model, param_dict['exp_params'])
    trainer = Trainer(logger=False, max_epochs=param_dict['train_params']['max_epochs'], enable_checkpointing=False)
    trainer.fit(experiment, datamodule=data)

    return trainer.callback_metrics['loss'].item()


study = optuna.create_study()
study.optimize(objective, n_trials=5000)

optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_contour(study).show()
optuna.visualization.plot_parallel_coordinate(study).show()