import torch
from pytorch_lightning import Trainer, loggers
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from dataloaders import CovDataModule
from experiment import VAExperiment
from models import BetaVAE, InfoVAE, WAE_MMD
import optuna

print(f'Cuda is available? {torch.cuda.is_available()}')

with open('./vae_config.yaml') as y:
    param_dict = yaml.safe_load(y.read())


def objective(trial: optuna.Trial):
    # Get the model, experiment, logger set up
    if param_dict['exp_params']['model_type'] == 'InfoVAE':
        model = InfoVAE(**param_dict['model_params'])
        beta = trial.suggest_float('beta', 1.0, 20.0)
        alpha = trial.suggest_float('alpha', -12.0, -1.0)
        param_dict['model_params']['alpha'] = alpha
        param_dict['model_params']['beta'] = beta
    elif param_dict['exp_params']['model_type'] == 'WAE_MMD':
        model = WAE_MMD(**param_dict['model_params'])
        reg_weight = trial.suggest_int('reg_weight', 100, 5000, 10)
        kernel_type = trial.suggest_categorical('kernel', ['imq', 'rbf'])
        param_dict['model_params']['reg_weight'] = reg_weight
        param_dict['model_params']['kernel_type'] = kernel_type
    else:
        model = BetaVAE(**param_dict['model_params'])
        gamma = trial.suggest_int('gamma', 30, 5000, step=200)
        kernel_type = trial.suggest_categorical('loss', ['H', 'B'])
        param_dict['model_params']['gamma'] = gamma
        param_dict['model_params']['loss_type'] = kernel_type
    batch_sz = trial.suggest_categorical('batch_size', [32, 64, 128])
    latent_dim = trial.suggest_int('latent_dim', 3, 128)
    weight_decay = trial.suggest_float('weight_decay', 0.0, .99, step=.01)
    kld_weight = trial.suggest_float('kld_weight', 0.0, 1.0, step=.0001)

    param_dict['exp_params']['weight_decay'] = weight_decay
    param_dict['exp_params']['kld_weight'] = kld_weight

    param_dict['dataset_params']['train_batch_size'] = batch_sz
    param_dict['dataset_params']['val_batch_size'] = batch_sz

    data = CovDataModule(**param_dict['dataset_params'])
    data.setup()
    param_dict['model_params']['latent_dim'] = latent_dim
    param_dict['exp_params']['is_tuning'] = True

    experiment = VAExperiment(model, param_dict['exp_params'])
    trainer = Trainer(logger=False, max_epochs=param_dict['train_params']['max_epochs'], enable_checkpointing=False,
                      devices=1)
    trainer.fit(experiment, datamodule=data)

    return trainer.callback_metrics['loss'].item()


study = optuna.create_study()
study.optimize(objective, n_trials=500)

print(study.best_params)

optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_contour(study).show()
optuna.visualization.plot_parallel_coordinate(study).show()