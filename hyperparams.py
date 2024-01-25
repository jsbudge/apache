import torch
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import EarlyStopping
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from dataloaders import CovDataModule
from experiment import VAExperiment
from models import BetaVAE, InfoVAE, WAE_MMD, init_weights
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
        reg_weight = trial.suggest_float('reg_weight', 10., 10000., step=10.)
        kernel_type = trial.suggest_categorical('kernel', ['imq', 'rbf'])
        param_dict['model_params']['reg_weight'] = reg_weight
        param_dict['model_params']['kernel_type'] = kernel_type
    else:
        model = BetaVAE(**param_dict['model_params'])
        gamma = trial.suggest_float('gamma', 30., 5000., step=5.)
        kernel_type = trial.suggest_categorical('loss', ['H', 'B'])
        param_dict['model_params']['gamma'] = gamma
        param_dict['model_params']['loss_type'] = kernel_type
    model.apply(init_weights)

    batch_sz = trial.suggest_categorical('batch_size', [32, 64, 128])
    latent_dim = trial.suggest_int('latent_dim', 3, 128)
    weight_decay = trial.suggest_float('weight_decay', 0.0, .99, step=.01)
    kld_weight = trial.suggest_float('kld_weight', 0.0, 1.0, step=.01)
    lr = trial.suggest_categorical('lr', [.0005, .005, .0001, .01, .00001, .000001, .00005])

    param_dict['exp_params']['weight_decay'] = weight_decay
    param_dict['exp_params']['kld_weight'] = kld_weight
    param_dict['exp_params']['LR'] = lr

    param_dict['dataset_params']['train_batch_size'] = batch_sz
    param_dict['dataset_params']['val_batch_size'] = batch_sz

    data = CovDataModule(**param_dict['dataset_params'])
    data.setup()
    param_dict['model_params']['latent_dim'] = latent_dim
    param_dict['exp_params']['is_tuning'] = True

    experiment = VAExperiment(model, param_dict['exp_params'])
    trainer = Trainer(logger=False, max_epochs=param_dict['train_params']['max_epochs'], enable_checkpointing=False,
                      devices=1, callbacks=[EarlyStopping(patience=5, monitor='Reconstruction_Loss',
                                                          check_finite=True)])
    trainer.fit(experiment, datamodule=data)

    return trainer.callback_metrics['Reconstruction_Loss'].item()


study = optuna.create_study()
study.optimize(objective, n_trials=5000)

print(study.best_params)

optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_contour(study).show()
optuna.visualization.plot_parallel_coordinate(study).show()