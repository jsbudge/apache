import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
import yaml
from dataloaders import EncoderModule
from experiment import VAExperiment
from models import BetaVAE, InfoVAE, WAE_MMD, init_weights
import optuna
import sys


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
    # Get the model, experiment, logger set up
    if param_dict['exp_params']['model_type'] == 'InfoVAE':
        model = InfoVAE(**param_dict['model_params'])
        beta = trial.suggest_float('beta', 1.0, 20.0)
        alpha = trial.suggest_float('alpha', -12.0, -1.0)
        param_dict['model_params']['alpha'] = alpha
        param_dict['model_params']['beta'] = beta
    elif param_dict['exp_params']['model_type'] == 'WAE_MMD':
        model = WAE_MMD(fft_len=param_dict['settings']['fft_len'], **param_dict['model_params'])
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

    batch_sz = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    weight_decay = trial.suggest_float('weight_decay', 0.0, .99, step=.01)
    lr = trial.suggest_categorical('lr', [.0005, .005, .0001, .01, .00001, .000001, .00005])
    swa_start = trial.suggest_float('swa_start', .1, .9, step=.1)
    scheduler_gamma = trial.suggest_float('scheduler_gamma', .1, .99, step=.01)
    kld_weight = trial.suggest_float('kld_weight', .01, .99, step=.01)

    param_dict['exp_params']['weight_decay'] = weight_decay
    param_dict['exp_params']['LR'] = lr
    param_dict['exp_params']['swa_start'] = swa_start
    param_dict['exp_params']['scheduler_gamma'] = scheduler_gamma
    param_dict['exp_params']['kld_weight'] = kld_weight

    param_dict['dataset_params']['train_batch_size'] = batch_sz
    param_dict['dataset_params']['val_batch_size'] = batch_sz

    data = EncoderModule(fft_len=param_dict['settings']['fft_len'], **param_dict["dataset_params"])
    data.setup()
    param_dict['exp_params']['is_tuning'] = True

    expected_lr = max((param_dict['exp_params']['LR'] *
                       param_dict['exp_params']['scheduler_gamma'] ** (param_dict['exp_params']['max_epochs'] *
                                                         param_dict['exp_params']['swa_start'])), 1e-9)
    experiment = VAExperiment(model, param_dict['exp_params'])
    trainer = Trainer(logger=False, max_epochs=10, enable_checkpointing=False,
                      strategy='ddp', deterministic=True, devices=1, callbacks=
                      [EarlyStopping(monitor='loss', patience=5,
                                     check_finite=True),
                       StochasticWeightAveraging(swa_lrs=expected_lr,
                                                 swa_epoch_start=param_dict['exp_params']['swa_start'])])
    trainer.fit(experiment, datamodule=data)

    return trainer.callback_metrics['Reconstruction_Loss'].item()


study = optuna.create_study()
study.optimize(objective, n_trials=140, callbacks=[log_callback])

print(study.best_params)

optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_contour(study).show()
optuna.visualization.plot_parallel_coordinate(study).show()