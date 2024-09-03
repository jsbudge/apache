import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
import yaml
from dataloaders import EncoderModule
from models import init_weights, Encoder
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

    batch_sz = trial.suggest_categorical('batch_size', [2, 8, 16])
    weight_decay = trial.suggest_float('weight_decay', 0.0, .99, step=.01)
    lr = trial.suggest_categorical('lr', [.00000001, .001, .0001, .01, .00001, .000001, .0000001])
    swa_start = trial.suggest_float('swa_start', .1, .9, step=.1)
    scheduler_gamma = trial.suggest_float('scheduler_gamma', .1, .99, step=.01)
    beta0 = trial.suggest_float('beta0', .01, .99, step=.01)
    beta1 = trial.suggest_float('beta1', .01, .99, step=.01)
    step_size = trial.suggest_int('step_size', 1, 5, step=1)

    param_dict['exp_params']['weight_decay'] = weight_decay
    param_dict['exp_params']['LR'] = lr
    param_dict['exp_params']['swa_start'] = swa_start
    param_dict['exp_params']['scheduler_gamma'] = scheduler_gamma
    param_dict['exp_params']['betas'] = [beta0, beta1]
    param_dict['exp_params']['step_size'] = step_size

    param_dict['exp_params']['dataset_params']['train_batch_size'] = batch_sz
    param_dict['exp_params']['dataset_params']['val_batch_size'] = batch_sz

    data = EncoderModule(fft_len=param_dict['settings']['fft_len'], **param_dict['exp_params']["dataset_params"])
    data.setup()
    param_dict['exp_params']['is_tuning'] = True

    expected_lr = max((param_dict['exp_params']['LR'] *
                       param_dict['exp_params']['scheduler_gamma'] ** (param_dict['exp_params']['max_epochs'] *
                                                         param_dict['exp_params']['swa_start'])), 1e-9)
    model = Encoder(**param_dict['model_params'], fft_len=param_dict['settings']['fft_len'], params=param_dict['exp_params'])
    model.apply(init_weights)
    trainer = Trainer(logger=False, max_epochs=5, enable_checkpointing=False,
                      strategy='ddp', deterministic=True, devices=[0], callbacks=
                      [StochasticWeightAveraging(swa_lrs=expected_lr,
                                                 swa_epoch_start=param_dict['exp_params']['swa_start'])])
    trainer.fit(model, datamodule=data)

    return trainer.callback_metrics['val_loss'].item()


study = optuna.create_study()
study.optimize(objective, n_trials=20, callbacks=[log_callback])

print(study.best_params)

optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_contour(study).show()
optuna.visualization.plot_parallel_coordinate(study).show()