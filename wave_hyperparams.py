import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
import yaml
from dataloaders import WaveDataModule
from experiment import GeneratorExperiment
from models import WAE_MMD, init_weights
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
except:
    pref_device = 0
    opt_study = './logs/opt0.txt'

with open('./vae_config.yaml') as y:
    param_dict = yaml.safe_load(y.read())

fft_len = param_dict['generate_data_settings']['fft_sz']
bin_bw = int(param_dict['settings']['bandwidth'] // (2e9 / fft_len))
bin_bw += 1 if bin_bw % 2 != 0 else 0

stft_bw = int(param_dict['settings']['bandwidth'] // (fs / param_dict['settings']['stft_win_sz']))
stft_bw += 1 if stft_bw % 2 != 0 else 0

franges = np.linspace(param_dict['perf_params']['vehicle_slant_range_min'],
                      param_dict['perf_params']['vehicle_slant_range_max'], 1000) * 2 / c0
nrange = franges[0]
pulse_length = (nrange - 1 / TAC) * param_dict['settings']['plp']
duty_cycle_time_s = pulse_length + franges
nr = int(pulse_length * fs)
param_dict['wave_exp_params']['nr'] = nr

stft_tbins = int(np.ceil(nr / (param_dict['settings']['stft_win_sz'] / 4)))
vae_mdl = WAE_MMD(**param_dict['model_params'])
vae_mdl.load_state_dict(torch.load('./model/inference_model.state'))
vae_mdl.eval()  # Set to inference mode

vae_mdl.to('cpu')


def log_callback(curr_study: optuna.Study, trial: optuna.Trial):
    if curr_study.best_trial.number == trial.number:
        params = trial.params
        with open(opt_study, 'a') as f:
            f.write(f'Trial {trial.number}: batch_sz {trial.params["batch_size"]} '
                    f'weight_decay {trial.params["weight_decay"]} kld_weight {trial.params["kld_weight"]} '
                    f'lr {trial.params["lr"]} ({curr_study.best_value})\n')


def objective(trial: optuna.Trial):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_sz = trial.suggest_categorical('batch_size', [32, 64, 128])
    weight_decay = trial.suggest_float('weight_decay', 0.0, .99, step=.01)
    kld_weight = trial.suggest_float('kld_weight', 0.0, 1.0, step=.01)
    lr = trial.suggest_categorical('lr', [.00001, .00005, .000001, .000005, .0000001, .0000005, .00000001,
                                          .00000005, .000000001, .000000005])

    param_dict['wave_exp_params']['weight_decay'] = weight_decay
    param_dict['wave_exp_params']['kld_weight'] = kld_weight
    param_dict['wave_exp_params']['LR'] = lr

    param_dict['dataset_params']['train_batch_size'] = batch_sz
    param_dict['dataset_params']['val_batch_size'] = batch_sz

    wave_mdl = GeneratorModel(fft_sz=fft_len,
                              stft_win_sz=param_dict['settings']['stft_win_sz'],
                              clutter_latent_size=param_dict['model_params']['latent_dim'],
                              target_latent_size=param_dict['model_params']['latent_dim'], n_ants=2)

    data = WaveDataModule(vae_model=vae_mdl, device=device, **param_dict["dataset_params"])
    data.setup()
    param_dict['wave_exp_params']['is_tuning'] = True

    experiment = GeneratorExperiment(wave_mdl, param_dict['wave_exp_params'])
    trainer = Trainer(logger=False, max_epochs=param_dict['train_params']['max_epochs'], enable_checkpointing=False,
                      devices=[pref_device], accelerator='gpu', callbacks=[EarlyStopping(patience=5, monitor='loss',
                                                                                         check_finite=True)])
    trainer.fit(experiment, datamodule=data)

    return trainer.callback_metrics['loss'].item()


study = optuna.create_study()
study.optimize(objective, n_trials=150, callbacks=[log_callback])

print(study.best_params)

optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_contour(study).show()
optuna.visualization.plot_parallel_coordinate(study).show()
