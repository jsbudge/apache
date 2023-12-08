import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
import yaml
from dataloaders import WaveDataModule
from experiment import GeneratorExperiment
from models import WAE_MMD, init_weights
from waveform_model import GeneratorModel
import optuna

print(f'Cuda is available? {torch.cuda.is_available()}')

with open('./vae_config.yaml') as y:
    param_dict = yaml.safe_load(y.read())

fft_len = param_dict['generate_data_settings']['fft_sz']
bin_bw = int(param_dict['settings']['bandwidth'] // (2e9 / fft_len))
bin_bw += 1 if bin_bw % 2 != 0 else 0
vae_mdl = WAE_MMD(**param_dict['model_params'])
vae_mdl.load_state_dict(torch.load('./model/inference_model.state'))
vae_mdl.eval()  # Set to inference mode

vae_mdl.to('cpu')


def objective(trial: optuna.Trial):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_sz = trial.suggest_categorical('batch_size', [32, 64, 128])
    weight_decay = trial.suggest_float('weight_decay', 0.0, .99, step=.01)
    kld_weight = trial.suggest_float('kld_weight', 0.0, 1.0, step=.001)
    lr = trial.suggest_categorical('lr', [.0005, .005, .0001, .01])

    param_dict['wave_exp_params']['weight_decay'] = weight_decay
    param_dict['wave_exp_params']['kld_weight'] = kld_weight
    param_dict['wave_exp_params']['LR'] = lr

    param_dict['dataset_params']['train_batch_size'] = batch_sz
    param_dict['dataset_params']['val_batch_size'] = batch_sz

    wave_mdl = GeneratorModel(bin_bw=bin_bw, clutter_latent_size=param_dict['model_params']['latent_dim'],
                              target_latent_size=param_dict['model_params']['latent_dim'], n_ants=2)

    wave_mdl.apply(init_weights)

    data = WaveDataModule(vae_model=vae_mdl, device=device, **param_dict["dataset_params"])
    data.setup()
    param_dict['wave_exp_params']['is_tuning'] = True

    experiment = GeneratorExperiment(wave_mdl, param_dict['wave_exp_params'])
    trainer = Trainer(logger=False, max_epochs=param_dict['train_params']['max_epochs'], enable_checkpointing=False,
                      devices=1, callbacks=[EarlyStopping(patience=5000, monitor='loss',
                                                          check_finite=True)])
    trainer.fit(experiment, datamodule=data)

    return trainer.callback_metrics['loss'].item()


study = optuna.create_study()
study.optimize(objective, n_trials=150)

print(study.best_params)

optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_contour(study).show()
optuna.visualization.plot_parallel_coordinate(study).show()

