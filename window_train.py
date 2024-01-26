import numpy as np
from simulib.simulation_functions import genPulse, findPowerOf2, db
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
import plotly.io as pio
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
import yaml
from dataloaders import WindowModule
from experiment import WinExperiment
from waveform_model import WindowModel

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

fs = 2e9
c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
m_to_ft = 3.2808


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.cuda.empty_cache()

    seed_everything(43, workers=True)

    with open('./vae_config.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    print('Setting up data generator...')
    win_mdl = WindowModel(256, 2)

    data = WindowModule(device=device, dataset_size=16384, **config["dataset_params"])
    data.setup()

    print('Setting up experiment...')
    experiment = WinExperiment(win_mdl, config['wave_exp_params'])
    logger = loggers.TensorBoardLogger(config['train_params']['log_dir'],
                                       name="WinModel")
    trainer = Trainer(logger=logger, max_epochs=config['train_params']['max_epochs'],
                      log_every_n_steps=config['exp_params']['log_epoch'],
                      strategy='ddp',
                      callbacks=[EarlyStopping(monitor='loss', patience=30,
                                               check_finite=True)])

    print("======= Training =======")
    trainer.fit(experiment, datamodule=data)

    if trainer.global_rank == 0:
        win_mdl.eval()
        inp, oup = next(iter(data.train_dataloader()))

        windows = win_mdl(inp).data.numpy()

        plt.figure('Window Comparison')
        plt.plot(oup.data.numpy()[0, 0, :])
        plt.plot(windows[0, 0, :])
    if trainer.is_global_zero:
        try:
            torch.save(win_mdl.state_dict(), './model/win_model.state')
            print('Model saved to disk.')
        except Exception as e:
            print(f'Model not saved: {e}')
