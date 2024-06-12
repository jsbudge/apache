import numpy as np
from simulib.simulation_functions import genPulse, findPowerOf2, db
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
import plotly.io as pio
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
import yaml
from dataloaders import RCSModule
from experiment import RCSExperiment
from waveform_model import RCSModel

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
    torch.set_float32_matmul_precision('medium')
    # torch.cuda.empty_cache()

    seed_everything(np.random.randint(1, 2048), workers=True)

    with open('./vae_config.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    print('Setting up data generator...')
    win_mdl = RCSModel()
    # win_mdl.load_state_dict(torch.load('./model/rcs_model.state'))

    data = RCSModule(device=device, **config["dataset_params"])
    data.setup()

    print('Setting up experiment...')
    experiment = RCSExperiment(win_mdl, config['rcs_exp_params'])
    logger = loggers.TensorBoardLogger(config['train_params']['log_dir'],
                                       name="RCSModel")
    trainer = Trainer(logger=logger, max_epochs=config['train_params']['max_epochs'],
                      log_every_n_steps=config['exp_params']['log_epoch'],
                      strategy='ddp', devices=1,
                      callbacks=[EarlyStopping(monitor='val_loss', patience=30,
                                               check_finite=True)])

    print("======= Training =======")
    trainer.fit(experiment, datamodule=data)

    if trainer.global_rank == 0:
        with torch.no_grad():
            win_mdl.to(device)
            win_mdl.eval()
            inp, oup, params = next(iter(data.train_dataloader()))
            inp = inp.to(device)
            oup = oup.to(device)
            params = params.to(device)

            windows = win_mdl(inp, params).cpu().data.numpy()

        grid_sz = int(np.ceil(np.sqrt(windows.shape[0])))
        plt.figure('NN output')
        for n in range(windows.shape[0]):
            plt.subplot(grid_sz, grid_sz, n + 1)
            plt.imshow(windows[n, 0, ...])
            plt.axis('off')
        plt.figure('Original SAR data')
        for n in range(windows.shape[0]):
            plt.subplot(grid_sz, grid_sz, n + 1)
            plt.imshow(oup[n, 0, ...].cpu().data.numpy())
            plt.axis('off')
        plt.figure('Google Map output')
        for n in range(windows.shape[0]):
            plt.subplot(grid_sz, grid_sz, n + 1)
            plt.imshow(inp[n, :3, ...].cpu().data.numpy().swapaxes(0, 2))
            plt.axis('off')

    if trainer.is_global_zero:
        try:
            torch.save(win_mdl.state_dict(), './model/rcs_model.state')
            print('Model saved to disk.')
        except Exception as e:
            print(f'Model not saved: {e}')
