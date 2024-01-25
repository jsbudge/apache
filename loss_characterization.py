import numpy as np
from simulib.simulation_functions import genPulse, findPowerOf2, db
# import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from scipy.signal.windows import taylor
from scipy.signal import stft, butter, sosfilt, istft
from scipy.stats import rayleigh
from data_converter.SDRParsing import SDRParse, load
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pickle
import torch
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
import yaml
from glob import glob
from torchvision import transforms
from pathlib import Path
from dataloaders import CovDataModule, WaveDataModule
from experiment import VAExperiment, GeneratorExperiment
from models import BetaVAE, InfoVAE, WAE_MMD
from waveform_model import GeneratorModel, init_weights

fs = 2e9
c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
m_to_ft = 3.2808

with open('./vae_config.yaml', 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

fft_len = config['generate_data_settings']['fft_sz']
bin_bw = int(config['settings']['bandwidth'] // (fs / fft_len))
bin_bw += 1 if bin_bw % 2 != 0 else 0

stft_bw = int(config['settings']['bandwidth'] // (fs / config['settings']['stft_win_sz']))
stft_bw += 1 if stft_bw % 2 != 0 else 0

franges = np.linspace(config['perf_params']['vehicle_slant_range_min'],
                      config['perf_params']['vehicle_slant_range_max'], 1000) * 2 / c0
nrange = franges[0]
pulse_length = (nrange - 1 / TAC) * config['settings']['plp']
duty_cycle_time_s = pulse_length + franges
nr = int(pulse_length * fs)

stft_tbins = int(np.ceil(nr / (config['settings']['stft_win_sz'] / 4)))

if config['exp_params']['model_type'] == 'InfoVAE':
    vae_mdl = InfoVAE(**config['model_params'])
elif config['exp_params']['model_type'] == 'WAE_MMD':
    vae_mdl = WAE_MMD(**config['model_params'])
else:
    vae_mdl = BetaVAE(**config['model_params'])
vae_mdl.load_state_dict(torch.load('./model/inference_model.state'))
vae_mdl.eval()  # Set to inference mode

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = WaveDataModule(vae_model=vae_mdl, device=device, **config["dataset_params"])
data.setup()

wave_mdl = GeneratorModel(bin_bw=bin_bw, stft_params=(stft_bw, stft_tbins),
                          stft_win_sz=config['settings']['stft_win_sz'],
                          clutter_latent_size=config['model_params']['latent_dim'],
                          target_latent_size=config['model_params']['latent_dim'], n_ants=2)

wave_mdl.apply(init_weights)
iters = 1000
loss_n = np.zeros((iters, 4))

for n in tqdm(range(iters)):
    cc, tc, cs, ts = next(iter(data.train_dataloader()))
    # Generate a random noise function
    noise_output = torch.randn((cs.shape[0], wave_mdl.n_ants * 2, wave_mdl.stft_bw, wave_mdl.stft_sz), dtype=torch.float32)
    losses = wave_mdl.loss_function(noise_output, cs, ts)
    for idx, (key, val) in enumerate(losses.items()):
        loss_n[n, idx] = val
