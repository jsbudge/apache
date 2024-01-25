import numpy as np
import torch
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
import yaml
from dataloaders import WaveDataModule
from experiment import GeneratorExperiment
from models import BetaVAE, InfoVAE, WAE_MMD
from waveform_model import GeneratorModel, init_weights
import argparse
import os

fs = 2e9
c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
m_to_ft = 3.2808


def upsample(val, fac=8):
    upval = np.zeros(len(val) * fac, dtype=np.complex128)
    upval[:len(val) // 2] = val[:len(val) // 2]
    upval[-len(val) // 2:] = val[-len(val) // 2:]
    return upval


'''def outBeamTime(theta_az, theta_el):
    return (np.pi ** 2 * wheel_height_m - 8 * np.pi * blade_chord_m * np.tan(theta_el) -
            4 * wheel_height_m * theta_az) / (8 * np.pi * wheel_height_m * rotor_velocity_rad_s)'''


def buildWaveform(wd, fft_len, bin_bw):
    ret = np.zeros((wd.shape[0], wd.shape[1] // 2, fft_len), dtype=np.complex64)
    ret[:, :, :bin_bw // 2] = wd[:, ::2, -bin_bw // 2:] + 1j * wd[:, 1::2, -bin_bw // 2:]
    ret[:, :, -bin_bw // 2:] = wd[:, ::2, :bin_bw // 2] + 1j * wd[:, 1::2, :bin_bw // 2]
    return normalize(ret)


def normalize(data):
    return data / np.expand_dims(np.sqrt(np.sum(data * data.conj(), axis=-1).real), axis=len(data.shape) - 1)


def getRange(alt, theta_el):
    return alt * np.sin(theta_el) * 2 / c0


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--master_port", type=int,
                        help="Master port to connect to.")
    parser.add_argument("--master_addr", type=str,
                        help="Master IP address.")
    parser.add_argument("--world_size", type=int,
                        help="World size, number of computers on network.")
    parser.add_argument("--local_rank", type=int,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    argv = parser.parse_args()

    os.environ['MASTER_PORT'] = str(argv.master_port)
    os.environ['MASTER_ADDR'] = argv.master_addr
    os.environ['WORLD_SIZE'] = str(argv.world_size)
    os.environ['NODE_RANK'] = str(argv.local_rank)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    strat = DDPStrategy(process_group_backend='nccl')
    seed_everything(43, workers=True)

    with open('./vae_config.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    fft_len = config['generate_data_settings']['fft_sz']
    bin_bw = int(config['settings']['bandwidth'] // (fs / fft_len))
    bin_bw += 1 if bin_bw % 2 != 0 else 0

    franges = np.linspace(config['perf_params']['vehicle_slant_range_min'],
                          config['perf_params']['vehicle_slant_range_max'], 1000) * 2 / c0
    nrange = franges[0]
    pulse_length = (nrange - 1 / TAC) * config['settings']['plp']
    duty_cycle_time_s = pulse_length + franges
    nr = int(pulse_length * fs)

    # Get the VAE set up
    print('Setting up model...')
    if config['exp_params']['model_type'] == 'InfoVAE':
        vae_mdl = InfoVAE(**config['model_params'])
    elif config['exp_params']['model_type'] == 'WAE_MMD':
        vae_mdl = WAE_MMD(**config['model_params'])
    else:
        vae_mdl = BetaVAE(**config['model_params'])
    vae_mdl.load_state_dict(torch.load('./model/inference_model.state'))
    vae_mdl.eval()  # Set to inference mode
    # vae_mdl.to(device)  # Move to GPU

    print('Setting up data generator...')
    wave_mdl = GeneratorModel(bin_bw=bin_bw, clutter_latent_size=config['model_params']['latent_dim'],
                              target_latent_size=config['model_params']['latent_dim'], n_ants=2)

    wave_mdl.apply(init_weights)

    data = WaveDataModule(vae_model=vae_mdl, device=device, **config["dataset_params"])
    data.setup()

    vae_mdl.to('cpu')

    print('Setting up experiment...')
    experiment = GeneratorExperiment(wave_mdl, config['wave_exp_params'])
    logger = loggers.TensorBoardLogger(config['train_params']['log_dir'],
                                       name="WaveModel")
    trainer = Trainer(logger=logger, max_epochs=config['train_params']['max_epochs'],
                      log_every_n_steps=config['exp_params']['log_epoch'],
                      callbacks=[EarlyStopping(monitor='loss', patience=config['wave_exp_params']['patience'],
                                               check_finite=True)],
                      strategy=strat, devices=-1, num_nodes=2)

    print("======= Training =======")
    trainer.fit(experiment, datamodule=data)
