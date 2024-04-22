import pickle

import numpy as np
from simulib import genPulse, findPowerOf2, db
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, ModelPruning
import yaml
from dataloaders import WaveDataModule
from experiment import GeneratorExperiment
from models import BetaVAE, InfoVAE, WAE_MMD
from waveform_model import GeneratorModel, init_weights
from os import listdir
import torch.nn as nn
import torch.nn.utils.prune as prune

fs = 2e9
c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
m_to_ft = 3.2808


def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


def upsample(val, fac=8):
    upval = np.zeros(len(val) * fac, dtype=np.complex128)
    upval[:len(val) // 2] = val[:len(val) // 2]
    upval[-len(val) // 2:] = val[-len(val) // 2:]
    return upval


'''def outBeamTime(theta_az, theta_el):
    return (np.pi ** 2 * wheel_height_m - 8 * np.pi * blade_chord_m * np.tan(theta_el) -
            4 * wheel_height_m * theta_az) / (8 * np.pi * wheel_height_m * rotor_velocity_rad_s)'''


def normalize(data):
    return data / np.expand_dims(np.sqrt(np.sum(data * data.conj(), axis=-1).real), axis=len(data.shape) - 1)


def getRange(alt, theta_el):
    return alt * np.sin(theta_el) * 2 / c0


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    force_cudnn_initialization()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.cuda.empty_cache()

    # seed_everything(np.random.randint(1, 2048), workers=True)
    seed_everything(42, workers=True)

    with open('./vae_config.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    fft_len = config['settings']['fft_len']
    nr = 5000  # int((config['perf_params']['vehicle_slant_range_min'] * 2 / c0 - 1 / TAC) * fs)

    print('Setting up wavemodel...')
    # Get the model, experiment, logger set up
    if config['exp_params']['model_type'] == 'InfoVAE':
        decoder = InfoVAE(**config['model_params'])
    elif config['exp_params']['model_type'] == 'WAE_MMD':
        decoder = WAE_MMD(fft_len=config['settings']['fft_len'], **config['model_params'])
    else:
        decoder = BetaVAE(**config['model_params'])
    print('Setting up model...')
    try:
        decoder.load_state_dict(torch.load('./model/inference_model.state'))
    except RuntimeError:
        print('Model save file does not match current structure. Re-running with new structure.')
        decoder.apply(init_weights)
    decoder.requires_grad = False
    warm_start = False
    if config['wave_exp_params']['warm_start']:
        print('Wavemodel loaded from save state.')
        try:
            with open('./model/current_model_params.pic', 'rb') as f:
                generator_params = pickle.load(f)
            wave_mdl = GeneratorModel(**generator_params)
            wave_mdl.load_state_dict(torch.load(generator_params['state_file']))
            warm_start = True
        except RuntimeError as e:
            print(f'Wavemodel save file does not match current structure. Re-running with new structure.\n{e}')
            wave_mdl = GeneratorModel(fft_sz=fft_len, decoder=decoder,
                                      stft_win_sz=config['settings']['stft_win_sz'],
                                      clutter_latent_size=config['model_params']['latent_dim'],
                                      target_latent_size=config['model_params']['latent_dim'], n_ants=2)
    else:
        print('Initializing new wavemodel...')
        wave_mdl = GeneratorModel(fft_sz=fft_len, decoder=decoder,
                                  stft_win_sz=config['settings']['stft_win_sz'],
                                  clutter_latent_size=config['model_params']['latent_dim'],
                                  target_latent_size=config['model_params']['latent_dim'], n_ants=2)

    # Since these are dependent on apache params, we set them up here instead of in the yaml file
    print('Setting up data generator...')
    config['dataset_params']['max_pulse_length'] = nr
    config['dataset_params']['min_pulse_length'] = 1000

    data = WaveDataModule(latent_dim=config['model_params']['latent_dim'], device=device, fft_sz=fft_len,
                          **config["dataset_params"])
    data.setup()

    print('Setting up experiment...')
    experiment = GeneratorExperiment(wave_mdl, config['wave_exp_params'])

    if warm_start:
        name = 'WaveModel'
        # Find the latest version and append to that
        try:
            mnum = max(int(n.split('_')[-1]) for n in listdir(f"{config['train_params']['log_dir']}/{name}"))
            logger = loggers.TensorBoardLogger(config['train_params']['log_dir'],
                                               name="WaveModel", version=mnum, log_graph=True)
        except ValueError:
            logger = loggers.TensorBoardLogger(config['train_params']['log_dir'],
                                               name="WaveModel", log_graph=True)
    else:
        logger = loggers.TensorBoardLogger(config['train_params']['log_dir'],
                                           name="WaveModel", log_graph=True)
    # logger.experiment.add_graph(trainer, wave_mdl.example_input_array)

    expected_lr = max((config['wave_exp_params']['LR'] *
                   config['wave_exp_params']['scheduler_gamma']**(config['wave_exp_params']['max_epochs'] * config['wave_exp_params']['swa_start'])), 1e-9)
    trainer = Trainer(logger=logger, max_epochs=config['wave_exp_params']['max_epochs'],
                      log_every_n_steps=config['wave_exp_params']['log_epoch'], devices=1, callbacks=
                      [EarlyStopping(monitor='loss', patience=config['wave_exp_params']['patience'],
                                     check_finite=True),
                       StochasticWeightAveraging(swa_lrs=expected_lr, swa_epoch_start=config['wave_exp_params']['swa_start'])])

    print("======= Training =======")
    try:
        trainer.fit(experiment, datamodule=data)
    except KeyboardInterrupt:
        print('Breaking out of training early.')

    if trainer.global_rank == 0:

        # Implement some pruning
        '''prune_layers = ([(m, 'weight') for m in wave_mdl.mixture if isinstance(m, nn.Conv1d)] +
                        [(m, 'weight') for m in wave_mdl.expand_to_ants if isinstance(m, nn.Conv1d)] +
                        [(m, 'weight') for m in wave_mdl.final if isinstance(m, nn.Conv2d)])

        prune.global_unstructured(prune_layers, pruning_method=prune.L1Unstructured, amount=.1)

        for m, n in prune_layers:
            prune.remove(m, 'weight')'''

        with torch.no_grad():
            wave_mdl.to(device)
            wave_mdl.eval()

            cc, tc, cs, ts, _ = next(iter(data.train_dataloader()))
            cc = cc.to(device)
            tc = tc.to(device)
            cs = cs.to(device)
            ts = ts.to(device)

            nn_output = wave_mdl([cc, tc, [nr], torch.tensor([config['settings']['bandwidth']])])
            nn_numpy = nn_output[0, 0, ...].cpu().data.numpy()

            waves = wave_mdl.getWaveform(nn_output=nn_output).cpu().data.numpy()
            print('Loaded waveforms...')

            clutter = cs.cpu().data.numpy()
            clutter = normalize(clutter[:, 0, :] + 1j * clutter[:, 1, :])
            targets = ts.cpu().data.numpy()
            targets = normalize(targets[:, 0, :] + 1j * targets[:, 1, :])
            print('Loaded clutter and target data...')

            # Run some plots for an idea of what's going on
            freqs = np.fft.fftshift(np.fft.fftfreq(fft_len, 1 / fs))
            plt.figure('Waveform PSD')
            plt.plot(freqs, db(np.fft.fftshift(waves[0, 0])))
            plt.plot(freqs, db(np.fft.fftshift(waves[0, 1])))
            plt.plot(freqs, db(np.fft.fftshift(targets[0])), linestyle='--', linewidth=.3)
            plt.plot(freqs, db(np.fft.fftshift(clutter[0])), linestyle=':', linewidth=.3)
            plt.legend(['Waveform 1', 'Waveform 2', 'Target', 'Clutter'])
            plt.ylabel('Relative Power (dB)')
            plt.xlabel('Freq (Hz)')

            # Save the model structure out to a PNG
            # plot_model(mdl, to_file='./mdl_plot.png', show_shapes=True)
            # waveforms = np.fft.fftshift(waveforms, axes=2)
            plt.figure('Autocorrelation')
            linear = np.fft.fft(
                genPulse(np.linspace(0, 1, 10),
                         np.linspace(0, 1, 10), nr, fs, config['settings']['fc'],
                         config['settings']['bandwidth']), fft_len)
            linear = linear / sum(linear * linear.conj())  # Unit energy
            inp_wave = waves[0, 0] * waves[0, 0].conj()
            autocorr1 = np.fft.fftshift(db(np.fft.ifft(upsample(inp_wave))))
            inp_wave = waves[0, 1] * waves[0, 1].conj()
            autocorr2 = np.fft.fftshift(db(np.fft.ifft(upsample(inp_wave))))
            inp_wave = waves[0, 0] * waves[0, 1].conj()
            autocorrcr = np.fft.fftshift(db(np.fft.ifft(upsample(inp_wave))))
            perf_autocorr = np.fft.fftshift(db(np.fft.ifft(upsample(linear * linear.conj()))))
            lags = np.arange(len(autocorr1)) - len(autocorr1) // 2
            plt.plot(lags[len(lags) // 2 - 200:len(lags) // 2 + 200],
                     autocorr1[len(lags) // 2 - 200:len(lags) // 2 + 200] - autocorr1.max())
            plt.plot(lags[len(lags) // 2 - 200:len(lags) // 2 + 200],
                     autocorr2[len(lags) // 2 - 200:len(lags) // 2 + 200] - autocorr2.max())
            plt.plot(lags[len(lags) // 2 - 200:len(lags) // 2 + 200],
                     autocorrcr[len(lags) // 2 - 200:len(lags) // 2 + 200] - autocorr1.max())
            plt.plot(lags[len(lags) // 2 - 200:len(lags) // 2 + 200],
                     perf_autocorr[len(lags) // 2 - 200:len(lags) // 2 + 200] - perf_autocorr.max(),
                     linestyle='--')
            plt.legend(['Waveform 1', 'Waveform 2', 'Cross Correlation', 'Linear Chirp'])
            plt.xlabel('Lag')

            waves = wave_mdl.getWaveform(cc, tc, [nr], torch.tensor([config['settings']['bandwidth']]),
                                         scale=True, custom_fft_sz=8192).cpu().data.numpy()

            plt.figure('Time Series')
            wave1 = waves.copy()
            plot_t = np.arange(nr) / fs
            plt.plot(plot_t, np.fft.ifft(wave1[0, 0]).real[:nr])
            plt.plot(plot_t, np.fft.ifft(wave1[0, 1]).real[:nr])
            plt.legend(['Waveform 1', 'Waveform 2'])
            plt.xlabel('Time')

            wave_t = np.fft.ifft(waves[0, 0])[:nr]
            win = torch.windows.hann(256).data.numpy()
            freq_stft, t_stft, wave_stft = stft(wave_t, return_onesided=False, window=win, fs=2e9)
            plt.figure('Wave STFT')
            plt.pcolormesh(t_stft, np.fft.fftshift(freq_stft), np.fft.fftshift(db(wave_stft), axes=0))
            plt.ylabel('Freq')
            plt.xlabel('Time')
            plt.colorbar()

        if trainer.is_global_zero and config['wave_exp_params']['save_model']:
            try:
                wave_mdl.save('./model')
                print('Model saved to disk.')
            except Exception as e:
                print(f'Model not saved: {e}')