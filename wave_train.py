from config import get_config
from utils import upsample, normalize, fs, narrow_band, getMatchedFilter
import numpy as np
from simulib.simulation_functions import genPulse, db, findPowerOf2
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.signal.windows import taylor
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, ModelCheckpoint
from dataloaders import WaveDataModule
from waveform_model import GeneratorModel


def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))



if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    torch.autograd.set_detect_anomaly(True)
    force_cudnn_initialization()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.cuda.empty_cache()

    seed_everything(np.random.randint(1, 2048), workers=True)
    # seed_everything(107, workers=True)

    config = get_config('wave_exp', './vae_config.yaml')

    fft_len = config.fft_len
    nr = 5000  # int((config['perf_params']['vehicle_slant_range_min'] * 2 / c0 - 1 / TAC) * fs)
    # Since these are dependent on apache params, we set them up here instead of in the yaml file
    print('Setting up data generator...')
    config.dataset_params['max_pulse_length'] = nr
    config.dataset_params['min_pulse_length'] = 1000

    data = WaveDataModule(device=device, **config.dataset_params)
    data.setup()

    print('Initializing wavemodel...')
    if config.warm_start:
        wave_mdl = GeneratorModel.load_from_checkpoint(f'{config.weights_path}/{config.model_name}.ckpt', config=config, strict=False)
    else:
        wave_mdl = GeneratorModel(config=config)
    logger = loggers.TensorBoardLogger(config.log_dir,
                                       name=config.model_name, log_graph=True)
    expected_lr = max((config.lr * config.scheduler_gamma ** (config.max_epochs * config.swa_start)), 1e-9)
    if config.distributed:
        trainer = Trainer(logger=logger, max_epochs=config.max_epochs, num_sanity_val_steps=0, default_root_dir=config.weights_path,
                          log_every_n_steps=config.log_epoch, check_val_every_n_epoch=1000, devices=[0, 1], strategy='ddp', callbacks=
                          [EarlyStopping(monitor='target_loss', patience=config.patience, check_finite=True),
                           StochasticWeightAveraging(swa_lrs=expected_lr, swa_epoch_start=config.swa_start),
                           ModelCheckpoint(monitor='loss_epoch')])
    else:
        trainer = Trainer(logger=logger, max_epochs=config.max_epochs, num_sanity_val_steps=0,
                          default_root_dir=config.weights_path, check_val_every_n_epoch=1000,
                          log_every_n_steps=config.log_epoch, devices=[0], callbacks=
                          [EarlyStopping(monitor='target_loss', patience=config.patience, check_finite=True),
                           StochasticWeightAveraging(swa_lrs=expected_lr, swa_epoch_start=config.swa_start),
                           ModelCheckpoint(monitor='loss_epoch')])
    print("======= Training =======")
    try:
        trainer.fit(wave_mdl, datamodule=data)
    except KeyboardInterrupt:
        if trainer.is_global_zero:
            print('Training interrupted.')
        else:
            print('adios!')
            exit(0)

    if trainer.global_rank == 0:
        if config.save_model:
            trainer.save_checkpoint(f'{config.weights_path}/{config.model_name}.ckpt')
            print('Checkpoint saved.')

        with torch.no_grad():
            wave_stack = []
            wave_mdl.to(device)
            wave_mdl.eval()
            clutter_profile = []
            target_profile = []
            target_index = []
            data_iter = iter(data.train_dataloader())
            scalings = data.train_dataloader().dataset.scaling
            nsam = data.train_dataloader().dataset.samples[0]

            for _ in range(min(len(data.train_dataloader()), 25)):
                cct, tct, tidx, _, _ = next(data_iter)
                clutter_profile.append(cct)
                target_profile.append(tct)
                target_index.append(tidx)

            tbandwidth = .3806

            for ts, cc in zip(target_profile, clutter_profile):
                # cc = cct.to(device) if cc is None else cc
                ts = ts.to(device)
                cc = cc.to(device)
                # ts = tst.to(device)
                plength = torch.tensor([3178]).to(device)
                bandwidth = torch.tensor([tbandwidth]).to(device)
                # plength = plength.to(device)
                # bandwidth = bandwidth.to(device)

                nn_output = wave_mdl(cc, ts, plength, bandwidth)
                # nn_numpy = nn_output[0, 0, ...].cpu().data.numpy()

                wave_stack.append(wave_mdl.getWaveform(nn_output=nn_output).cpu().data.numpy())
            waves = np.concatenate(wave_stack) # / np.sum(waves * waves.conj(), axis=2)[..., None]
            # waves = save_waves
            # waves = np.fft.fft(np.fft.ifft(waves, axis=2)[:, :, :nr], fft_len, axis=2)
            print('Loaded waveforms...')

            clutter = [cc.cpu().data.numpy() for cc in clutter_profile]
            clutter = np.array(
                [(t[0, -1, 0, :] + 1j * t[0, -1, 1, :]) * scalings for t in clutter])
            targets = [ts.cpu().data.numpy() for ts in target_profile]
            targets = np.array([(t[0, -1, 0, :] + 1j * t[0, -1, 1, :]) * scalings for t in targets])

            linear = np.fft.fft(genPulse(np.linspace(0, 1, 10),
                         np.linspace(0, 1, 10), nr, fs, config.fc,
                         bandwidth[0].cpu().data.numpy() * fs), fft_len)
            linear = linear / np.sqrt(sum(linear * linear.conj()))  # Unit energy
            # linear = linear / sum(linear * linear.conj())  # Unit energy
            print('Loaded clutter and target data...')

            # Run some plots for an idea of what's going on
            freqs = np.fft.fftshift(np.fft.fftfreq(fft_len, 1 / fs))
            plt.figure('Waveform PSD')
            plt.plot(freqs, db(np.fft.fftshift(waves[-1, 0])))
            if wave_mdl.n_ants > 1:
                plt.plot(freqs, db(np.fft.fftshift(waves[-1, 1])))
            if wave_mdl.n_ants > 1:
                plt.legend(['Waveform 1', 'Waveform 2'])
            else:
                plt.legend(['Waveform'])
            plt.ylabel('Relative Power (dB)')
            plt.xlabel('Freq (Hz)')

            taytay = np.zeros(fft_len, dtype=np.complex128)
            taytay_len = int(tbandwidth * fft_len) if int(tbandwidth * fft_len) % 2 == 0 else int(tbandwidth * fft_len) + 1
            taytay[:taytay_len // 2] = taylor(taytay_len)[-taytay_len // 2:]
            taytay[-taytay_len // 2:] = taylor(taytay_len)[:taytay_len // 2]

            mfiltered_linear = linear * linear.conj()# * taytay
            linear_corr = np.fft.ifft(targets * mfiltered_linear, axis=1)[:nsam]

            plt.figure('Target-Clutter vs. Linear')
            for tnum in range(waves.shape[0]):
                mfiltered_wave0 = waves[tnum, 0] * waves[tnum, 0].conj() * taytay
                mfiltered_wave1 = waves[tnum, 1] * waves[tnum, 1].conj() * taytay if wave_mdl.n_ants > 1 else 0
                if wave_mdl.n_ants > 1:
                    clutter_corr = np.fft.ifft(targets[0] * mfiltered_wave0 + targets[0] * mfiltered_wave1)[:nsam]
                    target0_corr = np.fft.ifft(targets[0] * mfiltered_wave0)[:nsam]
                    target1_corr = np.fft.ifft(targets[0] * mfiltered_wave1)[:nsam]
                else:
                    target0_corr = np.fft.ifft(targets[tnum] * mfiltered_wave0)[:nsam]

                plt.subplot(5, 5, tnum + 1)
                zoom_area = np.arange(nsam)
                zoom_sz = len(zoom_area)
                target_time = np.fft.ifft(targets[tnum])[:nsam]
                plt.title(f'Target {tnum}')
                plt.scatter(zoom_area, (db(target_time)[zoom_area] - db(target_time)[zoom_area].max()), color='black')

                plt.plot(zoom_area, (db(target0_corr)[zoom_area] - db(target0_corr)[zoom_area].max()))
                if wave_mdl.n_ants > 1:
                    plt.plot(zoom_area, (db(clutter_corr)[zoom_area] - db(clutter_corr)[zoom_area].max()))
                    plt.plot(zoom_area, (db(target1_corr)[zoom_area] - db(target1_corr)[zoom_area].max()))
                plt.plot(zoom_area, (db(linear_corr[tnum])[zoom_area] - db(linear_corr[tnum])[zoom_area].max()))
                plt.vlines(target_index[tnum][0], -50, 10, color='black')

            plt.legend(['Range Profile', 'NN', 'Linear'])
            plt.xlabel('Lag')
            plt.ylabel('Power (dB)')

            plt.figure('Target_Clutter vs. Linear')
            tnum = 7
            mfiltered_wave0 = waves[tnum, 0] * waves[tnum, 0].conj() * taytay
            mfiltered_wave1 = waves[tnum, 1] * waves[tnum, 1].conj() * taytay if wave_mdl.n_ants > 1 else 0
            if wave_mdl.n_ants > 1:
                clutter_corr = np.fft.ifft(targets[0] * mfiltered_wave0 + targets[0] * mfiltered_wave1)[:nsam]
                target0_corr = np.fft.ifft(targets[0] * mfiltered_wave0)[:nsam]
                target1_corr = np.fft.ifft(targets[0] * mfiltered_wave1)[:nsam]
            else:
                target0_corr = np.fft.ifft(targets * mfiltered_wave0, axis=1)[tnum, :nsam]

            zoom_area = np.arange(nsam)
            zoom_sz = len(zoom_area)
            target_time = np.fft.ifft(targets, axis=1)[tnum, :nsam]
            plt.title(f'Target {tnum}')
            plt.scatter(zoom_area, (db(target_time)[zoom_area] - db(target_time)[zoom_area].max()), color='black')

            plt.plot(zoom_area, (db(target0_corr)[zoom_area] - db(target0_corr)[zoom_area].max()))
            if wave_mdl.n_ants > 1:
                plt.plot(zoom_area, (db(clutter_corr)[zoom_area] - db(clutter_corr)[zoom_area].max()))
                plt.plot(zoom_area, (db(target1_corr)[zoom_area] - db(target1_corr)[zoom_area].max()))
            plt.plot(zoom_area, (db(linear_corr[tnum])[zoom_area] - db(linear_corr[tnum])[zoom_area].max()))
            plt.vlines(target_index[tnum][0], -50, 10, color='black')

            plt.legend(['Range Profile', 'NN', 'Linear'])
            plt.xlabel('Lag')
            plt.ylabel('Power (dB)')

            plt.figure('Target-Clutter Correlations')
            comb_corr = np.fft.ifft(targets[tnum])[:nsam]
            truth_corr = np.fft.ifft(clutter[tnum])[:nsam]
            plt.plot(db(comb_corr))
            plt.plot(db(truth_corr))
            plt.legend(['Target+Clutter', 'Clutter'])
            plt.xlabel('Lag')
            plt.ylabel('Power (dB)')

            # Save the model structure out to a PNG
            # plot_model(mdl, to_file='./mdl_plot.png', show_shapes=True)
            # waveforms = np.fft.fftshift(waveforms, axes=2)
            plt.figure('Autocorrelation')
            inp_wave = mfiltered_wave0
            autocorr1 = np.fft.fftshift(db(np.fft.ifft(upsample(inp_wave))))
            if wave_mdl.n_ants > 1:
                inp_wave = mfiltered_wave1
                autocorr2 = np.fft.fftshift(db(np.fft.ifft(upsample(inp_wave))))
                inp_wave = waves[0, 0] * getMatchedFilter(np.fft.ifft(waves[0, 1]), tbandwidth * fs, fs, config.fc, fft_len)
                autocorrcr = np.fft.fftshift(db(np.fft.ifft(upsample(inp_wave))))
            perf_autocorr = np.fft.fftshift(db(np.fft.ifft(upsample(mfiltered_linear))))
            lags = np.arange(len(autocorr1)) - len(autocorr1) // 2
            plt.plot(lags[len(lags) // 2 - 200:len(lags) // 2 + 200],
                     autocorr1[len(lags) // 2 - 200:len(lags) // 2 + 200] - autocorr1.max())
            if wave_mdl.n_ants > 1:
                plt.plot(lags[len(lags) // 2 - 200:len(lags) // 2 + 200],
                         autocorr2[len(lags) // 2 - 200:len(lags) // 2 + 200] - autocorr2.max())
                plt.plot(lags[len(lags) // 2 - 200:len(lags) // 2 + 200],
                         autocorrcr[len(lags) // 2 - 200:len(lags) // 2 + 200] - autocorr1.max())
            plt.plot(lags[len(lags) // 2 - 200:len(lags) // 2 + 200],
                     perf_autocorr[len(lags) // 2 - 200:len(lags) // 2 + 200] - perf_autocorr.max(),
                     linestyle='--')
            if wave_mdl.n_ants > 1:
                plt.legend(['Waveform 1', 'Waveform 2', 'Cross Correlation', 'Linear Chirp'])
            else:
                plt.legend(['Waveform', 'Linear Chirp'])
            plt.xlabel('Lag')

            plt.figure('Time Series')
            wave1 = waves.copy()
            plot_t = np.arange(nr) / fs
            plt.plot(plot_t, np.fft.ifft(wave1[0, 0]).real[:nr])
            if wave_mdl.n_ants > 1:
                plt.plot(plot_t, np.fft.ifft(wave1[0, 1]).real[:nr])
                plt.legend(['Waveform 1', 'Waveform 2'])
            plt.xlabel('Time')

            plt.figure('Waveform Differences')
            for w in waves:
                plt.plot(np.fft.fftshift(db(w[0])))

            wave_t = np.fft.ifft(waves[0, 0])[:nr]
            win = torch.windows.hann(256).data.numpy()
            freq_stft, t_stft, wave_stft = stft(wave_t, return_onesided=False, window=win, fs=2e9)
            plt.figure('Wave STFT')
            plt.pcolormesh(t_stft, np.fft.fftshift(freq_stft), np.fft.fftshift(db(wave_stft), axes=0))
            plt.ylabel('Freq')
            plt.xlabel('Time')
            plt.colorbar()

            tprof = target_profile[tnum][0].cpu().data.numpy()
            tprof = (tprof[:, 0] + 1j * tprof[:, 1]) * scalings
            linear_block = np.fft.fft(np.fft.ifft(mfiltered_linear * tprof, axis=-1)[:, :nsam], axis=0)

            # Rerun wavemodel with each successive pulse
            wave_stack = []
            for n in range(1, linear_block.shape[0]):
                nn_output = wave_mdl(clutter_profile[tnum][:n].to(device), target_profile[tnum][:n].to(device), plength, bandwidth)
                # nn_numpy = nn_output[0, 0, ...].cpu().data.numpy()
                wave_stack.append(wave_mdl.getWaveform(nn_output=nn_output).cpu().data.numpy())
            waves = np.concatenate(wave_stack)
            wave_mfilt = (waves * waves.conj() * taytay)[:, 0]
            wave_block = np.fft.fft(np.fft.ifft(wave_mfilt * tprof[1:], axis=-1)[:, :nsam], linear_block.shape[0], axis=0)

            plt.figure('Doppler Profiles')
            plt.subplot(1, 2, 1)
            plt.title('Linear')
            plt.imshow(db(linear_block).T)
            plt.hlines(target_index[tnum].cpu().data.numpy(), -.5, linear_block.shape[0] - .5, linestyle=':')
            plt.axis('tight')
            plt.subplot(1, 2, 2)
            plt.title('Wave')
            plt.imshow(db(wave_block).T)
            plt.hlines(target_index[tnum].cpu().data.numpy(), -.5, linear_block.shape[0] - .5, linestyle=':')
            plt.axis('tight')
            plt.colorbar()



        waf, tau, theta = narrow_band(np.fft.ifft(waves[0, 0]), np.arange(512) - 256)

        plt.figure('Ambiguity Function')
        plt.imshow(db(waf[4096 - 256:4096 + 256, :]))
        # plt.clim([-100, -76])
        plt.axis('tight')
        plt.show()

'''wave = np.fft.fft(np.fft.ifft(waves[0, 0]), 32768)
rp_back = np.load('/home/jeff/repo/simulib/scripts/single_rp_back.npy')
rp_old_mf = np.load('/home/jeff/repo/simulib/scripts/single_mf_pulse_back.npy')
rp_fft = np.fft.fft(rp_back[0, 0], 32768)
rp_imf = rp_fft * wave * wave.conj()
rp_mf = np.zeros(32768 * 8, dtype=np.complex128)
rp_mf[:16384] = rp_imf[:16384]
rp_mf[-16384:] = rp_imf[-16384:]
rp_mf = np.fft.ifft(rp_mf)[:rp_old_mf.shape[1]]
plt.figure()
plt.plot(db(rp_mf))
plt.plot(db(rp_old_mf[0]))'''

'''culprits = []
losses = []
resultses = []
for step, dt in enumerate(tqdm(data.train_dataloader())):
    if step in [399, 400, 401, 402, 1699, 1700, 1701, 1702, 3199, 3200, 3201]:
        bd = [b.to(wave_mdl.device) for b in dt]
        results = wave_mdl(bd[0], bd[2], bd[3], bd[4])
        resultses.append(wave_mdl(bd[0], bd[2], bd[3], bd[4]).to('cpu'))
        losses.append(wave_mdl.loss_function(results, *bd)['target_loss'].to('cpu'))
        culprits.append(dt)'''

# .5 * (-2 * 65 + np.trace(np.linalg.pinv(eb) @ ea) + np.trace(np.linalg.pinv(ea) @ eb) + (mub - mua).dot(np.linalg.pinv(ea) + np.linalg.pinv(eb)).dot(mub-mua))