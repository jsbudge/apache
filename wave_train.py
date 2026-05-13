from config import get_config
from utils import upsample, fs, narrow_band, getMatchedFilter, cfar, c0
import numpy as np
from simulib.simulation_functions import genPulse, db
import matplotlib.pyplot as plt
from scipy.signal import stft, convolve
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
    nr = 2669  # int((config['perf_params']['vehicle_slant_range_min'] * 2 / c0 - 1 / TAC) * fs)
    # Since these are dependent on apache params, we set them up here instead of in the yaml file
    print('Setting up data generator...')
    config.dataset_params['max_pulse_length'] = 5000
    config.dataset_params['min_pulse_length'] = 1000

    data = WaveDataModule(device=device, **config.dataset_params)
    data.setup()

    print('Initializing wavemodel...')
    if config.warm_start:
        wave_mdl = GeneratorModel.load_from_checkpoint(f'{config.weights_path}/{config.model_name}.ckpt', config=config, strict=False)
    else:
        wave_mdl = GeneratorModel(config=config)
    logger = loggers.TensorBoardLogger(config.log_dir,
                                       name=config.model_name, log_graph=False)
    if config.distributed:
        trainer = Trainer(logger=logger, max_epochs=config.max_epochs, num_sanity_val_steps=0, default_root_dir=config.weights_path,
                          log_every_n_steps=config.log_epoch, check_val_every_n_epoch=1000, devices=[0, 1], strategy='ddp', callbacks=
                          [EarlyStopping(monitor='target_loss', patience=config.patience, check_finite=True),
                           ModelCheckpoint(monitor='loss_epoch')])
    else:
        trainer = Trainer(logger=logger, max_epochs=config.max_epochs, num_sanity_val_steps=0,
                          default_root_dir=config.weights_path, check_val_every_n_epoch=1000,
                          log_every_n_steps=config.log_epoch, devices=[0], callbacks=
                          [EarlyStopping(monitor='target_loss', patience=config.patience, check_finite=True),
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
            filt_stack = []
            wave_mdl.to(device)
            wave_mdl.eval()
            clutter_profile = []
            target_profile = []
            just_targets = []
            target_index = []
            roc_locs = []
            data_iter = iter(data.train_dataloader())
            scalings = data.train_dataloader().dataset.scaling
            nsam = data.train_dataloader().dataset.samples[0]

            for _ in range(min(len(data.train_dataloader()), 25)):
                cct, sot, tct, tidx, _, _, _, roc = next(data_iter)
                clutter_profile.append(cct)
                target_profile.append(tct)
                just_targets.append(sot)
                target_index.append(tidx)
                roc_locs.append(roc)

            tbandwidth = .35

            taytay = np.zeros(fft_len, dtype=np.complex128)
            taytay_len = int(tbandwidth * fft_len) if int(tbandwidth * fft_len) % 2 == 0 else int(
                tbandwidth * fft_len) + 1
            taytay[:taytay_len // 2] = taylor(taytay_len)[-taytay_len // 2:]
            taytay[-taytay_len // 2:] = taylor(taytay_len)[:taytay_len // 2]

            for ts, cc in zip(just_targets, clutter_profile):
                # cc = cct.to(device) if cc is None else cc
                ts = ts.to(device)
                cc = cc.to(device)
                # ts = tst.to(device)
                plength = torch.tensor([nr]).to(device)
                bandwidth = torch.tensor([tbandwidth]).to(device)
                # plength = plength.to(device)
                # bandwidth = bandwidth.to(device)

                nn_output = wave_mdl(cc, ts, plength, bandwidth)
                # nn_numpy = nn_output[0, 0, ...].cpu().data.numpy()

                wave_stack.append(wave_mdl.getWaveform(nn_output=nn_output).cpu().data.numpy())
            waves = np.concatenate(wave_stack) # / np.sum(waves * waves.conj(), axis=2)[..., None]
            filts = np.concatenate(wave_stack).conj()# * taytay
            # waves = save_waves
            # waves = np.fft.fft(np.fft.ifft(waves, axis=2)[:, :, :nr], fft_len, axis=2)
            print('Loaded waveforms...')

            clutter = np.concatenate([cc.cpu().data.numpy() for cc in clutter_profile])
            clutter = (clutter[:, 64, 0] + 1j * clutter[:, 64, 1]) * scalings[0]
            targets = [ts.cpu().data.numpy() for ts in target_profile]
            targets = np.array([(t[0, 64, 0, :] + 1j * t[0, 64, 1, :]) * scalings[0] for t in targets])
            just_t = just_targets[0][0, 0, 0].data.cpu().numpy() + 1j * just_targets[0][0, 0, 1].data.cpu().numpy()
            roc_truth = np.array([t[0, 64, :].data.cpu().numpy() for t in roc_locs])

            linear = np.fft.fft(genPulse(np.linspace(0, 1, 10),
                         np.linspace(0, 1, 10), nr, fs, config.fc,
                         bandwidth[0].cpu().data.numpy() * fs), fft_len)
            linear = linear / np.sqrt(sum(linear * linear.conj()))  # Unit energy
            # linear = linear / sum(linear * linear.conj())  # Unit energy
            print('Loaded clutter and target data...')

            # Run some plots for an idea of what's going on
            freqs = np.fft.fftshift(np.fft.fftfreq(fft_len, 1 / fs))
            clutter_spectra = np.fft.fftshift(clutter[0] * linear.conj())
            target_spectra = np.fft.fftshift(targets[0] * linear.conj())
            plt.figure('Waveform PSD')
            plt.plot(freqs, db(clutter_spectra))
            plt.plot(freqs, db(target_spectra))
            plt.plot(freqs, db(np.fft.fftshift(waves[-1, 0])))
            if wave_mdl.n_ants > 1:
                plt.plot(freqs, db(np.fft.fftshift(waves[-1, 1])))
            if wave_mdl.n_ants > 1:
                plt.legend(['Waveform 1', 'Waveform 2'])
            else:
                plt.legend(['Clutter', 'Target', 'Waveform'])
            plt.ylabel('Relative Power (dB)')
            plt.xlabel('Freq (Hz)')

            mfiltered_linear = linear * linear.conj() * taytay

            '''plt.figure('Target-Clutter vs. Linear')
            for tnum in range(waves.shape[0]):
                mfiltered_wave0 = waves[tnum, 0] / (filts[tnum, 0] + 1e-12)
                mfiltered_wave1 = waves[tnum, 1] / (filts[tnum, 1] + 1e-12) if wave_mdl.n_ants > 1 else 0
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

            plt.legend(['Range Profile'] + [f'NN_{n}' for n in range(wave_mdl.n_ants)] + ['Linear'])
            plt.xlabel('Lag')
            plt.ylabel('Power (dB)')'''

            plt.figure('Target_Clutter vs. Linear')
            tnum = 7
            cv_sz = 25

            linear_corr = np.convolve(db(np.fft.ifft(targets * linear.conj() * taytay, axis=1)[tnum, :nsam]), np.ones(cv_sz) / cv_sz, mode='same')

            if wave_mdl.n_ants > 1:
                clutter_corr = np.convolve(db(np.fft.ifft(targets[0] * filts[tnum, 0] + targets[0] * filts[tnum, 1])), np.ones(cv_sz) / cv_sz, mode='same')
                target0_corr = np.fft.ifft(targets[0] * filts[tnum, 0])[:nsam]
                target1_corr = np.fft.ifft(targets[0] * filts[tnum, 1])[:nsam]
            else:
                target0_corr = np.convolve(db(np.fft.ifft(targets * filts[tnum, 0], axis=1)[tnum, :nsam]), np.ones(cv_sz) / cv_sz, mode='same')

            zoom_area = np.arange(nsam)
            zoom_sz = len(zoom_area)
            # target_time = np.convolve(db(np.fft.ifft(targets * linear.conj() * taytay, axis=1)[tnum, zoom_area]), np.ones(15), mode='same')
            plt.title(f'Target {tnum}')
            # plt.scatter(zoom_area, (target_time[zoom_area] - target_time[zoom_area].max()), color='black')

            plt.plot(zoom_area * c0 / (2 * fs), (target0_corr[zoom_area] - target0_corr[zoom_area].max()))
            if wave_mdl.n_ants > 1:
                plt.plot(zoom_area * c0 / (2 * fs), (db(clutter_corr)[zoom_area] - db(clutter_corr)[zoom_area].max()))
                plt.plot(zoom_area * c0 / (2 * fs), (db(target1_corr)[zoom_area] - db(target1_corr)[zoom_area].max()))
            plt.plot(zoom_area * c0 / (2 * fs), (linear_corr[zoom_area] - linear_corr[zoom_area].max()))
            plt.vlines(target_index[tnum][0] * c0 / (2 * fs), (linear_corr[zoom_area] - linear_corr[zoom_area].max()).min(), (linear_corr[zoom_area] - linear_corr[zoom_area].max()).max(), color='black')

            plt.legend([f'NN_{n}' for n in range(wave_mdl.n_ants)] + ['Linear'])
            plt.xlabel('Distance from Near Range')
            plt.ylabel('Power (dB)')


            tnum = 7
            # wave0 = waves[tnum, 0]

            '''correct = roc_truth[tnum]
            nlevels = [60]
            nthresh = np.linspace(-1, 3, 50)
            plt.figure('ROC curves')
            win = (np.arange(101) - 50.)**2
            win /= sum(win)

            for idx, nlevel in enumerate(nlevels):
                # plt.figure(f'Target_Clutter vs. Linear - {nlevel}')
                target_true_positive = [0 for nt in nthresh]
                target_false_positive = [0 for nt in nthresh]
                target_true_negative = [0 for nt in nthresh]
                target_false_negative = [0 for nt in nthresh]
                linear_true_positive = [0 for nt in nthresh]
                linear_false_positive = [0 for nt in nthresh]
                linear_true_negative = [0 for nt in nthresh]
                linear_false_negative = [0 for nt in nthresh]
                for _ in range(50):
                    t0_energy = abs(np.sqrt(sum(targets * targets.conj())))
                    t0_noise = (np.random.randn(*targets.shape) + 1j * np.random.randn(*targets.shape)) * t0_energy / 10**(nlevel / 20)
                    tgets = targets# + t0_noise
                    sans = clutter# + t0_noise
                    t0_nn = convolve(db(np.fft.ifft(tgets * filts[tnum, 0], axis=1)[:, :nsam]), np.ones((1, cv_sz)) / cv_sz, mode='same')
                    t0_l = convolve(db(np.fft.ifft(tgets * linear.conj() * taytay, axis=1)[:, :nsam]), np.ones((1, cv_sz)) / cv_sz, mode='same')
                    sans_nn = convolve(db(np.fft.ifft(sans * filts[tnum, 0], axis=1)[:, :nsam]), np.ones((1, cv_sz)) / cv_sz, mode='same')
                    sans_l = convolve(db(np.fft.ifft(sans * linear.conj() * taytay, axis=1)[:, :nsam]), np.ones((1, cv_sz)) / cv_sz, mode='same')
                    target_cfars = cfar(t0_nn, nthresh)
                    linear_cfars = cfar(t0_l, nthresh)
                    target_sans_cfars = cfar(sans_nn, nthresh)
                    linear_sans_cfars = cfar(sans_l, nthresh)
                    detections = ([np.sum(np.diff(tc + 0., axis=1) == 1) for tc in target_cfars],
                                  [np.sum(np.diff(tc + 0., axis=1) == 1) for tc in linear_cfars])
                    truth_detections = ([sum(np.any(np.logical_and(tc == 1, correct == 1), axis=1)) for tc in target_cfars],
                                  [sum(np.any(np.logical_and(tc == 1, correct == 1), axis=1)) for tc in linear_cfars])
                    sans_detections = ([np.sum(np.diff(tc + 0., axis=1) == 1) for tc in target_sans_cfars],
                                  [np.sum(np.diff(tc + 0., axis=1) == 1) for tc in linear_sans_cfars])
                    target_true_positive = [tp + td for tp, td in zip(target_true_positive, truth_detections[0])]
                    target_false_positive = [fp + d - td + sd for fp, d, td, sd in
                                             zip(target_false_positive, detections[0], truth_detections[0], sans_detections[0])]
                    linear_true_positive = [tp + td for tp, td in zip(linear_true_positive, truth_detections[1])]
                    linear_false_positive = [fp + d - td + sd for fp, d, td, sd in
                                             zip(linear_false_positive, detections[1], truth_detections[1],
                                                 sans_detections[1])]
                    target_true_negative = [tn + 25 for tn, sd in zip(target_true_negative, sans_detections[0])]
                    target_false_negative = [tn + (d == 0) + 0. for tn, d in zip(target_false_negative, detections[0])]
                    linear_true_negative = [tn + 25 for tn, sd in zip(linear_true_negative, sans_detections[1])]
                    linear_false_negative = [tn + (d == 0) + 0. for tn, d in zip(linear_false_negative, detections[1])]

                target_true = [tp / (tp + fn) for tp, fn in zip(target_true_positive, target_false_negative)]
                target_false = [tfp / (tfp + tn) for tfp, tn in zip(target_false_positive, target_true_negative)]
                linear_true = [tp / (tp + fn) for tp, fn in zip(linear_true_positive, linear_false_negative)]
                linear_false = [tfp / (tfp + tn) for tfp, tn in zip(linear_false_positive, linear_true_negative)]

                plt.subplot(1, 1, idx + 1)
                plt.title(f'SNR {nlevel:.2f} dB')
                plt.plot([1.] + target_false, [1.] + target_true)
                plt.plot([1.] + linear_false, [1.] + linear_true)
                plt.plot([0, 1], [0, 1], 'k--')
            plt.legend(['NN', 'Linear', 'Chance'])'''

            # Save the model structure out to a PNG
            # plot_model(mdl, to_file='./mdl_plot.png', show_shapes=True)
            # waveforms = np.fft.fftshift(waveforms, axes=2)
            plt.figure('Autocorrelation')
            inp_wave = waves[tnum, 0] * filts[tnum, 0]
            autocorr1 = np.fft.fftshift(db(np.fft.ifft(upsample(inp_wave))))
            if wave_mdl.n_ants > 1:
                inp_wave = waves[tnum, 1] * filts[tnum, 1]
                autocorr2 = np.fft.fftshift(db(np.fft.ifft(upsample(inp_wave))))
                inp_wave = waves[tnum, 0] * filts[tnum, 1]
                autocorrcr = np.fft.fftshift(db(np.fft.ifft(upsample(inp_wave))))
            perf_autocorr = np.fft.fftshift(db(np.fft.ifft(upsample(mfiltered_linear))))
            lags = np.arange(len(autocorr1)) - len(autocorr1) // 2
            plt.plot(lags[len(lags) // 2 - 200:len(lags) // 2 + 200],
                     autocorr1[len(lags) // 2 - 200:len(lags) // 2 + 200] - autocorr1.max())
            if wave_mdl.n_ants > 1:
                plt.plot(lags[len(lags) // 2 - 200:len(lags) // 2 + 200],
                         autocorr2[len(lags) // 2 - 200:len(lags) // 2 + 200] - autocorr2.max())
                plt.plot(lags[len(lags) // 2 - 200:len(lags) // 2 + 200],
                         autocorrcr[len(lags) // 2 - 200:len(lags) // 2 + 200] - autocorrcr.max())
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
            twave = np.fft.ifft(wave1[0, 0]).real[:nsam]
            plot_t = np.arange(nsam) / fs
            plt.plot(plot_t, twave)
            if wave_mdl.n_ants > 1:
                plt.plot(plot_t, np.fft.ifft(wave1[0, 1]).real[:nsam])
                plt.legend(['Waveform 1', 'Waveform 2'])
            plt.vlines([nr / fs], -abs(twave.max()), abs(twave.max()), color='black')
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
            tprof = (tprof[:, 0] + 1j * tprof[:, 1]) * scalings[0]
            linear_block = db(np.fft.fft(np.fft.ifft(tprof * linear.conj() * taytay, axis=-1)[:, :nsam], axis=0)).T

            # Rerun wavemodel with each successive pulse
            wave_block = db(np.fft.fft(np.fft.ifft(tprof * filts[tnum, 0], axis=-1)[:, :nsam], axis=0)).T

            plt.figure('Doppler Profiles')
            plt.subplot(1, 2, 1)
            plt.title('Linear')
            plt.imshow(linear_block - linear_block.max())
            plt.hlines(target_index[tnum].cpu().data.numpy(), -.5, linear_block.shape[1] - .5, linestyle=':')
            plt.clim([-80, 0])
            plt.axis('tight')
            plt.subplot(1, 2, 2)
            plt.title('Wave')
            plt.imshow(wave_block - wave_block.max())
            plt.hlines(target_index[tnum].cpu().data.numpy(), -.5, linear_block.shape[1] - .5, linestyle=':')
            plt.clim([-80, 0])
            plt.axis('tight')
            plt.colorbar()

            waf, tau, theta = narrow_band(np.fft.ifft(waves[0, 0])[:nsam], np.arange(512) - 256)

            plt.figure('Ambiguity Function')
            plt.imshow(db(waf), extent=(tau[0], tau[-1], theta[0], theta[-1]))
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