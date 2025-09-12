from config import get_config
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, ModelCheckpoint
from dataloaders import WaveDataModule
from waveform_model import GeneratorModel
import matplotlib as mplib
mplib.use('TkAgg')


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

    # seed_everything(np.random.randint(1, 2048), workers=True)
    seed_everything(107, workers=True)

    config = get_config('wave_exp', './vae_config.yaml')
    config.gradient_flow = True

    fft_len = config.fft_len
    nr = 5000  # int((config['perf_params']['vehicle_slant_range_min'] * 2 / c0 - 1 / TAC) * fs)
    # Since these are dependent on apache params, we set them up here instead of in the yaml file
    print('Setting up data generator...')
    config.dataset_params['max_pulse_length'] = nr
    config.dataset_params['min_pulse_length'] = 1000

    data = WaveDataModule(device=device, **config.dataset_params)
    data.setup()

    print('Initializing wavemodel...')
    wave_mdl = GeneratorModel(config=config)

    expected_lr = max((config.lr * config.scheduler_gamma ** (config.max_epochs * config.swa_start)), 1e-9)
    trainer = Trainer(logger=None, num_sanity_val_steps=0, max_steps=1,
                      default_root_dir=config.weights_path, check_val_every_n_epoch=1000,
                      log_every_n_steps=config.log_epoch, devices=[1], callbacks=
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





