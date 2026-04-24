import torch
from neuralop import TFNO1d
from torch import nn, Tensor
from pytorch_lightning import LightningModule
from models import ClutterTransformer, FlatModule
from torch.nn import functional as nn_func
from config import Config
from schedulers import CosineWarmupScheduler
from layers import PulseLength, WindowGenerate, RelativeMultiHeadAttention, Block1d, SwiGLU, FourierFeatureTrain, LKA1d
import numpy as np
from utils import normalize, _xavier_init, nonlinearities, plot_grad_flow, rbf_linear, l_norm

EPS = 1e-9


def _unflatten_to_state_dict(flat_w, shapes):
    state_dict = {}
    counter = 0
    for shape in shapes:
        name, tsize, tnum = shape
        param = flat_w[counter: counter + tnum].reshape(tsize)
        state_dict[name] = torch.nn.Parameter(param)
        counter += tnum
    assert counter == len(flat_w), "counter must reach the end of weight vector"
    return state_dict


class GeneratorModel(FlatModule):
    def __init__(self,
                 config: Config,
                 **kwargs,
                 ) -> None:
        super(GeneratorModel, self).__init__(config)
        self.n_ants = config.n_ants
        self.fft_len = config.fft_len
        self.target_latent_size = config.target_latent_size
        self.fs = config.fs
        self.fc = config.fc
        self.flowthrough_channels = config.flowthrough_channels
        self.wave_decoder_channels = config.wave_decoder_channels
        self.waveform_channels = [config.waveform_conv_channels, config.waveform_mid_channels, config.waveform_small_channels]
        self.target_channels = [config.target_conv_channels, config.target_mid_channels, config.target_small_channels]
        self.automatic_optimization = False
        self.conv_kernel_size = config.conv_kernel_size
        self.mid_kernel_size = config.mid_kernel_size
        self.small_kernel_size = config.small_kernel_size
        self.nonlinearity = config.nonlinearity
        nlin = nonlinearities[self.nonlinearity]

        '''TRANSFORMER'''
        self.predict_encoder = ClutterTransformer.load_from_checkpoint(
            f'{config.transformer_weights_path}/{config.transformer_model_name}.ckpt', strict=False)
        # Freeze the weights so that we don't try anything funny
        for param in self.predict_encoder.parameters():
            param.requires_grad = False
        '''self.predict_encoder = ClutterTransformer(self.fft_len, self.target_latent_size, 4, .01, 4, 100,
                                                  fourier_features=50, fourier_std=1., nonlinearity='psinlu')'''

        # self.attention = RelativeMultiHeadAttention(self.target_latent_size, 5)
        # self.pos_enc = PositionalEncoding(self.target_latent_size, max_len=256)

        # Step up to flowthrough channels
        self.clutter_init = nn.Sequential(
            nn.Conv1d(1, self.flowthrough_channels, 1, padding='same'),
            nlin,
            nn.Conv1d(self.flowthrough_channels, self.flowthrough_channels, 3, padding='same'),
            nlin,
        )

        self.target_init = nn.Sequential(
            nn.Conv1d(1, self.flowthrough_channels, 1, padding='same'),
            nlin,
            nn.Conv1d(self.flowthrough_channels, self.flowthrough_channels, 3, padding='same'),
            nlin,
        )

        '''SKIP LAYERS'''
        self.waveform_layers = nn.ModuleList()
        self.target_layers = nn.ModuleList()
        for _ in range(config.n_skip_layers):
            self.waveform_layers.append(nn.Sequential(
                nn.Conv1d(self.flowthrough_channels, self.waveform_channels[0], self.conv_kernel_size, padding='same'),
                nlin,
                LKA1d(self.waveform_channels[0], (self.conv_kernel_size, self.mid_kernel_size),
                      dilation=15, activation=self.nonlinearity),
                nn.Conv1d(self.waveform_channels[0], self.flowthrough_channels, self.small_kernel_size, padding='same'),
                nlin,
                nn.LayerNorm(self.target_latent_size),
            ))
            self.target_layers.append(nn.Sequential(
                nn.Linear(self.target_latent_size * 2, self.target_latent_size),
                nlin,
                nn.Conv1d(self.flowthrough_channels, self.target_channels[0], 1, padding='same'),
                nlin,
                LKA1d(self.target_channels[0], (self.conv_kernel_size, self.mid_kernel_size),
                      dilation=30, activation=self.nonlinearity),
                nn.LayerNorm(self.target_latent_size),
                nn.Conv1d(self.target_channels[0], self.target_channels[2], self.small_kernel_size, padding='same'),
                nlin,
                nn.Conv1d(self.target_channels[2], self.flowthrough_channels, self.small_kernel_size, padding='same'),
                nlin,
            ))

        self.real_decoder = nn.Sequential(
            nn.Conv1d(self.flowthrough_channels, self.n_ants, 1, padding='same'),
            nlin,
            nn.Linear(self.target_latent_size, self.fft_len),
            nn.Tanhshrink(),
        )

        self.imag_decoder = nn.Sequential(
            nn.Conv1d(self.flowthrough_channels, self.n_ants, 1, padding='same'),
            nlin,
            nn.Linear(self.target_latent_size, self.fft_len),
            nn.Tanhshrink(),
        )

        self.wave_info = nn.Sequential(
            FourierFeatureTrain(2, self.target_latent_size // 2, .33),
        )

        self.plength = PulseLength()
        self.bw_generate = WindowGenerate(self.fft_len, self.n_ants)

        _xavier_init(self)

    def forward(self, clutter, target, pulse_length, bandwidth) -> torch.Tensor:

        # Predict the next clutter step using transformer
        x_clutter = self.predict_encoder.encode(clutter)
        x_target = self.predict_encoder.encode(target)

        wave_info = self.wave_info(torch.cat([bandwidth.float().view(-1, 1),
                                              pulse_length.float().view(-1, 1) / 5000.], dim=1)).unsqueeze(1)

        x_skip = self.target_init(x_target[:, -1:] + wave_info)
        x = self.clutter_init(x_clutter[:, -1:] + wave_info)

        # Run through LKA layers to create a waveform according to spec
        for wl, tl in zip(self.waveform_layers, self.target_layers):
            x = wl(x)
            x = tl(torch.cat([x, x_skip], dim=-1))
        z = torch.cat([self.real_decoder(x), self.imag_decoder(x)], dim=1)
        z = self.plength(z, pulse_length) * self.bw_generate(bandwidth)
        z = z.view(-1, self.n_ants, 2, self.fft_len)
        return z

    def full_forward(self, clutter_array, target_array: torch.Tensor | np.ndarray, pulse_length: int, bandwidth: float) -> np.ndarray:
        """
        This function is meant to simplify using the waveforms. It takes a set of frequency domain pulses and outputs a set of waveforms.
        :param bandwidth: Normalized bandwidth. Should be between 0 and 1.
        :param clutter_array: N x fft_len array of pulse data.
        :param target_array: 1 x target_latent_size array of target embedding vector.
        :param pulse_length: int giving the length of the expected pulse in TAC.
        :return: n_ant x fft_len array of waveforms.
        """
        # Make everything into tensors and place on device
        clut_norm = normalize(clutter_array)
        clut_norm = (clut_norm - clut_norm.mean()) / clut_norm.std()
        clut = (torch.stack([torch.tensor(clut_norm.real, dtype=torch.float32, device=self.device),
                             torch.tensor(clut_norm.imag, dtype=torch.float32, device=self.device)])).swapaxes(0, 1)
        pl = torch.tensor([[pulse_length]], device=self.device)
        bw = torch.tensor([bandwidth], device=self.device)

        # Now that everything is formatted, get the waveform
        if isinstance(target_array, np.ndarray):
            tt = torch.tensor(target_array, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            tt = target_array.to(self.device).unsqueeze(0)
        z = self.forward(clut.unsqueeze(0), tt, pl, bw)
        unformatted_waveform = self.getWaveform(nn_output=z).cpu().data.numpy()

        # Return it in complex form without the leading dimension
        return unformatted_waveform[0]

    def genLFM(self, nfc, bandw, nfs, nnr):
        phase = nfc - bandw // 2 + bandw * torch.linspace(0, 1, int(nnr), dtype=torch.float32, device=self.device)
        return torch.fft.fft(torch.exp(1j * 2 * torch.pi * torch.cumsum(phase * 1 / nfs, dim=-1)), n=self.fft_len, dim=-1)

    def loss_function(self, *args) -> dict:

        # Get clutter spectrum into complex form and normalize to unit energy
        # clutter_spectrum = torch.complex(args[1][:, :, 0, :], args[1][:, :, 1, :])

        # Get target spectrum into complex form and normalize to unit energy
        target_spectrum = torch.complex(args[2][:, :, 0, :], args[2][:, :, 1, :])

        # Get waveform into complex form and normalize it to unit energy
        # First, get taylor window for sidelobe suppression
        taytay = torch.fft.fftshift(self.bw_generate(args[4])[0, 0])
        gen_waveform = self.getWaveform(nn_output=args[0])
        lfm = self.genLFM(self.fc, args[4] * self.fs, self.fs, args[3])
        lfm = (lfm / torch.sqrt(torch.sum(lfm * lfm.conj())))  # Unit energy calculation
        mfiltered = gen_waveform.conj() * gen_waveform * taytay
        lfiltered = lfm * lfm.conj() * taytay

        # Target and clutter power functions
        # Get compressed data for clutter and target
        nsam = args[6]
        targ_sans = torch.abs(torch.fft.fft(torch.fft.ifft(target_spectrum * lfiltered, dim=-1)[..., :nsam], dim=1))
        targ_ac = torch.abs(torch.fft.fft(torch.sum(torch.fft.ifft(target_spectrum.unsqueeze(1) * mfiltered.unsqueeze(2), dim=-1)[..., :nsam], dim=-3), dim=1))
        # clut_ac = torch.abs(torch.fft.fft(torch.sum(torch.fft.ifft(clutter_spectrum.unsqueeze(1) * mfiltered, dim=-1)[..., :nsam], dim=-3), dim=1))
        # Set the softplus back so that a zero value corresponds to zero in the softplus
        # This is how much higher a target is with NN than LFM plus how much higher the clutter is with an LFM vs. a NN
        # SNR of a target
        target_snr = torch.nansum(targ_ac * args[7] / (torch.sum(args[7]))) / (torch.nansum(targ_ac * ((1 - args[7]) / (torch.sum(1 - args[7])))) + EPS)

        target_loss = torch.nansum((targ_sans / (targ_ac + EPS) * args[7] / (torch.sum(args[7])) +
                     targ_ac / (targ_sans + EPS) * ((1 - args[7]) / (torch.sum(1 - args[7])))))
        # target_loss = torch.nansum((targ_ac / (targ_sans + EPS)) + (clut_ac / (targ_ac + EPS)))

        # Sidelobe loss functions
        mainlobe_softmax = ((torch.arange(int(self.fft_len), device=self.device) + EPS) / self.fft_len) ** 2
        mainlobe_softmax = torch.exp(-mainlobe_softmax / (2 * .02) ** 2)
        linear_sidelobe = torch.log(torch.abs(torch.fft.ifft(lfiltered)) + EPS)
        slf = nn_func.relu((torch.log(torch.abs(torch.fft.ifft(mfiltered)) + EPS) - linear_sidelobe))
        sidelobe_loss = torch.nansum(slf * mainlobe_softmax) / torch.sum(mainlobe_softmax)
        # sidelobe_loss = torch.tensor(0., device=self.device)
        '''
        import matplotlib.pyplot as plt
        plt.figure("Spectrum")
        plt.subplot(1, 2, 1)
        plt.imshow(np.log10(abs(np.fft.ifft(target_spectrum.data.cpu().numpy()[0]))))
        plt.axis('tight')
        plt.subplot(1, 2, 2)
        plt.imshow(np.log10(abs(np.fft.ifft(clutter_spectrum.data.cpu().numpy()[0]))))
        plt.axis('tight')
        
        plt.figure("AC")
        plt.subplot(1, 2, 1)
        plt.imshow(np.log10(targ_ac.data.cpu().numpy()[0]))
        plt.axis('tight')
        plt.subplot(1, 2, 2)
        plt.imshow(np.log10(clut_ac.data.cpu().numpy()[0]))
        plt.axis('tight')
        
        plt.figure("Sans")
        plt.imshow(np.log10(targ_sans.data.cpu().numpy()[0]))
        plt.axis('tight')
        
        plt.figure("Softmax")
        plt.plot(target_softmax.data.cpu().numpy()[0])
        
        plt.figure("LFM")
        plt.plot(np.log10(abs(lfm.data.cpu().numpy())))
        
        plt.figure("mfiltered")
        plt.plot(np.log10(abs(np.fft.ifft(mfiltered[0, 0].data.cpu().numpy()))))
        
        plt.figure('Comparison')
        plt.subplot(1, 3, 1)
        plt.title('above_lfm')
        plt.imshow(above_lfm.data.cpu().numpy()[0].T)
        plt.axis('tight')
        plt.subplot(1, 3, 2)
        plt.title('target_snr')
        plt.imshow(target_snr.data.cpu().numpy()[0].T)
        plt.axis('tight')
        plt.subplot(1, 3, 3)
        plt.title('clutter_mit')
        plt.imshow(clutter_mit.data.cpu().numpy()[0].T)
        plt.axis('tight')
        
        plt.figure()
        plt.plot((1 - target_softmax / (torch.sum(target_softmax) + EPS).data.cpu().numpy()[0])
        
        
        '''
        # Orthogonality
        if self.n_ants > 1:
            crossfiltered = gen_waveform * torch.flip(gen_waveform.conj(), dims=(1,))
            cross_sidelobe = torch.abs(torch.fft.ifft(crossfiltered, dim=2)) + EPS
            cross_sidelobe = torch.log(cross_sidelobe)
            clf = nn_func.relu((cross_sidelobe - linear_sidelobe))
            ortho_loss = 1. / (torch.nansum(clf * mainlobe_softmax) / torch.sum(mainlobe_softmax) + EPS)
        else:
            ortho_loss = torch.tensor(0., device=self.device)

        # total_loss = target_loss * (1 + min(.001 * self.current_epoch, 1.) * sidelobe_loss + ortho_loss)
        total_loss = target_loss**2  # torch.sqrt(sidelobe_loss * .25 + target_loss * .75 + ortho_loss)

        return {'target_loss': target_loss, 'loss': total_loss,
                'sidelobe_loss': sidelobe_loss, 'ortho_loss': ortho_loss,
                'target_snr': target_snr}

    def getWaveform(self, cc: Tensor = None, tc: Tensor = None, pulse_length=None,
                    bandwidth: Tensor | float = 400e6, nn_output: tuple[Tensor] = None) -> Tensor:
        """
        Given a clutter and target spectrum, produces a waveform FFT.
        :param custom_fft_sz: If scale is True, outputs waveforms of custom_fft_sz
        :param bandwidth: Desired bandwidth of waveform.
        :param scale: If True, scales the output FFT so that it is at least pulse_length long on IFFT.
        :param pulse_length: Length of pulse in samples.
        :param use_window: if True, applies a window to the finished waveform. Set to False for training.
        :param nn_output: Optional. If the waveform data is already created, use this to avoid putting in cc and tc.
        :param cc: Tensor of clutter spectrum. Same as input to model.
        :param tc: Tensor of target spectrum. Same as input to model.
        :return: Tensor of waveform FFTs, of size (batch_sz, n_ants, fft_sz).
        """
        if pulse_length is None:
            pulse_length = [1]

        # Get the STFT either from the clutter, target, and pulse length or directly from the neural net
        dual_waveform = self.forward(cc, tc, pulse_length, bandwidth) if nn_output is None else nn_output
        gen_waveform = torch.zeros((dual_waveform.shape[0], self.n_ants, self.fft_len), dtype=torch.complex64,
                                   device=self.device)
        for n in range(self.n_ants):
            complex_wave = torch.fft.fftshift(torch.complex(dual_waveform[:, n, 0, :],
                                                            dual_waveform[:, n, 1, :]), dim=1)
            gen_waveform[:, n, :] = (complex_wave / torch.sqrt(torch.sum(complex_wave *
                                                                         torch.conj(complex_wave),
                                                                         dim=1))[:, None])  # Unit energy calculation
        return gen_waveform

    def training_step(self, batch, batch_idx):
        train_loss = self.train_val_get(batch, batch_idx)
        self.manual_backward(train_loss['loss'])
        if (self.global_step + 1) % self.config.accumulation_steps == 0 and self.config.gradient_flow:
            plot_grad_flow(self.named_parameters())
        opt = self.optimizers()
        opt.step()
        opt.zero_grad()
        self.lr_schedulers().step()
        self.log_dict(train_loss, sync_dist=True,
                      prog_bar=True, rank_zero_only=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        self.log_dict(self.train_val_get(batch, batch_idx), sync_dist=True, prog_bar=True,
                      rank_zero_only=True)

    def on_validation_epoch_end(self) -> None:
        self.log('lr', self.lr_schedulers().get_last_lr()[0], prog_bar=True, rank_zero_only=True)

    def on_train_epoch_end(self) -> None:
        '''sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])
        else:
            sch.step()
        if self.trainer.is_global_zero and not self.config.is_tuning and self.config.loss_landscape:
            self.optim_path.append(self.get_flat_params().cpu().data.numpy())'''
        self.log('lr', self.lr_schedulers().get_last_lr()[0], prog_bar=True, rank_zero_only=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.config.lr,
                                      weight_decay=self.config.weight_decay,
                                      betas=self.config.betas,
                                      eps=1e-7)
        if self.config.scheduler_gamma is None:
            return optimizer
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config.scheduler_gamma)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=1e-9)
        scheduler = CosineWarmupScheduler(optimizer, warmup=self.config.warmup // self.config.accumulation_steps,
                                          max_iters=self.config.max_iters // self.config.accumulation_steps)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def train_val_get(self, batch, batch_idx):
        clutter_spec, target_spec, both_spec, target_rbin, pulse_length, bandwidth, nsam, truth = batch
        seq_len = clutter_spec.shape[1] // 2
        results = self.forward(clutter_spec[:, :seq_len], target_spec[:, :seq_len], pulse_length, bandwidth)
        return self.loss_function(results, clutter_spec[:, seq_len:], both_spec[:, seq_len:], pulse_length, bandwidth, target_rbin, nsam, truth)

    def on_after_backward(self):
        pass
        # example to inspect gradient information in tensorboard
        # if self.trainer.global_step % 100 == 0 and self.trainer.global_step != 0:
        #    self.plot_grad_flow(self.named_parameters())

if __name__ == '__main__':
    from config import get_config
    from torchviz import make_dot
    from pytorch_lightning import seed_everything
    torch.set_float32_matmul_precision('medium')
    gpu_num = 1
    device = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'
    seed_everything(np.random.randint(1, 2048), workers=True)
    # seed_everything(43, workers=True)

    target_config = get_config('wave_exp', './vae_config.yaml')
    mdl = GeneratorModel(config=target_config)

    dummy = (torch.zeros((1, 32, 2, target_config.fft_len)),
                                      torch.zeros((1, target_config.target_latent_size)),
                                      torch.tensor([1250]), torch.tensor([.4]))
    output = mdl(*dummy)
    import sys
    sys.setrecursionlimit(20000)
    dot = make_dot(output, params=dict(mdl.named_parameters()))

    dot.format = 'png'
    dot.render('wavemodel')