import torch
from neuralop import TFNO1d
from torch import nn, Tensor
from pytorch_lightning import LightningModule
from models import ClutterTransformer, FlatModule
from torch.nn import functional as nn_func
from config import Config
from schedulers import CosineWarmupScheduler
from layers import FourierFeature, PulseLength, LKA1d, WindowGenerate, PositionalEncoding, Block1d, RBFLayer, SwiGLU, FourierFeatureTrain
import numpy as np
from utils import normalize, _xavier_init, nonlinearities, plot_grad_flow, rbf_linear, l_norm

EPS = 1e-12


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
        self.clutter_target_channels = config.clutter_target_channels
        self.flowthrough_channels = config.flowthrough_channels
        self.wave_decoder_channels = config.wave_decoder_channels
        self.waveform_channels = config.waveform_channels
        self.target_channels = config.target_channels
        self.automatic_optimization = False
        self.bandwidth = config.bandwidth
        self.baseband_fc = (config.fc % self.fs)
        self.nonlinearity = config.nonlinearity
        nlin = nonlinearities[self.nonlinearity]

        '''TRANSFORMER'''
        self.predict_encoder = ClutterTransformer.load_from_checkpoint(
            f'{config.transformer_weights_path}/{config.transformer_model_name}.ckpt', strict=False)
        # Freeze the weights so that we don't try anything funny
        for param in self.predict_encoder.parameters():
            param.requires_grad = False

        self.rbf_distance = nn.Sequential(
            RBFLayer(self.target_latent_size, 15, self.target_latent_size, radial_function=rbf_linear, norm_function=l_norm, normalization=True),
        )

        self.clutter_init = nn.Sequential(
            nn.Linear(self.target_latent_size, self.target_latent_size),
            nlin,
            nn.Linear(self.target_latent_size, self.target_latent_size),
            nlin,
        )

        self.target_init = nn.Sequential(
            nn.Linear(self.target_latent_size, self.target_latent_size),
            nlin,
            nn.Linear(self.target_latent_size, self.target_latent_size),
            nlin,
        )

        self.clutter_target_init = nn.Sequential(
            nn.Linear(self.target_latent_size, self.target_latent_size),
            nlin,
            nn.Linear(self.target_latent_size, self.target_latent_size),
            nlin,
            nn.Conv1d(1, self.flowthrough_channels, 3, 1, 1),
            nlin,
        )

        '''SKIP LAYERS'''
        self.waveform_layers = nn.ModuleList()
        self.target_layers = nn.ModuleList()
        for _ in range(config.n_skip_layers):
            self.waveform_layers.append(nn.Sequential(
                nn.Conv1d(self.flowthrough_channels, self.waveform_channels, 3, 1, 1),
                nlin,
                nn.Linear(self.target_latent_size, self.target_latent_size),
                nlin,
                nn.Conv1d(self.waveform_channels, self.flowthrough_channels, 3, 1, 1),
                nlin,
                nn.LayerNorm(self.target_latent_size),
            ))
            self.target_layers.append(nn.Sequential(
                nn.Conv1d(self.flowthrough_channels, self.target_channels, 3, 1, 1),
                nlin,
                nn.Linear(self.target_latent_size, self.target_latent_size),
                nlin,
                nn.Conv1d(self.target_channels, self.flowthrough_channels, 3, 1, 1),
                nlin,
                nn.LayerNorm(self.target_latent_size),
            ))

        self.wave_decoder = nn.Sequential(
            nn.Conv1d(self.flowthrough_channels, self.wave_decoder_channels, 3, 1, 1),
            nlin,
            SwiGLU(self.target_latent_size, self.target_latent_size * 2, self.target_latent_size),
            nn.Conv1d(self.wave_decoder_channels, self.n_ants * 2, 3, 1, 1),
            nn.Linear(self.target_latent_size, self.fft_len),
        )

        self.wave_info = nn.Sequential(
            FourierFeatureTrain(2, self.target_latent_size // 2, 1.),
        )

        self.plength = PulseLength()
        self.bw_generate = WindowGenerate(self.fft_len, self.n_ants)

        _xavier_init(self.wave_decoder)
        _xavier_init(self.predict_encoder)
        _xavier_init(self.clutter_target_init)
        _xavier_init(self.wave_info)

    def forward(self, clutter, target, pulse_length, bandwidth) -> torch.Tensor:

        # Predict the next clutter step using transformer
        x_clutter = self.predict_encoder.encode(clutter)[:, -1].unsqueeze(1)
        x_target = self.predict_encoder.encode(target)[:, -1].unsqueeze(1)

        y = (self.rbf_distance(x_clutter) - self.rbf_distance(x_target)).unsqueeze(1)

        # Combine clutter prediction with target information
        xc = self.clutter_init(x_clutter + y)
        xt = self.target_init(x_target + y)
        x = self.clutter_target_init(xc + xt)

        wave_info = self.wave_info(torch.cat([bandwidth.float().view(-1, 1), pulse_length.float().view(-1, 1) / 5000.], dim=1)).unsqueeze(1)

        # Run through LKA layers to create a waveform according to spec
        for wl, tl in zip(self.waveform_layers, self.target_layers):
            x = wl(x + wave_info)
            x = tl(x + xt)
        x = self.wave_decoder(x)
        x = self.plength(x, pulse_length) * self.bw_generate(bandwidth)
        x = x.view(-1, self.n_ants, 2, self.fft_len)
        return x

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
        unformatted_waveform = self.getWaveform(nn_output=self.forward(clut.unsqueeze(0), tt, pl, bw)).cpu().data.numpy()

        # Return it in complex form without the leading dimension
        return unformatted_waveform[0]

    def loss_function(self, *args) -> dict:

        # Get clutter spectrum into complex form and normalize to unit energy
        clutter_spectrum = torch.complex(args[1][:, -1, 0, :], args[1][:, -1, 1, :])

        # Get target spectrum into complex form and normalize to unit energy
        target_spectrum = torch.complex(args[2][:, -1, 0, :], args[2][:, -1, 1, :])

        # Get waveform into complex form and normalize it to unit energy
        gen_waveform = self.getWaveform(nn_output=args[0])
        mfiltered = gen_waveform * gen_waveform.conj()

        # Target and clutter power functions
        targ_ac = torch.abs(torch.sum(torch.fft.ifft(target_spectrum.unsqueeze(1) * mfiltered, dim=2)[..., :3528], dim=1))
        clut_ac = torch.abs(torch.sum(torch.fft.ifft(clutter_spectrum.unsqueeze(1) * mfiltered, dim=2)[..., :3528], dim=1))
        targ_abs = torch.abs(torch.sum(torch.fft.ifft(target_spectrum.unsqueeze(1), dim=2)[..., :3528], dim=1))
        clut_abs = torch.abs(torch.sum(torch.fft.ifft(clutter_spectrum.unsqueeze(1), dim=2)[..., :3528], dim=1))
        # target_softmax = ((torch.arange(targ_ac.shape[-1], device=self.device) - args[3] + EPS) / targ_ac.shape[-1])**2
        # target_softmax = torch.exp(-target_softmax / (2 * self.config.temperature)**2)
        tsoft = nn_func.conv1d(torch.abs(targ_abs - clut_abs).unsqueeze(1), torch.tensor([[[.1, .5, 1., .5, .1]]], dtype=torch.float32, device=self.device), padding=2).squeeze(1)
        target_softmax = tsoft / (torch.sum(tsoft, dim=1, keepdim=True) + EPS)
        target_loss = torch.nansum(clut_ac / (EPS + targ_ac) * target_softmax / (torch.sum(target_softmax) + EPS))# + 1. / kld_loss
        # target_loss = torch.nanmean(nn_func.logsigmoid(clut_ac - targ_ac))# + kld_loss

        # Sidelobe loss functions
        slf = nn_func.threshold(torch.abs(torch.fft.ifft(gen_waveform * gen_waveform.conj(), dim=2)), 1e-9, 1e-9)
        sidelobe_func = 10 * torch.log(slf / 10)
        sidelobe_window = torch.ones(self.fft_len, device=self.device)
        theoretical_mainlobe_width = torch.max(torch.ceil(2. / args[5])).int()
        sidelobe_window[-theoretical_mainlobe_width:] = 1. - torch.sin(
            torch.pi * torch.arange(theoretical_mainlobe_width * 2, device=self.device)[:theoretical_mainlobe_width] / (theoretical_mainlobe_width * 2))
        sidelobe_window[:theoretical_mainlobe_width] = 1. - torch.sin(
            torch.pi * torch.arange(theoretical_mainlobe_width * 2, device=self.device)[-theoretical_mainlobe_width:] / (
                        theoretical_mainlobe_width * 2))
        pslrs = torch.max(sidelobe_func, dim=-1)[0] - torch.mean(
            sidelobe_func * sidelobe_window, dim=-1)
        # pslrs = torch.max(sidelobe_func, dim=-1)[0] - torch.mean(sidelobe_func[..., theoretical_mainlobe_width:-theoretical_mainlobe_width], dim=-1)
        # pslrs = get_pslr(sidelobe_func.squeeze(1))
        sidelobe_loss = (1e2 / (EPS + torch.nanmean(pslrs)))**2
        # sidelobe_loss = torch.tensor(0., device=self.device)

        # Orthogonality
        if self.n_ants > 1:
            crossfiltered = gen_waveform * torch.flip(gen_waveform.conj(), dims=(1,))
            cross_sidelobe = torch.abs(torch.fft.ifft(crossfiltered, dim=2))
            cross_sidelobe[cross_sidelobe == 0] = 1e-9
            cross_sidelobe = 10 * torch.log(cross_sidelobe / 10)
            ortho_loss = torch.nanmean(torch.abs(sidelobe_func.sum(dim=1)[:, 0]) /
                                       (EPS + torch.abs(cross_sidelobe.sum(dim=1)[:, 0])))**2
        else:
            ortho_loss = torch.tensor(0., device=self.device)

        total_loss = target_loss * (1 + sidelobe_loss + ortho_loss)

        return {'target_loss': target_loss, 'loss': total_loss,
                'sidelobe_loss': sidelobe_loss, 'ortho_loss': ortho_loss} #, 'kld_loss': kld_loss}

    def getWaveform(self, cc: Tensor = None, tc: Tensor = None, pulse_length=None,
                    bandwidth: [Tensor | float] = 400e6, nn_output: tuple[Tensor] = None) -> Tensor:
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

    def getMatchedFilter(self, wave: Tensor, bw: float):
        twin = torch.signal.windows.kaiser(int(bw / self.fs * self.fft_len), device=self.device)
        taytay = torch.zeros(self.fft_len, dtype=torch.complex128, device=self.device)
        winloc = int(self.baseband_fc * self.fft_len / self.fs) - len(twin) // 2
        if winloc + len(twin) > self.fft_len:
            taytay[winloc:self.fft_len] += twin[:self.fft_len - winloc]
            taytay[:len(twin) - (self.fft_len - winloc)] += twin[self.fft_len - winloc:]
        else:
            taytay[winloc:winloc + len(twin)] += twin

        return taytay / (wave + 1e-12)

    def training_step(self, batch, batch_idx):
        train_loss = self.train_val_get(batch, batch_idx)
        self.manual_backward(train_loss['loss'] / self.config.accumulation_steps)
        if (batch_idx + 1) % self.config.accumulation_steps == 0:
            if self.config.gradient_flow:
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
        clutter_spec, target_spec, target_rbin, pulse_length, bandwidth = batch
        results = self.forward(clutter_spec, target_spec, pulse_length, bandwidth)
        return self.loss_function(results, clutter_spec, target_spec, pulse_length, bandwidth, target_rbin)

    def on_after_backward(self):
        # example to inspect gradient information in tensorboard
        if self.trainer.global_step % 100 == 0 and self.trainer.global_step != 0:
           self.plot_grad_flow(self.named_parameters())

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