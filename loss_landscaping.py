import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
from config import get_config
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, ModelCheckpoint
from dataloaders import WaveDataModule
from waveform_model import GeneratorModel

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'


def dim_reduce(params_path: list, reduction_method: str = 'pca', seed: int = 43) -> dict:
    """
    Reduce the dimensions of a parameter path from a GeneratorModel object.
    :param params_path: list of weights and biases from a GeneratorModel, flattened to be in one dimension.
    :param reduction_method: {'pca', 'random'} Applies the reduction method. If PCA, uses PCA,
    else picks a random direction.
    :param seed: Seed integer used in the random states for reduction methods. Only for reproducability.
    :return: dict with directions for reduced dimension projection as well as the optimization path in 2d.
    """
    optim_path_matrix = np.vstack(params_path)
    reduce_dict = {'optim_path': optim_path_matrix}
    if reduction_method == 'pca':
        pca = PCA(n_components=2, random_state=seed)
        path_2d = pca.fit_transform(optim_path_matrix)
        reduced_dirs = pca.components_
        reduce_dict['pcvariances'] = pca.explained_variance_ratio_
    else:
        print("Generating random axes...")
        # Generate 2 random unit vectors (u, v)
        if seed:
            np.random.seed(seed)
        u_gen = np.random.normal(size=optim_path_matrix.shape[1])
        u = u_gen / np.linalg.norm(u_gen)
        v_gen = np.random.normal(size=optim_path_matrix.shape[1])
        v = v_gen / np.linalg.norm(v_gen)
        reduced_dirs = np.array([u, v])
        path_2d = optim_path_matrix.dot(reduced_dirs.T)
    reduce_dict['path_2d'] = path_2d
    reduce_dict['reduced_dirs'] = reduced_dirs
    return reduce_dict


class LossGrid:

    def __init__(
            self,
            optim_path: list,
            model: GeneratorModel,
            train_set: list,
            path_2d: np.ndarray,
            directions: np.ndarray,
            device: str,
            res: int = 30,
            margin: float = .3,
    ):
        self.path_2d = path_2d
        self.optim_point = optim_path[-1]
        self.optim_point_2d = path_2d[-1]
        self.device = device

        alpha = self._compute_stepsize(res, margin)
        model.to(device)
        cs, ts, tc, pl, bw, _, rbin = train_set

        # Build the loss grid
        i, j = np.meshgrid(np.arange(-res, res) * alpha, np.arange(-res, res) * alpha)
        losses = np.zeros_like(i)
        with tqdm(total=i.shape[0] * i.shape[1]) as pbar:
            for x in range(losses.shape[0]):
                for y in range(losses.shape[1]):
                    w_ij = torch.tensor(i[x, y] * directions[0] + j[x, y] * directions[1] +
                                        self.optim_point, device=self.device)
                    model.init_from_flat_params(w_ij)
                    y_pred = model(cs.to(device), tc.to(device), pl.to(device), bw.to(device))
                    losses[x, y] = model.loss_function(y_pred, cs.to(device), ts.to(device), tc.to(device),
                                                       pl.to(device), bw.to(device), rbin.to(device))['loss']
                    pbar.update(1)

        self.grid = losses

        self.coords = self._convert_coords(res, alpha)
        # True optim in loss grid
        # self.true_optim_point = self.indices_to_coords(np.argmin(losses), res, alpha)

    @property
    def shape(self):
        return self.grid.shape

    def _convert_coord(self, i, ref_point_coord, alpha):
        """
        Convert from integer index to the coordinate value.

        Given a reference point coordinate (1D), find the value i steps away with
        step size alpha.
        """
        return i * alpha + ref_point_coord

    def _convert_coords(self, res, alpha):
        """
        Convert the coordinates from (i, j) indices to (x, y) values.

        Remember that for PCA, the coordinates have unit vectors as the top 2 PCs.

        Original path_2d has PCA output, i.e. the 2D projections of each W step
        onto the 2D space spanned by the top 2 PCs.
        We need these steps in (i, j) terms with unit vectors
        reduced_w1 = (1, 0) and reduced_w2 = (0, 1) in the 2D space.

        We center the plot on optim_point_2d, i.e.
        let center_2d = optim_point_2d

        ```
        i = (x - optim_point_2d[0]) / alpha
        j = (y - optim_point_2d[1]) / alpha

        i.e.

        x = i * alpha + optim_point_2d[0]
        y = j * alpha + optim_point_2d[1]
        ```

        where (x, y) is the 2D points in path_2d from PCA. Again, the unit
        vectors are reduced_w1 and reduced_w2.
        Return the grid coordinates in terms of (x, y) for the loss values
        """
        converted_coord_xs = []
        converted_coord_ys = []
        for i in range(-res, res):
            x = self._convert_coord(i, self.optim_point_2d[0], alpha)
            y = self._convert_coord(i, self.optim_point_2d[1], alpha)
            converted_coord_xs.append(x)
            converted_coord_ys.append(y)
        return np.array(converted_coord_xs), np.array(converted_coord_ys)

    def indices_to_coords(self, indices, res, alpha):
        """Convert the (i, j) indices to (x, y) coordinates.

        Args:
            indices: (i, j) indices to convert.
            res: Resolution.
            alpha: Step size.

        Returns:
            The (x, y) coordinates in the projected 2D space.
        """
        grid_i, grid_j = indices
        i, j = grid_i - res, grid_j - res
        x = i * alpha + self.optim_point_2d[0]
        y = j * alpha + self.optim_point_2d[1]
        return x, y

    def _compute_stepsize(self, res, margin):
        """
        Compute the step size. This is calculated to span at least
        the distance of the optimization path, plus a margin.
        :param res: Number of steps on either side of optimization path.
        :param margin: Margin outside of optimization path in the plane.
        :return: Stepsize for plane directions.
        """
        dist_2d = self.path_2d[-1] - self.path_2d[0]
        dist = (dist_2d[0] ** 2 + dist_2d[1] ** 2) ** 0.5
        return dist * (1 + margin) / res


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    torch.autograd.set_detect_anomaly(True)
    # force_cudnn_initialization()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.cuda.empty_cache()

    # seed_everything(np.random.randint(1, 2048), workers=True)
    # seed_everything(107, workers=True)

    config = get_config('wave_exp', './vae_config.yaml')

    fft_len = config.fft_len
    nr = 5000  # int((config['perf_params']['vehicle_slant_range_min'] * 2 / c0 - 1 / TAC) * fs)
    # Since these are dependent on apache params, we set them up here instead of in the yaml file
    print('Setting up data generator...')
    config.dataset_params['max_pulse_length'] = nr
    config.dataset_params['min_pulse_length'] = 1000

    config.loss_landscape = True

    data = WaveDataModule(device=device, **config.dataset_params)
    data.setup()

    print('Initializing wavemodel...')
    if config.warm_start:
        wave_mdl = GeneratorModel.load_from_checkpoint(f'{config.weights_path}/{config.model_name}.ckpt', config=config,
                                                       strict=False)
    else:
        wave_mdl = GeneratorModel(config=config)
    logger = loggers.TensorBoardLogger(config.log_dir,
                                       name=config.model_name, log_graph=True)
    expected_lr = max((config.lr * config.scheduler_gamma ** (config.max_epochs * config.swa_start)), 1e-9)
    trainer = Trainer(logger=logger, limit_train_batches=32, max_epochs=10, num_sanity_val_steps=0,
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

    if trainer.global_rank == 0:
        reduced_dict = dim_reduce(wave_mdl.optim_path, 'rando')
        path_2d = reduced_dict["path_2d"]
        directions = reduced_dict["reduced_dirs"]

        train_set = next(iter(data.train_dataloader()))

        loss_grid = LossGrid(
            wave_mdl.optim_path,
            wave_mdl,
            train_set,
            path_2d,
            directions,
            device,
            res=30,
            margin=1.,
        )

        plt.figure('Loss Landscape')
        plt.imshow(loss_grid.grid,
                   extent=(loss_grid.coords[0].min(), loss_grid.coords[0].max(),
                           loss_grid.coords[1].min(), loss_grid.coords[1].max()))
        plt.plot(loss_grid.path_2d[:, 0], loss_grid.path_2d[:, 1])

        xx, yy = np.meshgrid(loss_grid.coords[0], loss_grid.coords[1])
        grid_min = loss_grid.grid.min()
        scaled_grid = loss_grid.grid.flatten() - grid_min
        grid_max = scaled_grid.max()
        scaled_grid /= grid_max

        optim_x, optim_y = np.where(loss_grid.grid == grid_min)
        landscape_optim = torch.tensor(loss_grid.coords[0][optim_x[0]] * directions[0] +
                           loss_grid.coords[1][optim_y[0]] * directions[1] + loss_grid.optim_point)

        path_interp = RegularGridInterpolator((loss_grid.coords[0], loss_grid.coords[1]), loss_grid.grid,
                                              bounds_error=False, fill_value=0)
        zs = path_interp(path_2d, method='cubic')

        fig = go.Figure(data=[
            go.Mesh3d(x=xx.flatten(), y=yy.flatten(), z=loss_grid.grid.flatten(),
                      alphahull=-1, colorscale='Turbo',
                      intensity=scaled_grid), go.Scatter3d(x=path_2d[:, 0], y=path_2d[:, 1], z=zs)])
        fig.show()
        plt.show()
