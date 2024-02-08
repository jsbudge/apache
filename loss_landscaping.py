import pickle
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging

import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
import torch
import yaml
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from dataloaders import WaveDataModule
from experiment import GeneratorExperiment
from models import BetaVAE, InfoVAE, WAE_MMD
from waveform_model import GeneratorModel
from data_converter.SDRParsing import load
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator

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
    optim_path_matrix = np.vstack([np.array(tensor.cpu()) for tensor in params_path])
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
            cc: torch.Tensor, tc: torch.Tensor, cs: torch.Tensor, ts: torch.Tensor,
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

        # Build the loss grid
        i, j = np.meshgrid(np.arange(-res, res) * alpha, np.arange(-res, res) * alpha)
        losses = np.zeros_like(i)
        with tqdm(total=i.shape[0] * i.shape[1]) as pbar:
            for x in range(losses.shape[0]):
                for y in range(losses.shape[1]):
                    w_ij = torch.tensor(i[x, y] * directions[0] + j[x, y] * directions[1] +
                                        self.optim_point.cpu().data.numpy(), device=self.device)
                    model.init_from_flat_params(w_ij)
                    y_pred = model(cc, tc, [2600])
                    losses[x, y] = model.loss_function(y_pred, cs, ts)['loss']
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
    with open('./wave_simulator.yaml') as y:
        settings = yaml.safe_load(y.read())
    with open('./vae_config.yaml', 'r') as file:
        try:
            wave_config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wave_config['dataset_params']['max_pulse_length'] = 5000
    wave_config['dataset_params']['min_pulse_length'] = wave_config['settings']['stft_win_sz'] + 64

    print('Setting up model...')
    if wave_config['exp_params']['model_type'] == 'InfoVAE':
        vae_mdl = InfoVAE(**wave_config['model_params'])
    elif wave_config['exp_params']['model_type'] == 'WAE_MMD':
        vae_mdl = WAE_MMD(**wave_config['model_params'])
    else:
        vae_mdl = BetaVAE(**wave_config['model_params'])
    vae_mdl.load_state_dict(torch.load('./model/inference_model.state'))
    vae_mdl.eval()  # Set to inference mode

    print('Setting up dataloader...')
    data = WaveDataModule(vae_model=vae_mdl, device=device, **wave_config["dataset_params"])
    data.setup()

    vae_mdl.to('cpu')

    with open('./model/current_model_params.pic', 'rb') as f:
        generator_params = pickle.load(f)

    print('Setting up wavemodel...')
    wave_mdl = GeneratorModel(**generator_params)
    wave_mdl.load_state_dict(torch.load(generator_params['state_file']))

    wave_config['wave_exp_params']['loss_landscape'] = True
    experiment = GeneratorExperiment(wave_mdl, wave_config['wave_exp_params'])
    logger = loggers.TensorBoardLogger(wave_config['train_params']['log_dir'],
                                       name="WaveModel")
    trainer = Trainer(logger=logger, max_epochs=wave_config['train_params']['max_epochs'],
                      log_every_n_steps=wave_config['exp_params']['log_epoch'], devices=1,
                      strategy='ddp', gradient_clip_val=.5, callbacks=
                      [EarlyStopping(monitor='loss', patience=wave_config['wave_exp_params']['patience'],
                                     check_finite=True), StochasticWeightAveraging(swa_lrs=1e-2)])

    print("======= Training =======")
    trainer.fit(experiment, datamodule=data)

    if trainer.global_rank == 0:
        reduced_dict = dim_reduce(experiment.optim_path, 'pca')
        path_2d = reduced_dict["path_2d"]
        directions = reduced_dict["reduced_dirs"]

        cc, tc, cs, ts, _ = next(iter(data.train_dataloader()))
        cc = cc.to(device)
        tc = tc.to(device)
        cs = cs.to(device)
        ts = ts.to(device)

        loss_grid = LossGrid(
            experiment.optim_path,
            wave_mdl,
            cc, tc, cs, cs,
            path_2d,
            directions,
            device,
            res=30,
            margin=.3,
        )

        plt.figure('Loss Landscape')
        plt.imshow(loss_grid.grid,
                   extent=(loss_grid.coords[0].min(), loss_grid.coords[0].max(),
                           loss_grid.coords[1].min(), loss_grid.coords[1].max()))
        plt.plot(loss_grid.path_2d[:, 0], loss_grid.path_2d[:, 1])

        xx, yy = np.meshgrid(loss_grid.coords[0], loss_grid.coords[1])
        scaled_grid = loss_grid.grid.flatten() - loss_grid.grid.min()
        scaled_grid /= scaled_grid.max()

        path_interp = RegularGridInterpolator((loss_grid.coords[0], loss_grid.coords[1]), loss_grid.grid)
        zs = path_interp(path_2d, method='cubic')

        fig = go.Figure(data=[
            go.Mesh3d(x=xx.flatten(), y=yy.flatten(), z=loss_grid.grid.flatten(),
                      alphahull=-1, colorscale='Turbo',
                      intensity=scaled_grid), go.Scatter3d(x=path_2d[:, 0], y=path_2d[:, 1], z=zs)])
        fig.show()
        plt.show()
