from glob import glob
from typing import List, Optional, Sequence, Union

import yaml
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch
from torchdata.datapipes.iter import FileLister, FileOpener
from torchvision import transforms
from pathlib import Path
import numpy as np
from multiprocessing import cpu_count
from scipy.ndimage import sobel
from scipy.signal.windows import taylor
from sklearn.model_selection import train_test_split

from models import BaseVAE, InfoVAE, WAE_MMD, BetaVAE


class PulseDataset(Dataset):
    def __init__(self, root_dir, fft_len, split=1., is_val=False, seed=42):
        self.root_dir = root_dir
        self.fft_len = fft_len
        _, self.data = self.get_filedata()
        # Do split
        if split < 1:
            Xt, Xs, _, _ = train_test_split(np.arange(self.data.shape[0]), np.arange(self.data.shape[0]),
                                            test_size=split, random_state=seed)
            self.data = self.data[Xs] if is_val else self.data[Xt]
        self.data = torch.tensor(self.data, dtype=torch.float32)
        
    def __getitem__(self, idx):
        img = self.data[idx, ...]
        return img, img

    def __len__(self):
        return self.data.shape[0]

    def get_filedata(self, concat=True):
        if not Path(self.root_dir).is_dir():
            return self.root_dir, np.fromfile(self.root_dir, dtype=np.float32).reshape((-1, 2, self.fft_len))
        clutter_files = glob(f'{self.root_dir}/clutter_*.spec')
        dt = [np.fromfile(c, dtype=np.float32).reshape((-1, 2, self.fft_len)) for c in clutter_files]
        return clutter_files, np.concatenate(dt) if concat else dt


class RCSDataset(Dataset):
    def __init__(self, datapath='./data', split=None):
        # Load in data
        optical_data = np.fromfile('/home/jeff/repo/apache/data/ge_img.dat',
                                   dtype=np.float32).reshape((-1, 256, 256, 3)).swapaxes(1, 3)
        sar_data = 10 * np.log(np.fromfile('/home/jeff/repo/apache/data/sar_img.dat',
                                           dtype=np.float32).reshape((-1, 1, 256, 256)))
        param_data = np.fromfile('/home/jeff/repo/apache/data/params.dat', dtype=np.float32).reshape((-1, 7))
        self.optical_data = torch.tensor(optical_data)
        self.sar_data = torch.tensor(1 - sar_data / sar_data.min())
        self.param_data = torch.tensor(param_data)

        if split:
            idxes = np.random.choice(np.arange(self.optical_data.shape[0]), split)
            self.optical_data = self.optical_data[idxes, ...]
            self.sar_data = self.sar_data[idxes, ...]
            self.param_data = self.param_data[idxes, ...]

    def __getitem__(self, idx):
        return self.optical_data[idx, :], self.sar_data[idx, :], self.param_data[idx, :]

    def __len__(self):
        return self.optical_data.shape[0]


class WaveDataset(Dataset):
    def __init__(self, root_dir: str, latent_dim: int = 50, fft_sz: int = 4096,
                 split: float = 1., single_example: bool = False, min_pulse_length: int = 1,
                 max_pulse_length: int = 2, seq_len: int = 32, is_val=False, seed=42):
        assert Path(root_dir).is_dir()

        clutter_spec_files = glob(f'{root_dir}/clutter_*.spec')
        target_spec_files = glob(f'{root_dir}/targets.spec')
        clutter_enc_files = glob(f'{root_dir}/clutter_*.enc')
        target_enc_files = glob(f'{root_dir}/targets.enc')
        self.ccdata = np.concatenate(
            [np.fromfile(c, dtype=np.float32).reshape((-1, latent_dim)) for c in clutter_enc_files])
        self.tcdata = torch.tensor(np.concatenate(
            [np.fromfile(c, dtype=np.float32).reshape((-1, latent_dim)) for c in target_enc_files]))
        self.csdata = torch.tensor(
            np.concatenate([np.fromfile(c, dtype=np.float32).reshape((-1, 2, fft_sz)) for c in clutter_spec_files]))
        self.tsdata = torch.tensor(
            np.concatenate([np.fromfile(c, dtype=np.float32).reshape((-1, 2, fft_sz)) for c in target_spec_files]))
        if split < 1:
            Xt, Xs, _, _ = train_test_split(np.arange(self.ccdata.shape[0] - seq_len),
                                            np.arange(self.ccdata.shape[0] - seq_len),
                                            test_size=split, random_state=seed)
        else:
            Xt = np.arange(self.ccdata.shape[0] - seq_len)
            Xs = np.arange(self.ccdata.shape[0] - seq_len)

        self.idxes = Xs if is_val else Xt

        self.spec_sz = self.tcdata.shape[0]
        self.data_sz = len(self.idxes)
        self.seq_len = seq_len

        self.min_pulse_length = min_pulse_length
        self.max_pulse_length = max_pulse_length

    def __getitem__(self, idx):
        ccd = torch.cat([torch.tensor(self.ccdata[idx + n, ...],
                                      dtype=torch.float32).unsqueeze(0) for n in range(self.seq_len)], dim=0)
        csd = self.csdata[idx + self.seq_len, ...]
        if self.spec_sz < self.csdata.shape[0]:
            true_idx = idx % self.spec_sz
            tsd = self.tsdata[true_idx, ...]
            tcd = self.tcdata[true_idx, ...]
        else:
            tsd = self.tsdata[idx, ...]
            tcd = self.tcdata[idx, ...]

        return ccd, tcd, csd, tsd, np.random.randint(self.min_pulse_length, self.max_pulse_length)

    def __len__(self):
        return self.data_sz


class BaseModule(LightningDataModule):
    def __init__(
            self,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            pin_memory: bool = False,
            train_split: int = 1024,
            val_split: int = 32,
            single_example: bool = False,
            device: str = 'cpu',
            **kwargs,
    ):
        super().__init__()

        self.val_dataset = None
        self.train_dataset = None
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = cpu_count() // 2
        self.pin_memory = pin_memory
        self.train_split = train_split
        self.val_split = val_split
        self.single_example = single_example
        self.device = device

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )


class WaveDataModule(BaseModule):
    def __init__(
            self,
            data_path: str,
            latent_dim: int = 50,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            pin_memory: bool = False,
            split: float = 1.,
            single_example: bool = False,
            mu: float = 0.,
            var: float = 1.,
            device: str = 'cpu',
            fft_sz: int = 4096,
            min_pulse_length: int = 1,
            max_pulse_length: int = 2,
            **kwargs,
    ):
        super().__init__(train_batch_size, val_batch_size, pin_memory, 1, 1, single_example, device)

        self.latent_dim = latent_dim
        self.data_dir = data_path
        self.split = split
        self.min_pulse_length = min_pulse_length
        self.max_pulse_length = max_pulse_length
        self.device = device
        self.fft_sz = fft_sz
        self.mu = mu
        self.var = var

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = WaveDataset(self.data_dir, self.latent_dim, self.fft_sz, split=self.split,
                                         single_example=self.single_example, min_pulse_length=self.min_pulse_length,
                                         max_pulse_length=self.max_pulse_length)

        self.val_dataset = WaveDataset(self.data_dir, self.latent_dim, self.fft_sz, split=self.split,
                                       single_example=self.single_example, min_pulse_length=self.min_pulse_length,
                                       max_pulse_length=self.max_pulse_length, is_val=True)


class RCSModule(BaseModule):
    def __init__(
            self,
            dataset_size: int = 256,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            pin_memory: bool = False,
            train_split: int = 1024,
            val_split: int = 32,
            single_example: bool = False,
            device: str = 'cpu',
            **kwargs,
    ):
        super().__init__(train_batch_size, val_batch_size, pin_memory, train_split, val_split, single_example, device)

        self.dataset_size = dataset_size

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = RCSDataset(split=self.train_split)
        self.val_dataset = RCSDataset(split=self.val_split)


class STFTModule(BaseModule):
    def __init__(
            self,
            fs, end_sz, width_min, width_max,
            dataset_size: int = 256,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            pin_memory: bool = False,
            train_split: int = 1024,
            val_split: int = 32,
            single_example: bool = False,
            device: str = 'cpu',
            **kwargs,
    ):
        super().__init__(train_batch_size, val_batch_size, pin_memory, train_split, val_split, single_example, device)

        self.dataset_size = dataset_size
        self.params = (width_min, width_max, fs, end_sz)
        self.end_sz = end_sz
        self.width_min = width_min
        self.width_max = width_max

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = STFTDataset(*self.params)
        self.val_dataset = STFTDataset(*self.params)


class EncoderModule(BaseModule):
    def __init__(
            self,
            fft_len,
            data_path,
            split: float = 1.,
            dataset_size: int = 256,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            pin_memory: bool = False,
            train_split: int = 1024,
            val_split: int = 32,
            single_example: bool = False,
            device: str = 'cpu',
            **kwargs,
    ):
        super().__init__(train_batch_size, val_batch_size, pin_memory, train_split, val_split, single_example, device)

        self.dataset_size = dataset_size
        self.data_path = data_path
        self.fft_len = fft_len
        self.split = split

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = PulseDataset(self.data_path, self.fft_len, self.split)
        self.val_dataset = PulseDataset(self.data_path, self.fft_len, split=1 - self.split if self.split < 1 else 1.,
                                        is_val=True)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open('./vae_config.yaml', 'r') as file:
        try:
            param_dict = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    if param_dict['exp_params']['model_type'] == 'InfoVAE':
        model = InfoVAE(**param_dict['model_params'])
    elif param_dict['exp_params']['model_type'] == 'WAE_MMD':
        model = WAE_MMD(**param_dict['model_params'])
    else:
        model = BetaVAE(**param_dict['model_params'])
    print('Setting up model...')
    model.load_state_dict(torch.load('./model/inference_model.state'))
    transformCovData(model, device, param_dict['dataset_params']['data_path'], param_dict['dataset_params']['mu'],
                     param_dict['dataset_params']['var'])
