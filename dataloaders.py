from glob import glob
from typing import List, Optional, Sequence, Union
import os
import yaml
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch
from torchvision import transforms
from pathlib import Path
import numpy as np
from multiprocessing import cpu_count
from scipy.ndimage import sobel
from scipy.signal.windows import taylor
from sklearn.model_selection import train_test_split

from models import BaseVAE, InfoVAE, WAE_MMD, BetaVAE


class PulseDataset(Dataset):
    def __init__(self, root_dir, fft_len, split=1., single_example=False, is_val=False, seed=42):
        self.root_dir = root_dir
        self.fft_len = fft_len
        _, self.data = self.get_filedata()
        # Do split
        if split < 1:
            Xs, Xt, _, _ = train_test_split(np.arange(self.data.shape[0]), np.arange(self.data.shape[0]),
                                            test_size=split, random_state=seed)
            self.data = self.data[Xs] if is_val else self.data[Xt]
        if single_example:
            self.data[1:] = self.data[0]
        # Drop the scaling parameters for the encoding
        self.data = torch.tensor(self.data[:, :, :-2], dtype=torch.float32)

    def __getitem__(self, idx):
        img = self.data[idx, ...]
        return img, img

    def __len__(self):
        return self.data.shape[0]

    def get_filedata(self, concat=True):
        # The extra +2 on the end is for scaling parameters mu and std. They are not needed for encoding.
        if not Path(self.root_dir).is_dir():
            return self.root_dir, np.fromfile(self.root_dir, dtype=np.float32).reshape((-1, 2, self.fft_len + 2))
        clutter_files = glob(f'{self.root_dir}/clutter_*.spec')
        dt = [np.fromfile(c, dtype=np.float32).reshape((-1, 2, self.fft_len + 2)) for c in clutter_files]
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


class WaveDataset(ConcatDataset):
    def __init__(self, root_dir: str, latent_dim: int = 50, fft_sz: int = 4096,
                 split: float = 1., single_example: bool = False, min_pulse_length: int = 1, max_pulse_length: int = 2,
                 seq_len: int = 32, is_val=False, seed=42):
        assert Path(root_dir).is_dir()

        clutter_spec_files = glob(f'{root_dir}/clutter_*.spec')
        target_spec_files = glob(f'{root_dir}/target_*.spec')
        clutter_enc_files = glob(f'{root_dir}/clutter_*.enc')
        target_enc_files = glob(f'{root_dir}/target_*.enc')

        # Arrange files into their pairs
        pair_dict = {}
        clutter_size = []
        file_idx = 0
        for fl, tp in [(clutter_spec_files, 'cs'), (clutter_enc_files, 'cc'),
                       (target_enc_files, 'tc'), (target_spec_files, 'ts')]:
            for cs in fl:
                path_stem = '_'.join(Path(cs).stem.split('_')[1:])
                if path_stem in pair_dict:
                    pair_dict[path_stem][tp] = cs
                else:
                    pair_dict[path_stem] = {tp: cs}
                if tp == 'cc':
                    csz = int((os.stat(cs).st_size / (4 * latent_dim) - 255) *
                              (1 - split if is_val and split < 1 else split))
                    clutter_size = clutter_size + [(file_idx, n) for n in range(csz)]
                    file_idx += 1
                    pair_dict[path_stem]['size'] = csz

        pairs = [l for _, l in pair_dict.items()]
        datasets = [WaveFileDataset(p, latent_dim, fft_sz, split, single_example, min_pulse_length, max_pulse_length,
                                    seq_len, is_val, seed) for p in pairs]
        super().__init__(datasets)


class WaveFileDataset(Dataset):
    def __init__(self, files: dict, latent_dim: int = 50, fft_sz: int = 4096,
                 split: float = 1., single_example: bool = False, min_pulse_length: int = 1,
                 max_pulse_length: int = 2, seq_len: int = 32, is_val=False, seed=42):

        # Clutter data
        tmp_cs = np.fromfile(files['cs'], dtype=np.float32).reshape((-1, 2, fft_sz + 2))
        # Scale appropriately
        tmp_cs = tmp_cs[:, :, :fft_sz]
        tmp_cc = np.fromfile(files['cc'], dtype=np.float32).reshape((-1, latent_dim))[:-255, :]

        # Target data
        tmp_ts = np.fromfile(files['ts'], dtype=np.float32).reshape((-1, 2, fft_sz + 2))
        # Scale appropriately
        tmp_ts = tmp_ts[:, :, :fft_sz]
        tmp_tc = np.fromfile(files['tc'], dtype=np.float32).reshape((-1, latent_dim))
        if split < 1:
            Xs, Xt, _, _ = train_test_split(np.arange(tmp_cs.shape[0]),
                                            np.arange(tmp_cs.shape[0]),
                                            test_size=split, random_state=seed)
        else:
            Xt = np.arange(tmp_cs.shape[0])
            Xs = np.arange(tmp_cs.shape[0])
        self.ccdata = tmp_cc[Xs] if is_val else tmp_cc[Xt]
        self.csdata = torch.tensor(tmp_cs[Xs]) if is_val else torch.tensor(tmp_cs[Xt])
        self.tcdata = torch.tensor(tmp_tc[:self.ccdata.shape[0]])
        self.tsdata = torch.tensor(tmp_ts[:self.csdata.shape[0]])
        if single_example:
            self.ccdata[1:] = self.ccdata[0]
            self.tcdata[1:] = self.tcdata[0]
            self.csdata[1:] = self.csdata[0]
            self.tsdata[1:] = self.tsdata[0]

        self.seq_len = seq_len
        self.data_sz = self.ccdata.shape[0]
        self.min_pulse_length = min_pulse_length
        self.max_pulse_length = max_pulse_length

    def __getitem__(self, idx):
        ccd = torch.cat([torch.tensor(self.ccdata[idx + n, ...],
                                      dtype=torch.float32).unsqueeze(0) for n in
                         range(min(self.seq_len, self.ccdata.shape[0] - idx))], dim=0)
        csd = self.csdata[idx, ...]
        tsd = self.tsdata[idx, ...]
        tcd = self.tcdata[idx, ...]

        return ccd, tcd, csd, tsd, np.random.randint(self.min_pulse_length, self.max_pulse_length), np.random.rand() * 1e9 + 400e6

    def __len__(self):
        return self.data_sz - self.seq_len


def collate_fun(batch):
    return (torch.stack([ccd for ccd, _, _, _, _, _ in batch]), torch.stack([tcd for _, tcd, _, _, _, _ in batch]),
            torch.stack([csd for _, _, csd, _, _, _ in batch]), torch.stack([tsd for _, _, _, tsd, _, _ in batch]),
            torch.tensor([pl for _, _, _, _, pl, _ in batch]), torch.tensor([bw for _, _, _, _, _, bw in batch]))


class BaseModule(LightningDataModule):
    def __init__(
            self,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            pin_memory: bool = False,
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
            collate_fn=collate_fun,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            collate_fn=collate_fun,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            collate_fn=collate_fun,
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
        super().__init__(train_batch_size, val_batch_size, pin_memory, single_example, device)

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
            single_example: bool = False,
            device: str = 'cpu',
            **kwargs,
    ):
        super().__init__(train_batch_size, val_batch_size, pin_memory, single_example, device)

        self.dataset_size = dataset_size

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = RCSDataset(split=self.train_split)
        self.val_dataset = RCSDataset(split=self.val_split)


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
            single_example: bool = False,
            device: str = 'cpu',
            **kwargs,
    ):
        super().__init__(train_batch_size, val_batch_size, pin_memory, single_example, device)

        self.dataset_size = dataset_size
        self.data_path = data_path
        self.fft_len = fft_len
        self.split = split

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = PulseDataset(self.data_path, self.fft_len, self.split, self.single_example)
        self.val_dataset = PulseDataset(self.data_path, self.fft_len, split=1 - self.split if self.split < 1 else 1.,
                                        single_example=self.single_example, is_val=True)


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
