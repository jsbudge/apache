from glob import glob
from typing import List, Optional, Union, Iterator
import os
import yaml
from click.core import batch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, ConcatDataset, BatchSampler, SubsetRandomSampler, Sampler
import torch
from itertools import chain
from pathlib import Path
import numpy as np
from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split


class BatchListSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(
        self,
        draw_lists: List[List[int]],
        batch_size: int,
        drop_last: bool,
    ) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                f"batch_size should be a positive integer value, but got batch_size={batch_size}"
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                f"drop_last should be a boolean value, but got drop_last={drop_last}"
            )
        self.samplers = [SubsetRandomSampler(l) for l in draw_lists]
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        sampler_iters = [iter(s) for s in self.samplers]
        while True:
            try:
                yield list(chain(*[[(next(s), idx) for idx, s in enumerate(sampler_iters)] for _ in range(self.batch_size // len(sampler_iters))]))[:self.batch_size]
            except StopIteration:
                break

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        return (
            sum(len(s) for s in self.samplers) + self.batch_size - 1
        ) // self.batch_size


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


class TargetDataset(Dataset):
    def __init__(self, datapath: str = './data/target_tensors', split: float = 1., is_val: bool = False, mu: float = .01,
                 var: float = 4.9, seed: int = 7):
        # Load in data
        self.datapath = datapath
        targets = os.listdir(datapath)
        file_sizes = [0 for _ in targets]
        for d in targets:
            ntarg = int(d[7:])
            nfiles = glob(f'{datapath}/{d}/target_*_*.pt')
            file_sizes[ntarg] = len(nfiles)
        file_idx = np.concatenate([np.ones(s) * i for i, s in enumerate(file_sizes)])
        sz = min(file_sizes)
        # file_sizes = [s.shape[0] for s in solo_load]

        if split < 1:
            Xs, Xt, _, _ = train_test_split(np.arange(sz),
                                            np.arange(sz),
                                            test_size=split, random_state=seed)
        else:
            Xt = np.arange(sz)
            Xs = np.arange(sz)
        valids = np.concatenate([(Xs if is_val else Xt) + sum(file_sizes[:f]) for f in range(len(file_sizes))])
        self.file_idx = file_idx[valids]
        self.file_list = [list(valids[np.where(self.file_idx == n)[0]].astype(int)) for n in range(len(file_sizes))]

    def __getitem__(self, idx):
        sample, label = torch.load(f'{self.datapath}/target_{idx[1]}/target_{idx[1]}_{idx[0]}.pt')
        return sample, label

    def __len__(self):
        return self.file_idx.shape[0]

    def get_data(self):
        solo_data = glob(f'{self.datapath}/target_*_solo.dat')
        return [torch.tensor(np.fromfile(s, dtype=np.float32).reshape((-1, 2, 8192))) for s in solo_data]


class WaveDataset(Dataset):
    def __init__(self, root_dir: str, target_latent_dim: int = 1024, clutter_latent_dim: int = 512, fft_sz: int = 4096,
                 split: float = 1., single_example: bool = False, min_pulse_length: int = 1, max_pulse_length: int = 2,
                 seq_len: int = 32, is_val=False, seed=42):
        assert Path(root_dir).is_dir()
        self.datapath = f'{root_dir}/target_tensors/clutter_tensors'

        clutter_spec_files = glob(f'{self.datapath}/tc_*.pt')
        patterns = torch.load(f'{root_dir}/target_embedding_means.pt')

        # Clutter data
        file_idxes = np.arange(len(clutter_spec_files) - seq_len)
        if split < 1:
            Xs, Xt, _, _ = train_test_split(file_idxes,
                                            file_idxes,
                                            test_size=split, random_state=seed)
        else:
            Xt = file_idxes
            Xs = file_idxes
        self.idxes = Xs if is_val else Xt
        self.single = single_example
        self.patterns = patterns[0]

        self.seq_len = seq_len
        self.min_pulse_length = min_pulse_length
        self.max_pulse_length = max_pulse_length
        self.seed = seed

    def __getitem__(self, idx):
        data = [torch.load(f'{self.datapath}/tc_{n}.pt') for n in range(idx, idx + self.seq_len)]
        return (torch.cat([c.unsqueeze(0) for c, t, i in data], dim=0), data[-1][1], self.patterns[data[-1][2]].clone().detach(),
                np.random.randint(self.min_pulse_length, self.max_pulse_length))

    def __len__(self):
        return self.idxes.shape[0]


class WaveFileDataset(Dataset):
    def __init__(self, files: dict, latent_dim: int = 50, fft_sz: int = 4096,
                 split: float = 1., single_example: bool = False, min_pulse_length: int = 1,
                 max_pulse_length: int = 2, seq_len: int = 32, is_val: bool = False,
                 seed: int = 42):

        # Clutter data
        tmp_cs = np.fromfile(files['cs'], dtype=np.float32).reshape((-1, 2, fft_sz + 2))
        # Scale appropriately
        tmp_cs = tmp_cs[:, :, :fft_sz]
        tmp_cc = np.fromfile(files['cc'], dtype=np.float32).reshape((-1, latent_dim))

        # Target data
        tmp_ts = np.fromfile(files['ts'], dtype=np.float32).reshape((-1, 2, fft_sz + 2))
        # Scale appropriately
        tmp_ts = tmp_ts[:, :, :fft_sz]
        self.tcdata = torch.tensor(files['tc'], dtype=torch.float32)
        if split < 1:
            Xs, Xt, _, _ = train_test_split(np.arange(tmp_cs.shape[0]),
                                            np.arange(tmp_cs.shape[0]),
                                            test_size=split, random_state=seed)
        else:
            Xt = np.arange(tmp_cs.shape[0])
            Xs = np.arange(tmp_cs.shape[0])
        self.ccdata = tmp_cc[Xs] if is_val else tmp_cc[Xt]
        self.csdata = torch.tensor(tmp_cs[Xs]) if is_val else torch.tensor(tmp_cs[Xt])
        self.tsdata = torch.tensor(tmp_ts[:self.csdata.shape[0]])
        if single_example:
            self.ccdata[1:] = self.ccdata[0]
            self.csdata[1:] = self.csdata[0]
            self.tsdata[1:] = self.tsdata[0]

        self.seq_len = seq_len
        self.data_sz = self.tsdata.shape[0]
        self.min_pulse_length = min_pulse_length
        self.max_pulse_length = max_pulse_length
        self.seed = seed

    def __getitem__(self, idx):
        ccd = torch.cat([torch.tensor(self.ccdata[idx + n, ...],
                                      dtype=torch.float32).unsqueeze(0) for n in
                         range(min(self.seq_len, self.tsdata.shape[0] - idx))], dim=0)
        csd = self.csdata[idx, ...]
        tsd = self.tsdata[idx, ...]

        return (ccd, self.tcdata, csd, tsd, np.random.randint(self.min_pulse_length, self.max_pulse_length),
                np.random.rand() * 1e9 + 400e6)

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
            collate: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.val_dataset = None
        self.train_dataset = None
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = 0  # cpu_count() // 2
        self.pin_memory = pin_memory
        self.single_example = single_example
        self.device = device
        self.collate = collate
        self.train_sampler = None
        self.val_sampler = None

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        if self.collate:
            return DataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size if self.train_sampler is None else 1,
                num_workers=self.num_workers,
                batch_sampler=self.train_sampler,
                pin_memory=self.pin_memory,
                collate_fn=collate_fun,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size if self.train_sampler is None else 1,
                num_workers=self.num_workers,
                batch_sampler=self.train_sampler,
                pin_memory=self.pin_memory,
            )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if self.collate:
            return DataLoader(
                self.val_dataset,
                batch_size=self.val_batch_size if self.val_sampler is None else 1,
                num_workers=self.num_workers,
                batch_sampler=self.val_sampler,
                pin_memory=self.pin_memory,
                collate_fn=collate_fun,
            )
        else:
            return DataLoader(
                self.val_dataset,
                batch_size=self.val_batch_size if self.val_sampler is None else 1,
                num_workers=self.num_workers,
                batch_sampler=self.val_sampler,
                pin_memory=self.pin_memory,
            )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size if self.val_sampler is None else 1,
            num_workers=self.num_workers,
            batch_sampler=self.val_sampler,
            pin_memory=self.pin_memory,
            collate_fn=collate_fun,
        )


class WaveDataModule(BaseModule):
    def __init__(
            self,
            data_path: str,
            clutter_latent_dim: int = 50,
            target_latent_dim: int = 1024,
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

        self.clutter_latent_dim = clutter_latent_dim
        self.target_latent_dim = target_latent_dim
        self.data_dir = data_path
        self.split = split
        self.min_pulse_length = min_pulse_length
        self.max_pulse_length = max_pulse_length
        self.device = device
        self.fft_sz = fft_sz
        self.mu = mu
        self.var = var

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = WaveDataset(self.data_dir, clutter_latent_dim=self.clutter_latent_dim,
                                         target_latent_dim=self.target_latent_dim, fft_sz=self.fft_sz, split=self.split,
                                         single_example=self.single_example, min_pulse_length=self.min_pulse_length,
                                         max_pulse_length=self.max_pulse_length)

        self.val_dataset = WaveDataset(self.data_dir, clutter_latent_dim=self.clutter_latent_dim,
                                       target_latent_dim=self.target_latent_dim, fft_sz=self.fft_sz, split=self.split,
                                       single_example=self.single_example, min_pulse_length=self.min_pulse_length,
                                       max_pulse_length=self.max_pulse_length, is_val=True)


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
            mu: float = .01,
            var: float = 4.9,
            **kwargs,
    ):
        super().__init__(train_batch_size, val_batch_size, pin_memory, single_example, device)

        self.dataset_size = dataset_size
        self.data_path = data_path
        self.fft_len = fft_len
        self.split = split
        self.mu = mu
        self.var = var

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = PulseDataset(self.data_path, self.fft_len, self.split, self.single_example)
        self.val_dataset = PulseDataset(self.data_path, self.fft_len, split=1 - self.split if self.split < 1 else 1.,
                                        single_example=self.single_example, is_val=True)
        
        
class TargetEncoderModule(BaseModule):
    def __init__(
            self,
            data_path,
            split: float = 1.,
            dataset_size: int = 256,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            pin_memory: bool = False,
            single_example: bool = False,
            device: str = 'cpu',
            mu: float = .01,
            var: float = 4.9,
            **kwargs,
    ):
        super().__init__(train_batch_size, val_batch_size, pin_memory, single_example, device)

        self.dataset_size = dataset_size
        self.data_path = data_path
        self.split = split
        self.mu = mu
        self.var = var

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = TargetDataset(self.data_path, self.split, self.single_example, mu=self.mu,
                                          var=self.var)
        self.val_dataset = TargetDataset(self.data_path, split=1 - self.split if self.split < 1 else 1.,
                                         is_val=True, mu=self.mu, var=self.var)
        self.train_sampler = BatchListSampler(self.train_dataset.file_list, batch_size=self.train_batch_size, drop_last=False)
        self.val_sampler = BatchListSampler(self.val_dataset.file_list, batch_size=self.val_batch_size, drop_last=False)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open('./vae_config.yaml', 'r') as file:
        try:
            param_dict = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    