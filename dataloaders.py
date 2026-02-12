from glob import glob
from typing import List, Optional, Union, Iterator
import os
import yaml
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Sampler
import torch
from itertools import chain
from pathlib import Path
import numpy as np
from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split
import pickle


class BatchListSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        draw_lists (list): List of lists of sample indexes.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, draw_lists: List[List[int]], batch_size: int, drop_last: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        super().__init__()
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
        targets = [o for o in os.listdir(datapath) if 'target_' in o and 'embedding' not in o]
        self.file_idx = np.sort(np.concatenate(
            [list(glob(f'{datapath}/{d}/target_*_*.pt')) for d in targets]
        ))
        sz = len(self.file_idx)
        # file_sizes = [s.shape[0] for s in solo_load]

        if split < 1:
            Xs, Xt, _, _ = train_test_split(np.arange(sz),
                                            np.arange(sz),
                                            test_size=split, random_state=seed)
        else:
            Xt = np.arange(sz)
            Xs = np.arange(sz)
        valids = Xs if is_val else Xt
        self.file_idx = self.file_idx[valids]

    def __getitem__(self, idx):
        sample, label = torch.load(self.file_idx[idx], weights_only=True)
        return sample, label

    def __len__(self):
        return self.file_idx.shape[0]

    def get_data(self):
        solo_data = glob(f'{self.datapath}/target_*_solo.dat')
        return [torch.tensor(np.fromfile(s, dtype=np.float32).reshape((-1, 2, 8192))) for s in solo_data]


class WaveDataset(Dataset):
    def __init__(self, data_path: str, split: float = 1., single_example: bool = False, min_pulse_length: int = 1,
                 max_pulse_length: int = 2, seq_len: int = 32, is_val=False, seed=43):
        assert Path(data_path).is_dir()
        self.datapath = f'{data_path}'

        clutter_spec_files = glob(f'{self.datapath}/*.pic')
        total_seq = 2000
        n_per_file = int(np.round(total_seq / len(clutter_spec_files)))
        clutter_data = []
        target_data = []
        index_data = []

        for clut in np.random.choice(clutter_spec_files, 10):
            with open(clut, 'rb') as f:
                params = pickle.load(f)
                clutter_data.append(params['clutter'])
                target_data.append(params['target'])
                index_data.append(params['t_idx'])

        # Clutter data
        clutter_data = np.concatenate(clutter_data, axis=0)
        target_data = np.concatenate(target_data, axis=0)
        index_data = np.concatenate(index_data).reshape((-1, 1))
        idxes = np.arange(clutter_data.shape[0])
        if split < 1:
            Xsidx, Xtidx, _, _ = train_test_split(idxes, idxes, test_size=split, random_state=seed)
        else:
            Xtidx = idxes
            Xsidx = idxes
        cd_std = clutter_data[Xsidx if is_val else Xtidx].std()
        self.clutter = torch.tensor(clutter_data[Xsidx if is_val else Xtidx] / cd_std, dtype=torch.float32)
        self.target = torch.tensor(target_data[Xsidx if is_val else Xtidx] / cd_std, dtype=torch.float32)
        self.t_idx = torch.tensor(index_data[Xsidx if is_val else Xtidx], dtype=torch.int)

        self.seed = seed
        self.scaling = cd_std
        self.min_pulse_length = min_pulse_length
        self.max_pulse_length = max_pulse_length

    def __getitem__(self, idx):
        # Clutter profile, target+clutter range profile, target range index, pulse length, bandwidth
        return (self.clutter[idx], self.target[idx], self.t_idx[idx],
                np.random.randint(self.min_pulse_length, self.max_pulse_length), np.random.rand() * .6 + .2)

    def __len__(self):
        return self.clutter.shape[0]


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
        self.num_workers = 0 if 'num_workers' not in kwargs else cpu_count() // 2 if kwargs['num_workers'] == -1 else kwargs['num_workers']
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
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            pin_memory: bool = False,
            split: float = 1.,
            single_example: bool = False,
            device: str = 'cpu',
            min_pulse_length: int = 1,
            max_pulse_length: int = 2,
            **kwargs,
    ):
        super().__init__(train_batch_size, val_batch_size, pin_memory, single_example, device)

        self.data_dir = data_path
        self.split = split
        self.min_pulse_length = min_pulse_length
        self.max_pulse_length = max_pulse_length
        self.device = device

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = WaveDataset(self.data_dir, split=self.split,
                                         single_example=self.single_example, min_pulse_length=self.min_pulse_length,
                                         max_pulse_length=self.max_pulse_length)

        self.val_dataset = WaveDataset(self.data_dir, split=self.split,
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
        super().__init__(train_batch_size, val_batch_size, pin_memory, single_example, device, **kwargs)

        self.dataset_size = dataset_size
        self.data_path = data_path
        self.split = split
        self.mu = mu
        self.var = var

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = TargetDataset(self.data_path, self.split, self.single_example, mu=self.mu,
                                          var=self.var)
        self.val_dataset = TargetDataset(self.data_path, split=self.split if self.split < 1 else 1.,
                                         is_val=True, mu=self.mu, var=self.var)
        # self.train_sampler = BatchListSampler(self.train_dataset.file_list, batch_size=self.train_batch_size, drop_last=False)
        # self.val_sampler = BatchListSampler(self.val_dataset.file_list, batch_size=self.val_batch_size, drop_last=False)


class ClutterDataset(Dataset):
    def __init__(self, data_path: str, split: float = 1., is_val: bool = False, seed: int = 7):
        assert Path(data_path).is_dir()
        self.datapath = f'{data_path}'

        clutter_spec_files = glob(f'{self.datapath}/*.pic')
        total_seq = 2000
        n_per_file = int(np.round(total_seq / len(clutter_spec_files)))
        clutter_data = []

        for clut in np.random.choice(clutter_spec_files, 3):
            with open(clut, 'rb') as f:
                params = pickle.load(f)
                clutter_data.append(params['clutter'])

        # Clutter data
        clutter_data = np.concatenate(clutter_data, axis=0)
        if split < 1:
            Xs, Xt, _, _ = train_test_split(clutter_data,
                                            clutter_data,
                                            test_size=split, random_state=seed)
        else:
            Xt = clutter_data
            Xs = clutter_data
        data = Xs if is_val else Xt
        # Minmaxscale
        self.data = torch.tensor(data / data.std(), dtype=torch.float32)
        # self.data = torch.tensor(data * 1e5, dtype=torch.float32)
        # self.data = torch.tensor(data, dtype=torch.float32)

        self.seed = seed

    def __getitem__(self, idx):
        # File contains target range profile, compressed clutter data, target index, and the target range bin
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


class ClutterEncoderModule(BaseModule):
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
            **kwargs,
    ):
        super().__init__(train_batch_size, val_batch_size, pin_memory, single_example, device, **kwargs)

        self.dataset_size = dataset_size
        self.data_path = data_path
        self.split = split

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = ClutterDataset(self.data_path, self.split, self.single_example)
        self.val_dataset = ClutterDataset(self.data_path, split=self.split if self.split < 1 else 1.,
                                         is_val=True)
        # self.train_sampler = BatchListSampler(self.train_dataset.file_list, batch_size=self.train_batch_size, drop_last=False)
        # self.val_sampler = BatchListSampler(self.val_dataset.file_list, batch_size=self.val_batch_size, drop_last=False)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open('./vae_config.yaml', 'r') as file:
        try:
            param_dict = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    