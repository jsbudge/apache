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


class WaveDataset(Dataset):
    def __init__(self, data_path: str, split: float = 1., single_example: bool = False, min_pulse_length: int = 1,
                 max_pulse_length: int = 2, std: tuple[float, float] = (1.0, 1.0), is_val=False, seed=43):
        assert Path(data_path).is_dir()
        self.datapath = f'{data_path}'

        clutter_spec_files = glob(f'{self.datapath}/*-training.pic')
        embeddings_file = f'{self.datapath}/embeddings.pic'
        total_seq = 2000
        n_per_file = int(np.round(total_seq / len(clutter_spec_files)))
        clutter_data = []
        target_data = []
        both_data = []
        index_data = []
        nsam = []

        if single_example:
            for clut in clutter_spec_files[:1]:
                with open(clut, 'rb') as f:
                    params = pickle.load(f)
                    clutter_data.append(params['clutter'])
                    target_data.append(params['target'])
                    both_data.append(params['both'])
                    index_data.append(params['t_idx'])
                    nsam.append(params['build']['nsam'])
        else:
            for clut in np.random.choice(clutter_spec_files, 10):
                with open(clut, 'rb') as f:
                    params = pickle.load(f)
                    clutter_data.append(params['clutter'])
                    target_data.append(params['target'])
                    both_data.append(params['both'])
                    index_data.append(params['t_idx'])
                    nsam.append(params['build']['nsam'])

        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)


        # Clutter data
        clutter_data = np.concatenate(clutter_data, axis=0)
        target_data = np.concatenate(target_data, axis=0)
        both_data = np.concatenate(both_data, axis=0)
        index_data = np.concatenate(index_data).reshape((-1, 1))
        idxes = np.arange(clutter_data.shape[0])
        if split < 1:
            Xsidx, Xtidx, _, _ = train_test_split(idxes, idxes, test_size=split, random_state=seed)
        else:
            Xtidx = idxes
            Xsidx = idxes
        ci = clutter_data[Xsidx if is_val else Xtidx] / std[0]
        tloc = target_data[Xsidx if is_val else Xtidx]
        ti = np.stack([embeddings[7] for _ in ci])
        # ti = np.zeros((ci.shape[0], 1, 25))
        # ti[..., 7] = 1.
        bi = both_data[Xsidx if is_val else Xtidx] / std[0]
        ii = index_data[Xsidx if is_val else Xtidx]
        self.clutter = torch.tensor(ci, dtype=torch.float32)
        self.target = torch.tensor(ti, dtype=torch.float32).unsqueeze(1)
        self.both = torch.tensor(bi, dtype=torch.float32)
        self.t_idx = torch.tensor(ii, dtype=torch.int)

        self.samples = np.array(nsam)
        correct = np.logical_or(tloc[..., 0, :] > 0, tloc[..., 1, :] > 0)
        for c in correct:
            _, locs = np.where(c)
            c[:, locs.min():locs.max()] = 1.
        self.truth = torch.tensor(correct, dtype=torch.float32)

        self.seed = seed
        self.scaling = std
        self.min_pulse_length = min_pulse_length
        self.max_pulse_length = max_pulse_length
        self.is_single = single_example

    def __getitem__(self, idx):
        # Clutter profile, target profile, target+clutter range profile, target range index, pulse length, bandwidth
        if self.is_single:
            return self.clutter[0], self.target[0], self.both[0], self.t_idx[0], 2669, .35, self.samples[0], self.truth[0]
        else:
            return self.clutter[idx], self.target[idx], self.both[idx], self.t_idx[idx],  2669, .35, self.samples[0], self.truth[idx]



    def __len__(self):
        return self.clutter.shape[0]


class TargetDataset(Dataset):
    def __init__(self, data_path: str, split: float = 1., single_example: bool = False, min_pulse_length: int = 1,
                 max_pulse_length: int = 2, std: tuple[float, float] = (1.0, 1.0), is_val=False, seed=43):
        assert Path(data_path).is_dir()
        self.datapath = f'{data_path}'

        clutter_spec_files = glob(f'{self.datapath}/*-embedding.pic')
        target_data = []
        target_num = []

        for clut in clutter_spec_files:
            with open(clut, 'rb') as f:
                params = pickle.load(f)
                target_data.append(np.stack(params['target']).swapaxes(-1, -2))
                target_num.append(np.stack([params['target_id'] for _ in range(len(params['target']))]))

        # Standardize the size to fit all (10000)
        anchor_data = np.concatenate(target_data, axis=0) / std[0]
        positive_data = np.concatenate([np.roll(t, 1, axis=0) for t in target_data], axis=0) / std[0]
        negative_data = np.roll(anchor_data, target_data[0].shape[0], axis=0) / std[0]
        idxes = np.arange(anchor_data.shape[0])
        tnums = np.concatenate(target_num)
        if split < 1:
            Xsidx, Xtidx, _, _ = train_test_split(idxes, idxes, test_size=split, random_state=seed)
        else:
            Xtidx = idxes
            Xsidx = idxes
        self.anchor = torch.tensor(anchor_data[Xsidx if is_val else Xtidx], dtype=torch.float32)
        self.positive = torch.tensor(positive_data[Xsidx if is_val else Xtidx], dtype=torch.float32)
        self.negative = torch.tensor(negative_data[Xsidx if is_val else Xtidx], dtype=torch.float32)
        self.t_idx = torch.tensor(tnums[Xsidx if is_val else Xtidx], dtype=torch.float32)

        self.seed = seed
        self.scaling = std
        self.is_single = single_example

    def __getitem__(self, idx):
        # Clutter profile, target profile, target+clutter range profile, target range index, pulse length, bandwidth
        if self.is_single:
            return self.anchor[0], self.positive[0], self.negative[0], self.t_idx[0]
        else:
            return self.anchor[idx], self.positive[idx], self.negative[idx], self.t_idx[idx]



    def __len__(self):
        return self.anchor.shape[0]


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
            std: float = 1.0,
            **kwargs,
    ):
        super().__init__(train_batch_size, val_batch_size, pin_memory, single_example, device)

        self.data_dir = data_path
        self.split = split
        self.min_pulse_length = min_pulse_length
        self.max_pulse_length = max_pulse_length
        self.device = device
        self.std = std

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = WaveDataset(self.data_dir, split=self.split,
                                         single_example=self.single_example, min_pulse_length=self.min_pulse_length,
                                         max_pulse_length=self.max_pulse_length, std=self.std)

        self.val_dataset = WaveDataset(self.data_dir, split=self.split,
                                       single_example=self.single_example, min_pulse_length=self.min_pulse_length,
                                       max_pulse_length=self.max_pulse_length, std=self.std, is_val=True)
        
        
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
            std: tuple[float, float] = (1., 1.),
            **kwargs,
    ):
        super().__init__(train_batch_size, val_batch_size, pin_memory, single_example, device, **kwargs)

        self.dataset_size = dataset_size
        self.data_path = data_path
        self.split = split
        self.std = std

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = TargetDataset(self.data_path, self.split, self.single_example, std=self.std)
        self.val_dataset = TargetDataset(self.data_path, split=self.split if self.split < 1 else 1.,
                                         is_val=True, std=self.std)
        # self.train_sampler = BatchListSampler(self.train_dataset.file_list, batch_size=self.train_batch_size, drop_last=False)
        # self.val_sampler = BatchListSampler(self.val_dataset.file_list, batch_size=self.val_batch_size, drop_last=False)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open('./vae_config.yaml', 'r') as file:
        try:
            param_dict = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    