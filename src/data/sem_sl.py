from torch.utils.data import DataLoader

from copy import deepcopy

from .utils import (
    ConcatDataloader,
    TransformFixMatch,
    activesubset_from_subset,
    seed_worker,
)

from .data import TorchVisionDM
from .random_fixed_length_sampler import RandomFixedLengthSampler

from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from torch.utils.data import Subset
import numpy as np
from .transformations import get_transform


def fixmatch_train_dataloader(dm: TorchVisionDM, mu: int):
    """Returns the Concatenated Daloader used for FixMatch Training given the datamodule"""
    train_pool = activesubset_from_subset(dm.train_set.pool._dataset)
    train_pool.transform = TransformFixMatch(mean=dm.mean, std=dm.std)

    # Keep amount of workers fixed for training.
    workers_sup = max(2, (dm.num_workers) // (mu + 1))
    workers_sem = dm.num_workers - workers_sup

    sem_loader = DataLoader(
        train_pool,
        batch_size=dm.batch_size * mu,
        num_workers=workers_sem,
        shuffle=True,
        pin_memory=dm.pin_memory,
        drop_last=True,
        worker_init_fn=seed_worker,
    )

    # Increase size of small datasets to make use of multiple workers
    # and limit the amount of dataloader reinits in concat dataloader
    sample_trainset = len(dm.train_set)
    if sample_trainset // dm.batch_size < len(sem_loader):
        resample_size = sample_trainset * (
            len(sem_loader) // max(1, sample_trainset // dm.batch_size)
        )
        resample_size = min(6400, resample_size)
        sup_loader = DataLoader(
            dm.train_set,
            batch_size=dm.batch_size,
            sampler=RandomFixedLengthSampler(dm.train_set, resample_size),
            num_workers=dm.num_workers,
            pin_memory=dm.pin_memory,
            drop_last=dm.drop_last,
            worker_init_fn=seed_worker,
        )
    else:
        sup_loader = DataLoader(
            dm.train_set,
            batch_size=dm.batch_size,
            shuffle=dm.shuffle,
            num_workers=dm.num_workers,
            pin_memory=dm.pin_memory,
            drop_last=dm.drop_last,
            worker_init_fn=seed_worker,
        )

    return ConcatDataloader(
        sup_loader,
        sem_loader,
    )


def wrap_fixmatch_train_dataloader(dm: TorchVisionDM, mu: int):
    """Returns the executable function which allows to obtain the fixmatch train_dataloaders."""

    def train_dataloader():
        return fixmatch_train_dataloader(dm, mu)

    return train_dataloader
