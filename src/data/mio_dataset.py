import os
import tarfile
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class MIOTCDDataset(Dataset):
    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        transform: Callable[[Image.Image], Image.Image] = None,
        download: bool = True,
        val: bool = False,
        preprocess: bool = True,
    ):
        """Dataset for MIO-TCD Dataset.

        Args:
            root (str, optional): Path to root folder. Defaults to "./data".
            train (bool, optional): Use Train or Test Split. Defaults to True.
            transform (Callable[[Image.Image], Image.Image], optional): Pytorch Transform. Defaults to None.
            download (bool, optional): Tries to download dataset. Defaults to True.
            preprocess (bool, optional): Placeholder, as this is not necessary. Defaults to True.
        """
        self.data_name = {}
        self.data_name["full"] = "data"
        self.data_url = {}
        self.data_url[
            "full"
        ] = "https://tcd.miovision.com/static/dataset/MIO-TCD-Classification.tar"

        self.train = train
        self.folder_name = "MIO-TCD-Classification"

        self.root = Path(root)
        self.root = self.root / self.folder_name

        self.transform = transform
        self.val = val

        self.csv = {
            "train": self.root / "train_data.csv",
            "test": self.root / "test_data.csv",
            "val": self.root / "val_data.csv",
        }
        self.data, self.targets = self.get_data()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the
                   target class.
        """
        path = self.data[index]
        target = self.targets[index]
        img = Image.open(path)
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, target, index
    
    def get_data(self):
        if self.train:
            return self.get_data_by_label('train')
        elif self.val:
            return self.get_data_by_label('val')
        else:
            return self.get_data_by_label('test')

    def get_data_by_label(self, label):
        csv = self.csv[label]
        csv = pd.read_csv(csv)
        self.names = list(csv.iloc[:, 1:].columns)

        data = csv["path"].to_list()
        for i, path in enumerate(data):
            data[i] = self.root / path
        targets = csv["target"].to_numpy()

        return data, targets

    def __len__(self):
        return len(self.data)

    def preprocess(self):
        basepath = self.root
        preprocessed_data = []
        for filepath in self.data:
            filepath = Path(filepath)
            filename = str(filepath.name).split(".")[0]
            pardir = str(filepath.parent)

            preprocessed_path = (
                basepath / (pardir + "_preprocessed") / (filename + ".png")
            )
            if not preprocessed_path.is_file():
                img = Image.open(filepath)
                img = img.convert("RGB")
                img = img.resize(
                    (self.prep_size, self.prep_size), resample=Image.BILINEAR
                )
                # import cv2

                # cv2.imread()
                # img = cv2.imread(filepath)

                if not preprocessed_path.parent.exists():
                    preprocessed_path.parent.mkdir()

                img.save(preprocessed_path)
            preprocessed_data.append(preprocessed_path)
        self.data = preprocessed_data
