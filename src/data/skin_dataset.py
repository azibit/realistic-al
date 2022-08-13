import os
import zipfile

import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from urllib.request import urlretrieve


class AbstractISIC(Dataset):
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

        return img, target

    def __len__(self):
        return len(self.data)

    def preprocess(self):
        from pathlib import Path

        basepath = Path(self.root)
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
                if not preprocessed_path.parent.exists():
                    preprocessed_path.parent.mkdir()

                img.save(preprocessed_path)
            preprocessed_data.append(preprocessed_path)
        self.data = preprocessed_data

    def download(self):
        from utils.download_url import download_url

        mode = "train" if self.train else "test"
        if mode in self.data_url:
            if not os.path.exists(os.path.join(self.root, self.data_name[mode])):
                print(
                    "Downloading and extracting {} skin lesion data...".format(
                        self.folder_name
                    )
                )
                save_path = os.path.join(self.root, self.data_name[mode] + ".zip")
                os.makedirs(os.path.join(self.root), exist_ok=True)

                download_url(self.data_url[mode], save_path)

                zip_ref = zipfile.ZipFile(save_path, "r")
                zip_ref.extractall(self.root)
                zip_ref.close()

                os.remove(save_path)
                print("Finished donwload and extraction")

        if mode in self.csv_url:
            if not os.path.exists(os.path.join(self.root, self.csv[mode])):
                print(
                    "Downloading and extracting {} skin lesion labels...".format(
                        self.folder_name
                    )
                )
                save_path = os.path.join(self.root, self.csv[mode])
                urlretrieve(self.csv_url[mode], save_path)
                print("Finished donwload and extraction")


class ISIC2019(AbstractISIC):
    def __init__(
        self, root="./data", train=True, transform=None, download=True, preprocess=True
    ):
        """Dataset for ISIC-2019 Dataset.
        Link: https://challenge.isic-archive.com/data/#2019

        Class Count: 0: 4522, 1: 12875,  2: 3323, 3: 867, 4: 2624, 5: 239, 6: 253, 7: 628


        Args:
            root (str, optional): _description_. Defaults to "./data".
            train (bool, optional): _description_. Defaults to True.
            transform (_type_, optional): _description_. Defaults to None.
            download (bool, optional): _description_. Defaults to True.
            preprocess (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        self.folder_name = "ISIC-2019"

        self.csv = {}
        self.csv["train"] = "ISIC_2019_Training_GroundTruth.csv"
        self.data_name = {}
        self.data_name["train"] = "ISIC_2019_Training_Input"

        self.data_url = {
            "train": "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip",
        }

        self.csv_url = {
            "train": "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv",
        }

        self.prep_size = 300  # potentially this could be changed for the test set!

        self.train = train
        self.root = os.path.join(root, self.folder_name)
        if download:
            self.download()
        self.transform = transform

        self.data, self.targets = self.get_data()
        if preprocess:
            self.preprocess()

    def get_data(self):
        csv_name = self.csv["train"]
        csv = os.path.join(self.root, csv_name)
        csv = pd.read_csv(csv)
        self.names = list(csv.iloc[:, 1:].columns)

        data = []
        targets = []
        for filename in csv.loc[:, "image"]:
            img_folder = self.data_name["train"]
            data.append(os.path.join(self.root, img_folder, filename + ".jpg"))

        for label in csv.iloc[:, 1:].values:
            targets.append(np.argmax(label))
        targets = np.array(targets)

        return data, targets


class ISIC2016(AbstractISIC):
    """Skin Lesion"""

    def __init__(
        self, root="./data", train=True, transform=None, download=True, preprocess=True,
    ):
        super().__init__()
        self.folder_name = "ISIC-2016"

        self.csv = {}
        self.csv["test"] = "ISBI2016_ISIC_Part3_Test_GroundTruth.csv"
        self.csv["train"] = "ISBI2016_ISIC_Part3_Training_GroundTruth.csv"
        self.data_name = {}
        self.data_name["train"] = "ISBI2016_ISIC_Part3_Training_Data"
        self.data_name["test"] = "ISBI2016_ISIC_Part3_Test_Data"

        self.data_url = {
            "train": "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_Data.zip",
            "test": "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Test_Data.zip",
        }

        self.csv_url = {
            "train": "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_GroundTruth.csv",
            "test": "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Test_GroundTruth.csv",
        }

        self.prep_size = 300  # potentially this could be changed for the test set!
        self.classes_name = ["DIA"]
        self.classes = list(range(len(self.classes_name)))

        self.train = train
        self.root = os.path.join(root, self.folder_name)
        if download:
            self.download()
        self.transform = transform

        self.data, self.targets = self.get_data()
        if preprocess:
            self.preprocess()

    def get_data(self):
        if self.train:
            csv_name = self.csv["train"]
        else:
            csv_name = self.csv["test"]
        csv = os.path.join(self.root, csv_name)
        csvfile = pd.read_csv(csv, header=None)

        raw_data = csvfile.values
        data = []
        targets = []
        for filename, label in raw_data:
            # data.append(os.path.join(self.root, "ISIC2018_Task3_Training_Input", path))
            if self.train:
                img_folder = self.data_name["train"]
            else:
                img_folder = self.data_name["test"]
            data.append(os.path.join(self.root, img_folder, filename + ".jpg"))
            if not isinstance(label, (int, float)):
                label = 0 if label == "benign" else 1
            else:
                label = int(label)
            targets.append(label)
        targets = np.array(targets)

        return data, targets


# def print_dataset(dataset, print_time):
#     print(len(dataset))
#     from collections import Counter

#     counter = Counter()
#     labels = []
#     # for index, (img, label) in enumerate(dataset):
#     # if index % print_time == 0:
#     #     print(img.size(), label)
#     for index, label in enumerate(dataset.targets):
#         labels.append(label)
#     counter.update(labels)
#     print(counter)


if __name__ == "__main__":
    import os

    dataroot = os.getenv("DATA_ROOT")
    # dataset = ISIC2016(root=dataroot, train=True, preprocess=False, download=False)

    # print(len(dataset))
    from tqdm import tqdm

    # for _ in tqdm(dataset):
    #     pass

    dataset = ISIC2016(root=dataroot, train=False, preprocess=True, download=False)
    print(dataset.targets)

    print(len(dataset))
    from tqdm import tqdm

    for _ in tqdm(dataset):
        pass

