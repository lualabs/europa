import os
from typing import List, Optional, Any
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import albumentations as A
import json

LABEL_MAP = {
    "CNH_Aberta": 0,
    "CNH_Frente": 1,
    "CNH_Verso": 2,
    "CPF_Frente": 3,
    "CPF_Verso": 4,
    "RG_Aberto": 5,
    "RG_Frente": 6,
    "RG_Verso": 7,
}

INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


class BIDProtoDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            images: List[str],
            label_paths: List[int],
            transforms: Optional[A.Compose] = None
        ):
        self.images = images
        self.label_paths = label_paths
        self.transforms = transforms
        self.dataset_length = len(self.images)
        self.labels = []
        for label in self.label_paths:
            self.labels.append(self.path2token(label))

    @staticmethod
    def read_dataset(data_dir: str):
        """
        Reads the dataset in the format of the bid-proto dataset.
        Returns a list of paths to images and a list of path to labels.        
        """
        data = []
        labels = []
        # this assumes the data is separated in labels by subdirectories
        subdirs = [subdir for subdir in os.listdir(data_dir) if subdir in LABEL_MAP.keys()]
        for label in subdirs:
            local_path = os.path.join(data_dir, label)
            images = [os.path.join(local_path, file) for file in os.listdir(local_path) if (file.endswith(".jpg") and "gt" not in file)]
            label_paths = [os.path.join(local_path, file) + ".json" for file in images]
            data.extend(images)
            labels.extend(label_paths)
        return data, labels

    @staticmethod
    def from_directory(
        dir_path: str,
        transforms: Optional[A.Compose] = None
    ):
        data, labels = BIDProtoDataset.read_dataset(dir_path)
        return BIDProtoDataset(data, labels, transforms=transforms)

    @staticmethod
    def from_list_of_paths(
        path_list: List[str],
        transforms: Optional[A.Compose] = None
    ):
        images = []
        labels = []
        for path in path_list:
            assert os.path.exists(path), f"Path {path} does not exist"
            assert path.endswith(".jpg"), f"Expected path {path} to be a jpg file"
            labels.append(path + ".json")
            images.append(path)
        return BIDProtoDataset(images, labels, transforms=transforms)

    def path2token(self, label_path: str):
        with open(label_path, "r", encoding="ISO-8859-15") as f:
            label = json.load(f)
            if "document_type" not in label.keys():
                label["document_type"] = os.path.basename(os.path.dirname(label_path))
                assert label["document_type"] in LABEL_MAP.keys(), f"Document type {label['document_type']} not in LABEL_MAP"
            return self.json2token(label)

    def json2token(self, obj: Any, sort_json_key: bool = True):
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    output += (
                        fr"<s_{k}>"
                        + self.json2token(obj[k], sort_json_key)
                        + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/".join(
                [self.json2token(item, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            return obj

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx: int):
        # img = np.array(Image.open(self.images[idx]).convert("RGB")).astype(np.uint8)
        # img = img/255.0
        # if self.transforms is not None:
        #     img = self.transforms(image=img)["image"].to(torch.float32)
        img = Image.open(self.images[idx]).convert("RGB")
        return img, self.labels[idx]