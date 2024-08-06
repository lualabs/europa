
from typing import Optional, List, Union, Tuple
import lightning as pl
import albumentations as A
from torch.utils.data import DataLoader, random_split
import torch
from .utils import train_collate_fn, eval_collate_fn
from .bid_dataset import BIDProtoDataset
from europa.config import DataConfig


class BIDProtoDataModule(pl.LightningDataModule):
    def __init__(
            self,
            cfg: DataConfig
        ):
        super(BIDProtoDataModule, self).__init__()
        self.data_dir = cfg.data_dir
        self.test_data_dir = cfg.test_data_dir
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.val_split = cfg.val_split
        self.transforms = cfg.transforms
        self.val_transforms = cfg.val_transforms if cfg.val_transforms is not None else cfg.transforms 
        self.test_transforms = cfg.test_transforms if cfg.test_transforms is not None else cfg.transforms
        self.seed = cfg.seed
        self.train_collate_fn = lambda x: train_collate_fn(processor=self.processor, max_length=cfg.max_length, examples=x)
        self.eval_collate_fn = lambda x: eval_collate_fn(processor=self.processor, examples=x)
        self.bid_train = None
        self.bid_val = None
        self.bid_test = None
        self.processor = cfg.load_processor()
        

    def load_data(
            self,
            data_dir: Union[str, List[str]],
            transforms: Optional[A.Compose] = None,
            **kwargs
        ):
        # if the user passes a string, we assume it's the
        # root directory of all subdirs of images
        if transforms is None:
            transforms = self.transforms
        if isinstance(data_dir, str):
            return BIDProtoDataset.from_directory(
                dir_path=data_dir,
                transforms=transforms,
                **kwargs
            )
        return BIDProtoDataset.from_list_of_paths(
            path_list=self.data_dir,
            transforms=transforms,
            **kwargs
        )
    
    def load_and_split_data(
            self,
            data: List[str],
            **kwargs,
    ) -> Tuple[BIDProtoDataset, BIDProtoDataset]:
        # random split the list into val and train
        train_paths, val_paths =  random_split(
            data,
            [
                int(len(data) * (1 - self.val_split)),
                len(data) - int(len(data) * (1 - self.val_split))
            ],
            generator=torch.Generator().manual_seed(self.seed)
        )
        train = BIDProtoDataset.from_list_of_paths(
            path_list=[data[d] for d in train_paths.indices],
            transforms=self.transforms,
            **kwargs,
        )
        val = BIDProtoDataset.from_list_of_paths(
            path_list=[data[d] for d in val_paths.indices],
            transforms=self.val_transforms,
            **kwargs,
        )
        return train, val
    
    def load_and_split_data_from_dir(
            self,
            data_dir: str,
            **kwargs
        ) -> Tuple[BIDProtoDataset, BIDProtoDataset]:
        data, _ = BIDProtoDataset.read_dataset(data_dir)
        return self.load_and_split_data(data=data_dir, **kwargs)
    
    def _setup_collates(self):
        if self.train_collate_fn is None:
            self.train_collate_fn = lambda x: train_collate_fn(processor=self.processor, examples=x)
        if self.eval_collate_fn is None:
            self.eval_collate_fn = lambda x: eval_collate_fn(processor=self.processor, examples=x)

    def setup(self, stage: str):
        self._setup_collates()
        if isinstance(self.data_dir, str):
            self.bid_train, self.bid_val = self.load_and_split_data_from_dir(self.data_dir)
        elif isinstance(self.data_dir, list):
            self.bid_train, self.bid_val = self.load_and_split_data(self.data_dir)
        else:
            raise ValueError(
                f"Expected data_dir to be a string or a list of strings, got {type(self.data_dir)}"
            )
        if self.test_data_dir is not None:
            self.bid_test = self.load_data(self.test_data_dir, transforms=self.test_transforms)

    def train_dataloader(self):
        return DataLoader(
            self.bid_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.train_collate_fn,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.bid_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.eval_collate_fn,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.bid_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.eval_collate_fn,
            shuffle=False
        )