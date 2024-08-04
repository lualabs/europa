import os
from typing import Optional, List, Dict, Union, Any
from pydantic import BaseModel, ConfigDict, field_validator
import yaml
from transformers import AutoProcessor
from .utils import load_obj

class DataConfig(BaseModel):
    repo_id: str
    max_length: int = 512
    data_dir: Union[str, List[str]]
    test_data_dir: Optional[str] = None
    batch_size: int
    num_workers: int = 0
    val_split: float = 0.1
    seed: int = 42
    # TODO: parse tranforms
    transforms: Optional[Dict[str, List[str]]] = None
    val_transforms: Optional[Dict[str, List[str]]] = None
    test_transforms: Optional[Dict[str, List[str]]] = None
    train_collate_fn: Optional[str] = None
    eval_collate_fn: Optional[str] = None
    processor: str = "auto"
    hf_token: str = None
    wandb_project: Optional[str] = None

    @staticmethod
    def from_yaml(config_file):
        if not os.path.exists(config_file):
                raise FileNotFoundError(f"No path to config file found: {config_file}")
        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            config = config["data"]
        return DataConfig(**config)

    @field_validator("data_dir")
    @classmethod
    def check_data_dir(cls, v):
        if not os.path.exists(v):
            raise FileNotFoundError(f"Data directory {v} not found")
        return v

    @field_validator("test_data_dir")
    @classmethod
    def check_test_data_dir(cls, v):
        if v is not None and not os.path.exists(v):
            raise FileNotFoundError(f"Test data directory {v} not found")
        return v

    @field_validator("hf_token", mode="before")
    @classmethod
    def parse_hf_token(cls, v):
        if v is not None:
            return v
        token = os.getenv("HF_TOKEN")
        if token is None:
            raise ValueError("Variable hf_token not passed and HF_TOKEN not found in environment variables.")

    def load_processor(self, args: Optional[Dict[str, Any]] = None):
        if self.processor == "auto":
            return AutoProcessor.from_pretrained(self.repo_id, token=self.hf_token)
        return load_obj(self.processor, **args)


# class DocClassifierConfig(BaseModel):
#     model_config  = ConfigDict(protected_namespaces=())
#     num_classes: int
#     from_pretrained: bool
#     model_name: str = "mobilenetv3_small_100"

#     @staticmethod
#     def from_yaml(config_file):
#         if not os.path.exists(config_file):
#                 raise FileNotFoundError(f"No path to config file found: {config_file}")
#         with open(config_file) as f:
#             config = yaml.load(f, Loader=yaml.FullLoader)
#         return DocClassifierConfig(**config)

#     def load_optimizer(self, params):
#         """
#         Loads the optimizer from the configuration.

#         Args:
#             params: The model parameters. This is equivelent to self.parameters().
#         """
#         return self.optimizer.load(params)

#     def load_scheduler(self, optimizer):
#         """
#         Loads the scheduler from the configuration.

#         Args:
#             optimizer: The optimizer to be used with the scheduler.
#         """
#         return self.scheduler.load(optimizer)
    
#     def load_metric(self, num_classes):
#         """
#         Loads the metric from the configuration

#         Args:
#             num_classes: The number of classes in the dataset
#         """
#         return self.metric.load(num_classes,)
    
#     def load_loss_fn(self):
#         """
#         Loads the loss function from the configuration
#         """
#         return self.loss_fn.load()