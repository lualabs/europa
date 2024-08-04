import os
from typing import Optional, List, Dict, Union, Any
from typing_extensions import Self
from pydantic import BaseModel, model_validator, field_validator
import yaml
from .data_config import DataConfig
from .training_config import TrainingConfig, ModelConfig

class EuropaConfig(BaseModel):
    train: TrainingConfig
    model: ModelConfig
    data: DataConfig

    @staticmethod
    def from_yaml(config_file):
        if not os.path.exists(config_file):
                raise FileNotFoundError(f"No path to config file found: {config_file}")
        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return EuropaConfig(**config)
    
    @model_validator(mode="after")
    def copy_processor_from_data(self) -> Self:
        self.model.processor = self.data.load_processor()
        return self
    
    @model_validator(mode="after")
    def copy_batch_size_from_data(self) -> Self:
        self.train.batch_size = self.data.batch_size
        return self
    
    @model_validator(mode="after")
    def share_hf_token_with_model(self) -> Self:
        self.model.hf_token = self.data.hf_token
        return self