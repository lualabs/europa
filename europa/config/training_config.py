import os
from typing import Optional, List, Any, Dict
from pydantic import BaseModel, ConfigDict, field_validator
import yaml
import transformers
from transformers import BitsAndBytesConfig, PaliGemmaForConditionalGeneration
from peft import get_peft_model, LoraConfig
from .utils import load_obj
from .config_map import SCHEDULER_MAP, OPTIMIZER_MAP

CONFIG_DEFAULTS = {
    "optimizer": "torch.optim.AdamW",
}

class QuantizationConfig(BaseModel):
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "torch.bfloat16"

    def load_bnb_config(self):
        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype= load_obj(self.bnb_4bit_compute_dtype)
        )
    
class OptimizerConfig(BaseModel):
    name: str = CONFIG_DEFAULTS["optimizer"]
    lr: float = 1e-4
    args: Optional[Dict[str, Any]] = None
    
    @field_validator("name", mode="before")
    @classmethod
    def maybe_transform_optimizer_string(cls, v):
        if v in OPTIMIZER_MAP.keys():
            return OPTIMIZER_MAP[v]
        return v

    def load(self, params):
        args = self.args if self.args is not None else {}
        return load_obj(self.name)(params=params, lr=self.lr, **args)

class SchedulerConfig(BaseModel):
    name: str
    args: Optional[Dict[str, Any]] = None

    @field_validator("name", mode="before")
    @classmethod
    def maybe_transform_scheduler_string(cls, v):
        if v in SCHEDULER_MAP.keys():
            return SCHEDULER_MAP[v]
        return v

    def load(self, optimizer):
        args = self.args if self.args is not None else {}
        # transformers uses a method to load the scheduler... cant use load_obj
        return getattr(transformers, self.name)(optimizer, **args)

class LoRAConfig(BaseModel):
    r: int = 8
    target_modules: List[str] = ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
    task_type: str = "CAUSAL_LM"

    def load_lora_config(self):
        return LoraConfig(
            r=self.r,
            target_modules=self.target_modules,
            task_type=self.task_type
        )

class ModelConfig(BaseModel):
    model_name: str
    repo_id: str
    quantization_config: QuantizationConfig = QuantizationConfig()
    lora_config: LoRAConfig = LoRAConfig()
    processor: Any = None
    hf_token: str = None
    model_config  = ConfigDict(protected_namespaces=())

    @staticmethod
    def from_yaml(config_file):
        if not os.path.exists(config_file):
                raise FileNotFoundError(f"No path to config file found: {config_file}")
        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            config = config["model"]
        return ModelConfig(**config)

    def load_model(self):
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.repo_id,
            quantization_config=self.quantization_config.load_bnb_config(),
            device_map={"":0},
            token=self.hf_token
        )
        model = get_peft_model(model, self.lora_config.load_lora_config())
        return model

class TrainingConfig(BaseModel):
    max_epochs: int
    check_val_every_n_epoch: int = 1
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 8
    scheduler: Optional[SchedulerConfig] = None
    optimizer: Optional[OptimizerConfig] = OptimizerConfig()
    batch_size: Optional[int] = None
    num_nodes: int = 1
    warmup_steps: int = 50
    result_path: str = "./result"
    verbose: bool = True
    max_length: int = 512
    

    @staticmethod
    def from_yaml(config_file):
        if not os.path.exists(config_file):
                raise FileNotFoundError(f"No path to config file found: {config_file}")
        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            config = config["train"]
        return TrainingConfig(**config)
    
    def load_scheduler(self, optimizer):
        return self.scheduler.load(optimizer)
    
    def load_optimizer(self, params):
        return self.optimizer.load(params)
    