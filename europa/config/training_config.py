import os
from typing import Optional, List, Any
from pydantic import BaseModel, ConfigDict
import yaml
from transformers import BitsAndBytesConfig, PaliGemmaForConditionalGeneration
from peft import get_peft_model, LoraConfig
from .utils import load_obj

class QuantizationConfig(BaseModel):
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "torch.bfloat16"

    def load_bnb_config(self):
        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_compute_type= load_obj(self.bnb_4bit_compute_dtype)
        )

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
    lr: float = 1e-4
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
    