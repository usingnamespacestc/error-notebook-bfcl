from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class ModelConfig:
    model_name: str
    display_name: str
    underscore_to_dot: bool = False

class ModelConfigMapping(dict):
    def __getitem__(self, key):
        if key not in self:
            return ModelConfig(model_name=key, display_name=key, underscore_to_dot=False) # Default to false - most models support dots in function names
        return super().__getitem__(key)

MODEL_CONFIG_MAPPING = ModelConfigMapping()
