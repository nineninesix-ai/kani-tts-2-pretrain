"""
Configuration classes and loaders for KaniTTS2-Pretrain.
"""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
import torch


@dataclass
class ModelConfig:
    """Model configuration."""

    # Model identification
    model_id: str = "LiquidAI/LFM2-350M"

    # Vocabulary
    text_vocab_size: int = 64400
    total_vocab_size: int = 80538

    # Audio tokens
    audio_codebook_size: int = 4032
    tokens_per_frame: int = 4
    num_audio_layers: int = 4
    audio_tokens_start: int = 64410  # text_vocab_size + 10 special tokens
    audio_step: float = 1.0  # Position step per audio frame (fractional encoding)

    # Special tokens
    pad_token_id: int = 64407

    # Model settings
    attn_implementation: str = "flash_attention_2"
    dtype: str = "bfloat16"

    # Training mode
    first_train: bool = True
    init_from_scratch: bool = False  # If True, initialize weights randomly (no pretrained weights loaded)

    # Learnable RoPE
    use_learnable_rope: bool = False
    alpha_min: float = 0.1
    alpha_max: float = 2.0

    # Speaker embeddings
    speaker_emb_dim: int = 128

    @property
    def torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.dtype, torch.bfloat16)

    @property
    def num_audio_tokens(self) -> int:
        """Calculate total number of audio tokens."""
        return self.audio_codebook_size * self.num_audio_layers

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config.get("model", {}))


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Output
    output_dir: str = "./checkpoints"
    overwrite_output_dir: bool = True
    save_steps: int = 5000
    save_total_limit: int = 10

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4

    # Optimizer
    learning_rate: float = 9e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 2000
    max_grad_norm: float = 3.0
    optim: str = "adamw_torch_fused"
    alpha_lr_ratio: float = 0.01  # Learning rate ratio for learnable RoPE alpha parameters

    # Precision
    bf16: bool = True
    fp16: bool = False

    # Logging
    logging_steps: int = 1
    report_to: List[str] = field(default_factory=lambda: ["wandb"])

    # Data
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = False
    dataset_path: str = "./train_dataset"

    # Performance
    average_tokens_across_devices: bool = False

    # Wandb
    wandb_project: str = "LFM2-nano-codec-custom-attention"
    wandb_name: str = "train_flash_attention_v1"
    wandb_entity: Optional[str] = None

    # Metrics
    enable_metrics: bool = True
    metrics_log_steps: int = 100
    metrics_influence_steps: int = 1000
    metrics_print_interpretation: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        training = config.get("training", {})
        wandb = config.get("wandb", {})
        metrics = config.get("metrics", {})

        # Merge wandb config into training
        if wandb:
            training["wandb_project"] = wandb.get("project", training.get("wandb_project"))
            training["wandb_name"] = wandb.get("name", training.get("wandb_name"))
            training["wandb_entity"] = wandb.get("entity", training.get("wandb_entity"))

        # Merge metrics config into training
        if metrics:
            training["enable_metrics"] = metrics.get("enable_metrics", training.get("enable_metrics", True))
            training["metrics_log_steps"] = metrics.get("log_steps", training.get("metrics_log_steps", 100))
            training["metrics_influence_steps"] = metrics.get("influence_steps", training.get("metrics_influence_steps", 1000))
            training["metrics_print_interpretation"] = metrics.get("print_interpretation", training.get("metrics_print_interpretation", True))

        return cls(**training)


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    max_duration_sec: Optional[float] = None
    lenght_statistics: bool = False
    hf_datasets: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> "DatasetConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)


def load_configs(
    model_config_path: str = "configs/model_config.yaml",
    training_config_path: str = "configs/training_config.yaml",
    dataset_config_path: str = "configs/dataset_config.yaml",
) -> tuple[ModelConfig, TrainingConfig, DatasetConfig]:
    """
    Load all configuration files.

    Args:
        model_config_path: Path to model configuration YAML
        training_config_path: Path to training configuration YAML
        dataset_config_path: Path to dataset configuration YAML

    Returns:
        Tuple of (ModelConfig, TrainingConfig, DatasetConfig)
    """
    model_config = ModelConfig.from_yaml(model_config_path)
    training_config = TrainingConfig.from_yaml(training_config_path)
    dataset_config = DatasetConfig.from_yaml(dataset_config_path)

    return model_config, training_config, dataset_config
