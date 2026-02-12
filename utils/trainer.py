"""
Trainer class for KaniTTS2-Pretrain.
"""

import torch
import wandb
from transformers import AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
from transformers.trainer_pt_utils import is_sagemaker_mp_enabled
from typing import Optional, List
from pathlib import Path

# For compatibility with SageMaker MP (not used but needed for create_optimizer)
try:
    import smdistributed.modelparallel.torch as smp
except ImportError:
    smp = None

from .config import ModelConfig, TrainingConfig
from .model import FlashCompatibleLfm2ForCausalLM
from .data import DataCollator, prepare_dataset
from .metrics import MetricsConfig
from .callbacks import AttentionMetricsCallback, EvaluationMetricsCallback
from .rope_callbacks import AlphaMonitoringCallback


class LearnableRoPETrainer(Trainer):
    """
    Custom Trainer that creates separate optimizer parameter groups for learnable RoPE.

    Creates two parameter groups:
    1. Model weights: normal learning rate
    2. Alpha parameters: reduced learning rate (gamma * base_lr)
    """

    def __init__(self, alpha_lr_ratio: float = 0.01, *args, **kwargs):
        """
        Initialize trainer with learnable RoPE support.

        Args:
            alpha_lr_ratio: Learning rate ratio for alpha parameters (default: 0.01)
        """
        self.alpha_lr_ratio = alpha_lr_ratio
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        """
        Create optimizer with separate parameter groups for alpha parameters.
        """
        if self.optimizer is None:
            # Separate parameters into alpha and non-alpha groups
            alpha_params = []
            non_alpha_params = []

            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue

                # Check if this is an alpha parameter
                if "alpha_weight" in name:
                    alpha_params.append(param)
                    print(f"   📊 Alpha parameter: {name}")
                else:
                    non_alpha_params.append(param)

            # Create parameter groups
            if len(alpha_params) > 0:
                # Learnable RoPE is enabled
                base_lr = self.args.learning_rate
                alpha_lr = base_lr * self.alpha_lr_ratio

                optimizer_grouped_parameters = [
                    {
                        "params": non_alpha_params,
                        "lr": base_lr,
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": alpha_params,
                        "lr": alpha_lr,
                        "weight_decay": 0.0,  # No weight decay for alpha
                    },
                ]

                print(f"\n🎯 Optimizer Configuration:")
                print(f"   - Model parameters: {len(non_alpha_params)} params, LR={base_lr:.2e}")
                print(f"   - Alpha parameters: {len(alpha_params)} params, LR={alpha_lr:.2e} (γ={self.alpha_lr_ratio})")
                print(f"   - Alpha LR ratio: {self.alpha_lr_ratio} (alpha_lr = {self.alpha_lr_ratio} × base_lr)")
            else:
                # No learnable RoPE, use standard optimizer
                optimizer_grouped_parameters = [
                    {
                        "params": non_alpha_params,
                        "lr": self.args.learning_rate,
                        "weight_decay": self.args.weight_decay,
                    }
                ]
                print(f"\n🎯 Optimizer Configuration (Standard):")
                print(f"   - Model parameters: {len(non_alpha_params)} params, LR={self.args.learning_rate:.2e}")

            # Create optimizer
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if self.args.deepspeed:
                self.optimizer = self.optimizer
            if is_sagemaker_mp_enabled():
                self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer


class TokenizerSaveCallback(TrainerCallback):
    """
    Callback to save tokenizer alongside model checkpoints.
    """

    def __init__(self, tokenizer):
        """
        Initialize the callback.

        Args:
            tokenizer: The tokenizer to save with checkpoints
        """
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        """
        Save tokenizer when a checkpoint is saved.

        Args:
            args: TrainingArguments
            state: TrainerState
            control: TrainerControl
        """
        # Get the checkpoint directory from the output_dir
        checkpoint_folder = f"checkpoint-{state.global_step}"
        output_dir = Path(args.output_dir) / checkpoint_folder

        if output_dir.exists():
            print(f"💾 Saving tokenizer to {output_dir}")
            self.tokenizer.save_pretrained(output_dir)
            print("✅ Tokenizer saved alongside checkpoint")

        return control


class KaniTTS2Trainer:
    """
    High-level trainer class for KaniTTS2 pretraining.

    Handles:
    - Model initialization
    - Tokenizer setup
    - Dataset loading
    - Training configuration
    - Wandb integration
    """

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig
    ):
        """
        Initialize the trainer.

        Args:
            model_config: Model configuration
            training_config: Training configuration
        """
        self.model_config = model_config
        self.training_config = training_config

        self.model: Optional[FlashCompatibleLfm2ForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.trainer: Optional[Trainer] = None
        self.dataset = None

    def setup(self):
        """Setup all components for training."""
        self._init_wandb()
        self._load_tokenizer()
        self._load_model()
        self._load_dataset()
        self._create_trainer()

    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        if "wandb" in self.training_config.report_to:
            wandb_kwargs = {
                "project": self.training_config.wandb_project,
                "name": self.training_config.wandb_name,
            }
            if self.training_config.wandb_entity:
                wandb_kwargs["entity"] = self.training_config.wandb_entity

            wandb.init(**wandb_kwargs)
            print(f"✅ Wandb initialized: {self.training_config.wandb_project}/{self.training_config.wandb_name}")

    def _load_tokenizer(self):
        """Load the tokenizer."""
        print(f"📝 Loading tokenizer from {self.model_config.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_id)

        # Resize tokenizer if first train
        if self.model_config.first_train:
            num_new_tokens = self.model_config.num_audio_tokens + 10  # audio tokens + special tokens
            new_tokens = [f"<custom_token_{i}>" for i in range(num_new_tokens)]
            self.tokenizer.add_tokens(new_tokens)
            print(f"✅ Added {num_new_tokens} new tokens to tokenizer")
            print(f"   Total vocabulary size: {len(self.tokenizer)}")
        else:
            print(f"✅ Using existing tokenizer (vocab_size={len(self.tokenizer)})")

    def _load_model(self):
        """Load and configure the model."""
        print(f"🚀 Loading model from {self.model_config.model_id}")

        self.model = FlashCompatibleLfm2ForCausalLM.from_pretrained(
            self.model_config.model_id,
            audio_tokens_start=self.model_config.audio_tokens_start,
            tokens_per_frame=self.model_config.tokens_per_frame,
            audio_step=self.model_config.audio_step,
            use_learnable_rope=self.model_config.use_learnable_rope,
            alpha_min=self.model_config.alpha_min,
            alpha_max=self.model_config.alpha_max,
            speaker_emb_dim=self.model_config.speaker_emb_dim,
            attn_implementation=self.model_config.attn_implementation,
            dtype=self.model_config.torch_dtype,
        )

        # Resize embeddings if first train
        if self.model_config.first_train:
            print("🔧 Resizing token embeddings...")
            self.model.resize_token_embeddings(len(self.tokenizer))
            # Re-cast to ensure all parameters are in correct dtype (Flash Attention requirement)
            self.model = self.model.to(self.model_config.torch_dtype)
            print(f"✅ Resized embeddings to {len(self.tokenizer)} tokens")
        else:
            print("✅ Using existing model embeddings")

        print(f"\n📊 Model Configuration:")
        print(f"   - Model ID: {self.model_config.model_id}")
        print(f"   - Vocabulary size: {self.model.config.vocab_size}")
        print(f"   - Text vocab size: {self.model_config.text_vocab_size}")
        print(f"   - Audio tokens start: {self.model_config.audio_tokens_start}")
        print(f"   - Audio tokens count: {self.model_config.num_audio_tokens}")
        print(f"   - Tokens per frame: {self.model_config.tokens_per_frame}")
        print(f"   - Audio step: {self.model_config.audio_step}")
        print(f"   - Speaker embedding: {self.model_config.speaker_emb_dim} -> {self.model.config.hidden_size}")
        print(f"   - Attention: {self.model_config.attn_implementation}")
        print(f"   - Dtype: {self.model_config.dtype}")

    def _load_dataset(self):
        """Load the training dataset."""
        self.dataset = prepare_dataset(self.training_config.dataset_path)

    def _create_trainer(self):
        """Create the HuggingFace Trainer."""
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            overwrite_output_dir=self.training_config.overwrite_output_dir,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            warmup_steps=self.training_config.warmup_steps,
            max_grad_norm=self.training_config.max_grad_norm,
            optim=self.training_config.optim,
            bf16=self.training_config.bf16,
            fp16=self.training_config.fp16,
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps,
            save_total_limit=self.training_config.save_total_limit,
            report_to=self.training_config.report_to,
            dataloader_num_workers=self.training_config.dataloader_num_workers,
            remove_unused_columns=self.training_config.remove_unused_columns,
            average_tokens_across_devices=self.training_config.average_tokens_across_devices,
        )

        # Create data collator
        data_collator = DataCollator(pad_token_id=self.model_config.pad_token_id)

        # Create callbacks
        callbacks = []

        # Add tokenizer save callback (always enabled)
        callbacks.append(TokenizerSaveCallback(tokenizer=self.tokenizer))

        # Add alpha monitoring callback if learnable RoPE is enabled
        if self.model_config.use_learnable_rope:
            alpha_log_steps = getattr(self.training_config, 'metrics_log_steps', 100)
            callbacks.append(AlphaMonitoringCallback(log_steps=alpha_log_steps))
            print(f"   - Alpha monitoring: ENABLED (log every {alpha_log_steps} steps)")

        # Add attention metrics callback if enabled
        if self.training_config.enable_metrics:
            metrics_config = MetricsConfig(
                text_vocab_size=self.model_config.text_vocab_size,
                audio_codebook_size=self.model_config.audio_codebook_size,
                tokens_per_frame=self.model_config.tokens_per_frame,
                num_audio_layers=self.model_config.num_audio_layers,
            )

            callbacks.append(
                AttentionMetricsCallback(
                    metrics_config=metrics_config,
                    log_steps=self.training_config.metrics_log_steps,
                    influence_steps=self.training_config.metrics_influence_steps,
                    print_interpretation=self.training_config.metrics_print_interpretation,
                )
            )

            print(f"\n📊 Metrics Configuration:")
            print(f"   - Attention metrics: ENABLED")
            print(f"   - Log every: {self.training_config.metrics_log_steps} steps")
            print(f"   - Token influence every: {self.training_config.metrics_influence_steps} steps")

        # Create trainer (use custom trainer if learnable RoPE is enabled)
        if self.model_config.use_learnable_rope:
            alpha_lr_ratio = getattr(self.training_config, 'alpha_lr_ratio', 0.01)
            self.trainer = LearnableRoPETrainer(
                alpha_lr_ratio=alpha_lr_ratio,
                model=self.model,
                args=training_args,
                train_dataset=self.dataset,
                data_collator=data_collator,
                callbacks=callbacks,
            )
        else:
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset,
                data_collator=data_collator,
                callbacks=callbacks,
            )

        print(f"\n🎯 Training Configuration:")
        print(f"   - Epochs: {self.training_config.num_train_epochs}")
        print(f"   - Batch size (per device): {self.training_config.per_device_train_batch_size}")
        print(f"   - Gradient accumulation: {self.training_config.gradient_accumulation_steps}")
        print(f"   - Learning rate: {self.training_config.learning_rate}")
        print(f"   - LR scheduler: {self.training_config.lr_scheduler_type}")
        print(f"   - Warmup steps: {self.training_config.warmup_steps}")
        print(f"   - Save every: {self.training_config.save_steps} steps")
        print(f"   - Output dir: {self.training_config.output_dir}")

    def train(self, resume_from_checkpoint: Optional[str] = None):
        """
        Start training.

        Args:
            resume_from_checkpoint: Optional path to checkpoint to resume from
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call setup() first.")

        print("\n" + "=" * 60)
        print("🚀 Starting Training")
        print("=" * 60)
        print(f"   - Frame-level position encoding: ENABLED")
        print(f"   - Flash Attention 2: {self.model_config.attn_implementation}")
        print(f"   - Dataset samples: {len(self.dataset)}")
        print("=" * 60 + "\n")

        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        print("\n" + "=" * 60)
        print("✅ Training Completed!")
        print("=" * 60)

    def save_model(self, output_dir: str):
        """
        Save the trained model and tokenizer.

        Args:
            output_dir: Directory to save model and tokenizer
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not initialized.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n💾 Saving model to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print("✅ Model and tokenizer saved")

    def push_to_hub(self, repo_id: str, private: bool = True):
        """
        Push model and tokenizer to Hugging Face Hub.

        Args:
            repo_id: Repository ID (e.g., "username/model-name")
            private: Whether to make repository private
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not initialized.")

        print(f"\n📤 Pushing to Hugging Face Hub: {repo_id}")
        self.model.push_to_hub(repo_id, private=private)
        self.tokenizer.push_to_hub(repo_id, private=private)
        print(f"✅ Model pushed to https://huggingface.co/{repo_id}")
