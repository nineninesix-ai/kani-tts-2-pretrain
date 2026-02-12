"""
Training callbacks for attention metrics logging.
"""

import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from typing import Dict, Optional
import wandb

from .metrics import AttentionMetrics, MetricsConfig


class AttentionMetricsCallback(TrainerCallback):
    """
    Callback to compute and log attention analysis metrics during training.

    This callback:
    1. Registers hooks on model attention layers
    2. Computes metrics every N steps
    3. Logs to wandb with detailed interpretation
    4. Runs expensive metrics (token influence) less frequently
    """

    def __init__(
        self,
        metrics_config: MetricsConfig,
        log_steps: int = 100,
        influence_steps: int = 1000,
        print_interpretation: bool = True,
    ):
        """
        Args:
            metrics_config: Configuration for metrics computation
            log_steps: Compute and log metrics every N steps
            influence_steps: Compute token influence every N steps (expensive)
            print_interpretation: Whether to print human-readable interpretation
        """
        self.metrics_config = metrics_config
        self.log_steps = log_steps
        self.influence_steps = influence_steps
        self.print_interpretation = print_interpretation
        self.metrics = AttentionMetrics(metrics_config)

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model,
        **kwargs
    ):
        """Register hooks on model at training start."""
        print("\n" + "=" * 60)
        print("Registering attention metrics hooks...")
        print("=" * 60)
        self.metrics.output_variance.register_hooks(model)
        print(f"Registered {len(self.metrics.output_variance.hooks)} hooks")
        print(f"Will log metrics every {self.log_steps} steps")
        print(f"Will compute token influence every {self.influence_steps} steps")
        print("=" * 60 + "\n")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model,
        **kwargs
    ):
        """Compute metrics at regular intervals."""
        if state.global_step % self.log_steps != 0:
            return

        # Get a batch from the data loader
        # Note: We need to grab the batch from the trainer's internal state
        # This is a bit hacky but works with HF Trainer
        train_dataloader = kwargs.get('train_dataloader')
        if train_dataloader is None:
            return

        # Get one batch
        try:
            batch = next(iter(train_dataloader))
            batch = {k: v.to(model.device) for k, v in batch.items()}
        except (StopIteration, AttributeError):
            return

        # Compute metrics
        compute_influence = (state.global_step % self.influence_steps == 0)

        print(f"\n[Step {state.global_step}] Computing attention metrics...")

        metrics = self.metrics.compute_all(
            model=model,
            input_ids=batch['input_ids'],
            labels=batch['labels'],
            attention_mask=batch.get('attention_mask'),
            compute_influence=compute_influence,
        )

        # Log to wandb
        wandb.log(metrics, step=state.global_step)

        # Print interpretation
        if self.print_interpretation:
            interpretation = self.metrics.interpret_all(metrics)
            print(interpretation)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Clean up hooks at training end."""
        print("\n" + "=" * 60)
        print("Removing attention metrics hooks...")
        print("=" * 60)
        self.metrics.output_variance.remove_hooks()
        print("Hooks removed successfully")
        print("=" * 60 + "\n")


class EvaluationMetricsCallback(TrainerCallback):
    """
    Callback to compute comprehensive metrics during evaluation.

    This runs more expensive metrics on the validation set.
    """

    def __init__(
        self,
        metrics_config: MetricsConfig,
        max_eval_batches: int = 10,
    ):
        """
        Args:
            metrics_config: Configuration for metrics computation
            max_eval_batches: Maximum number of batches to evaluate
        """
        self.metrics_config = metrics_config
        self.max_eval_batches = max_eval_batches
        self.metrics = AttentionMetrics(metrics_config)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model,
        eval_dataloader,
        **kwargs
    ):
        """Compute metrics on evaluation set."""
        if eval_dataloader is None:
            return

        print("\n" + "=" * 60)
        print("Computing evaluation metrics...")
        print("=" * 60)

        # Register hooks
        self.metrics.output_variance.register_hooks(model)

        all_metrics = []

        # Evaluate on multiple batches
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(eval_dataloader):
                if i >= self.max_eval_batches:
                    break

                batch = {k: v.to(model.device) for k, v in batch.items()}

                # Compute metrics (with influence for thorough evaluation)
                metrics = self.metrics.compute_all(
                    model=model,
                    input_ids=batch['input_ids'],
                    labels=batch['labels'],
                    attention_mask=batch.get('attention_mask'),
                    compute_influence=(i == 0),  # Only on first batch
                )

                all_metrics.append(metrics)

        model.train()

        # Average metrics across batches
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics if not torch.isnan(torch.tensor(m[key]))]
                if values:
                    avg_metrics[key] = sum(values) / len(values)

            # Log to wandb with eval prefix
            eval_metrics = {k.replace('metrics/', 'eval_metrics/'): v for k, v in avg_metrics.items()}
            wandb.log(eval_metrics, step=state.global_step)

            # Print interpretation
            interpretation = self.metrics.interpret_all(avg_metrics)
            print("\nEVALUATION RESULTS:")
            print(interpretation)

        # Clean up hooks
        self.metrics.output_variance.remove_hooks()

        print("=" * 60 + "\n")
