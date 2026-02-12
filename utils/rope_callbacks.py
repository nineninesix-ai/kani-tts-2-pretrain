"""
Callbacks for monitoring learnable RoPE alpha parameters.
"""

import torch
import wandb
from transformers import TrainerCallback
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig


class AlphaMonitoringCallback(TrainerCallback):
    """
    Callback to monitor and log learnable RoPE alpha parameters to WandB.

    Logs:
    - Alpha values (constrained) per layer
    - Alpha weights (unconstrained) per layer
    - Gradient norms for alpha parameters
    """

    def __init__(self, log_steps: int = 100):
        """
        Initialize the callback.

        Args:
            log_steps: Log alpha values every N steps
        """
        self.log_steps = log_steps

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """
        Log alpha values and gradients at the end of each step.

        Args:
            args: TrainingArguments
            state: TrainerState
            control: TrainerControl
            model: The training model
        """
        if state.global_step % self.log_steps != 0:
            return

        if model is None:
            return

        # Unwrap model if using FSDP or other wrappers
        # This ensures we access the actual model instance, not the wrapper
        unwrapped_model = model

        # Try to unwrap FSDP
        if hasattr(unwrapped_model, '_fsdp_wrapped_module'):
            unwrapped_model = unwrapped_model._fsdp_wrapped_module

        # Try standard unwrap
        while hasattr(unwrapped_model, 'module'):
            unwrapped_model = unwrapped_model.module

        # Check if model has learnable RoPE
        if not hasattr(unwrapped_model, 'model') or not hasattr(unwrapped_model.model, 'use_learnable_rope'):
            return

        if not unwrapped_model.model.use_learnable_rope:
            return

        # Collect alpha values and gradients
        metrics = {}

        # Use FSDP summon_full_params context to access full (unsharded) parameters
        # This is critical when using FSDP, otherwise we only see local shards!
        with FSDP.summon_full_params(model, writeback=False):
            for layer_idx, learnable_rope in enumerate(unwrapped_model.model.learnable_rope_layers):
                if learnable_rope is None:
                    continue

                # Get alpha value (constrained) - compute it fresh each time!
                # Now we have access to the full parameter, not just a shard
                alpha_value = learnable_rope.alpha.item()
                metrics[f"rope_alpha/layer_{layer_idx}/value"] = alpha_value

                # Get raw weight (unconstrained)
                alpha_weight = learnable_rope.alpha_weight.item()
                metrics[f"rope_alpha/layer_{layer_idx}/raw_weight"] = alpha_weight

                # Get gradient if available
                if learnable_rope.alpha_weight.grad is not None:
                    grad_norm = learnable_rope.alpha_weight.grad.norm().item()
                    metrics[f"rope_grad/layer_{layer_idx}/norm"] = grad_norm

        # Compute statistics across layers
        alpha_values = [m for k, m in metrics.items() if "/value" in k]
        if len(alpha_values) > 0:
            metrics["rope_alpha/mean"] = sum(alpha_values) / len(alpha_values)
            metrics["rope_alpha/min"] = min(alpha_values)
            metrics["rope_alpha/max"] = max(alpha_values)
            metrics["rope_alpha/std"] = torch.tensor(alpha_values).std().item()

        # Debug: print to console every 500 steps to verify values are changing
        if state.global_step % 100 == 0 and len(metrics) > 0:
            print(f"\n🔍 [DEBUG] Learnable RoPE values at step {state.global_step}:")
            # Use summon_full_params again for debug output
            with FSDP.summon_full_params(model, writeback=False):
                for layer_idx, learnable_rope in enumerate(unwrapped_model.model.learnable_rope_layers):
                    if learnable_rope is not None:
                        w = learnable_rope.alpha_weight.item()
                        a = learnable_rope.alpha.item()
                        print(f"   Layer {layer_idx}: raw_weight={w:.6f} → alpha={a:.6f}")

        # Log to WandB
        if len(metrics) > 0:
            wandb.log(metrics, step=state.global_step)

        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Print alpha values to console when logging.

        Args:
            args: TrainingArguments
            state: TrainerState
            control: TrainerControl
            logs: Current logs
        """
        if state.global_step % (self.log_steps * 10) != 0:
            return

        # Print alpha values summary
        if logs and any("rope_alpha" in k for k in logs.keys()):
            print(f"\n🎯 Learnable RoPE Alpha Values (Step {state.global_step}):")
            for key, value in sorted(logs.items()):
                if "rope_alpha" in key and "/value" in key:
                    layer_idx = key.split("_")[2].split("/")[0]
                    print(f"   Layer {layer_idx}: α = {value:.4f}")

            if "rope_alpha/mean" in logs:
                print(f"   Statistics: mean={logs['rope_alpha/mean']:.4f}, "
                      f"std={logs.get('rope_alpha/std', 0):.4f}, "
                      f"range=[{logs.get('rope_alpha/min', 0):.4f}, {logs.get('rope_alpha/max', 0):.4f}]")
            print()

        return control
