"""
Attention Analysis Metrics for Frame-Level Position Encoding

This module implements metrics to verify that frame-level position encoding
achieves layer isolation without explicit attention masking.

Key Insights:
- Frame-level PE assigns same position to all 4 tokens in a frame
- This should implicitly encourage layer-wise attention patterns
- These metrics validate if the model learns proper layer isolation

Metrics:
1. Layer-Specific Perplexity: Measures prediction quality per audio layer
2. Output Variance: Measures attention focus via GQA output statistics
3. Token Influence: Gradient-based analysis of which tokens influence predictions
4. Cross-Layer Confusion: Matrix showing if model confuses different layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class MetricsConfig:
    """Configuration for attention analysis metrics."""
    text_vocab_size: int = 64400
    audio_codebook_size: int = 4032
    num_audio_layers: int = 4
    tokens_per_frame: int = 4

    # Thresholds for good layer isolation
    perplexity_variance_threshold: float = 0.1  # Max 10% variance across layers
    output_variance_threshold: float = 1.5  # Min variance for focused attention
    same_layer_influence_threshold: float = 0.6  # Min influence from same layer
    cross_layer_influence_threshold: float = 0.2  # Max influence from other layers
    confusion_diagonal_threshold: float = 0.8  # Min diagonal values in confusion matrix

    @property
    def audio_tokens_start(self) -> int:
        """Start index for audio tokens (text_vocab + 10 special tokens)."""
        return self.text_vocab_size + 10

    @property
    def total_vocab_size(self) -> int:
        """Total vocabulary size."""
        return self.text_vocab_size + 10 + (self.num_audio_layers * self.audio_codebook_size)

    def get_layer_range(self, layer_idx: int) -> Tuple[int, int]:
        """Get token ID range for a specific audio layer (0-3)."""
        start = self.audio_tokens_start + layer_idx * self.audio_codebook_size
        end = start + self.audio_codebook_size
        return start, end

    def token_id_to_layer(self, token_id: int) -> Optional[int]:
        """Convert token ID to audio layer index (0-3), or None if not audio token."""
        if token_id < self.audio_tokens_start:
            return None
        offset = token_id - self.audio_tokens_start
        layer = offset // self.audio_codebook_size
        if layer >= self.num_audio_layers:
            return None
        return layer


class LayerSpecificPerplexity:
    """
    Compute perplexity separately for each audio layer.

    Good layer isolation means:
    - Perplexity should be similar across all layers (±10%)
    - If one layer has much higher perplexity, the model is struggling with that layer

    Interpretation:
    - PPL < 50: Excellent
    - PPL 50-100: Good
    - PPL 100-200: Acceptable
    - PPL > 200: Poor (model not learning this layer well)
    """

    def __init__(self, config: MetricsConfig):
        self.config = config

    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute perplexity for each audio layer.

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            labels: Ground truth labels [batch_size, seq_len]
            attention_mask: Optional mask [batch_size, seq_len]

        Returns:
            Dictionary with keys:
                - layer_0_ppl, layer_1_ppl, layer_2_ppl, layer_3_ppl: Per-layer perplexity
                - avg_audio_ppl: Average across all audio layers
                - ppl_variance: Variance across layers (lower is better)
                - text_ppl: Perplexity for text tokens
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Shift for causal LM: predict next token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        if attention_mask is not None:
            shift_mask = attention_mask[..., 1:].contiguous().bool()
        else:
            shift_mask = torch.ones_like(shift_labels, dtype=torch.bool)

        # Flatten batch and sequence dimensions for easier indexing
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))  # [batch*seq, vocab]
        flat_labels = shift_labels.view(-1)  # [batch*seq]
        flat_mask = shift_mask.view(-1)  # [batch*seq]

        # Ignore padding (-100) in labels
        valid_mask = (flat_labels != -100) & flat_mask

        results = {}
        layer_ppls = []

        # Compute perplexity for each audio layer
        for layer_idx in range(self.config.num_audio_layers):
            layer_start, layer_end = self.config.get_layer_range(layer_idx)

            # Mask for tokens in this layer
            layer_mask = (flat_labels >= layer_start) & (flat_labels < layer_end) & valid_mask

            if layer_mask.sum() > 0:
                # Compute cross-entropy loss for this layer
                layer_logits = flat_logits[layer_mask]
                layer_labels = flat_labels[layer_mask]

                loss = F.cross_entropy(layer_logits, layer_labels, reduction='mean')
                ppl = torch.exp(loss).item()
                layer_ppls.append(ppl)
            else:
                ppl = float('nan')

            results[f'layer_{layer_idx}_ppl'] = ppl

        # Compute text perplexity
        text_mask = (flat_labels < self.config.text_vocab_size) & valid_mask
        if text_mask.sum() > 0:
            text_logits = flat_logits[text_mask]
            text_labels = flat_labels[text_mask]
            loss = F.cross_entropy(text_logits, text_labels, reduction='mean')
            results['text_ppl'] = torch.exp(loss).item()
        else:
            results['text_ppl'] = float('nan')

        # Compute statistics across audio layers
        valid_ppls = [p for p in layer_ppls if not np.isnan(p)]
        if valid_ppls:
            results['avg_audio_ppl'] = np.mean(valid_ppls)
            results['ppl_variance'] = np.var(valid_ppls) / (np.mean(valid_ppls) ** 2)  # Coefficient of variation
            results['ppl_std'] = np.std(valid_ppls)
            results['ppl_max_diff'] = max(valid_ppls) - min(valid_ppls)
        else:
            results['avg_audio_ppl'] = float('nan')
            results['ppl_variance'] = float('nan')
            results['ppl_std'] = float('nan')
            results['ppl_max_diff'] = float('nan')

        return results

    def interpret(self, metrics: Dict[str, float]) -> str:
        """Interpret perplexity metrics."""
        lines = ["=== Layer-Specific Perplexity Analysis ==="]

        # Check each layer
        for i in range(self.config.num_audio_layers):
            ppl = metrics.get(f'layer_{i}_ppl', float('nan'))
            if np.isnan(ppl):
                lines.append(f"Layer {i}: No data")
                continue

            if ppl < 50:
                status = "✓ Excellent"
            elif ppl < 100:
                status = "✓ Good"
            elif ppl < 200:
                status = "⚠ Acceptable"
            else:
                status = "✗ Poor"

            lines.append(f"Layer {i}: {ppl:.2f} {status}")

        # Check variance across layers
        variance = metrics.get('ppl_variance', float('nan'))
        if not np.isnan(variance):
            if variance < self.config.perplexity_variance_threshold:
                status = "✓ Good layer isolation"
            else:
                status = "✗ High variance - layers learning differently"
            lines.append(f"\nVariance: {variance:.4f} {status}")

        return "\n".join(lines)


class OutputVarianceTracker:
    """
    Measure attention output variance to assess attention focus.

    Higher variance = more focused attention (good)
    Lower variance = diffuse attention (potentially mixing layers)

    We hook into GQA layers and measure output.var(dim=1).mean()

    Interpretation:
    - Variance > 2.0: Excellent focus
    - Variance 1.5-2.0: Good focus
    - Variance 1.0-1.5: Acceptable
    - Variance < 1.0: Poor (too diffuse, likely mixing information)
    """

    def __init__(self, config: MetricsConfig):
        self.config = config
        self.hooks = []
        self.variance_values = []

    def register_hooks(self, model: nn.Module):
        """Register forward hooks on attention layers."""
        self.remove_hooks()

        def hook_fn(module, input, output):
            # output is typically (batch_size, seq_len, hidden_dim)
            if isinstance(output, tuple):
                output = output[0]

            # Compute variance across hidden dimension
            variance = output.var(dim=-1).mean().item()
            self.variance_values.append(variance)

        # Find all attention layers (Lfm2Attention or similar)
        for name, module in model.named_modules():
            if 'attention' in name.lower() and hasattr(module, 'forward'):
                handle = module.register_forward_hook(hook_fn)
                self.hooks.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def compute(self) -> Dict[str, float]:
        """
        Compute statistics from collected variance values.

        Returns:
            Dictionary with:
                - avg_output_variance: Average variance across all layers
                - min_output_variance: Minimum variance
                - max_output_variance: Maximum variance
        """
        if not self.variance_values:
            return {
                'avg_output_variance': float('nan'),
                'min_output_variance': float('nan'),
                'max_output_variance': float('nan'),
            }

        return {
            'avg_output_variance': np.mean(self.variance_values),
            'min_output_variance': np.min(self.variance_values),
            'max_output_variance': np.max(self.variance_values),
        }

    def reset(self):
        """Reset collected values."""
        self.variance_values = []

    def interpret(self, metrics: Dict[str, float]) -> str:
        """Interpret output variance metrics."""
        lines = ["=== Output Variance Analysis ==="]

        avg_var = metrics.get('avg_output_variance', float('nan'))
        if np.isnan(avg_var):
            return "\n".join(lines + ["No data collected"])

        if avg_var > 2.0:
            status = "✓ Excellent focus"
        elif avg_var > self.config.output_variance_threshold:
            status = "✓ Good focus"
        elif avg_var > 1.0:
            status = "⚠ Acceptable focus"
        else:
            status = "✗ Poor focus (too diffuse)"

        lines.append(f"Average variance: {avg_var:.3f} {status}")
        lines.append(f"Range: [{metrics['min_output_variance']:.3f}, {metrics['max_output_variance']:.3f}]")

        return "\n".join(lines)


class TokenInfluenceAnalyzer:
    """
    Gradient-based analysis of which tokens influence predictions.

    For each predicted audio token, we measure:
    - Influence from text tokens
    - Influence from same-layer audio tokens
    - Influence from other-layer audio tokens

    Good layer isolation means:
    - High influence from text tokens (needed for semantic content)
    - High influence from same-layer tokens (layer continuity)
    - Low influence from other-layer tokens (layer isolation)

    Interpretation:
    - Same-layer influence > 0.6: Good
    - Cross-layer influence < 0.2: Good isolation
    """

    def __init__(self, config: MetricsConfig):
        self.config = config

    def compute(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_samples: int = 32  # Limit to avoid OOM
    ) -> Dict[str, float]:
        """
        Compute token influence metrics using gradients.

        Args:
            model: The model to analyze
            input_ids: Input token IDs [batch_size, seq_len]
            labels: Labels [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            max_samples: Maximum number of tokens to analyze

        Returns:
            Dictionary with influence metrics per layer
        """
        model.eval()

        batch_size, seq_len = input_ids.shape

        # Get embeddings and enable gradients
        embeddings = model.get_input_embeddings()(input_ids)
        embeddings.requires_grad_(True)
        embeddings.retain_grad()  # Required for non-leaf tensors

        # Forward pass
        outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
        logits = outputs.logits

        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_input_ids = input_ids[..., :-1].contiguous()

        if attention_mask is not None:
            shift_mask = attention_mask[..., 1:].contiguous()
        else:
            shift_mask = torch.ones_like(shift_labels, dtype=torch.bool)

        valid_mask = (shift_labels != -100) & shift_mask

        # Collect influence statistics per layer
        results = {}

        for layer_idx in range(self.config.num_audio_layers):
            layer_start, layer_end = self.config.get_layer_range(layer_idx)

            # Find positions predicting this layer
            layer_pred_mask = (shift_labels >= layer_start) & (shift_labels < layer_end) & valid_mask

            if layer_pred_mask.sum() == 0:
                continue

            # Sample up to max_samples positions
            positions = torch.nonzero(layer_pred_mask, as_tuple=False)
            if len(positions) > max_samples:
                indices = torch.randperm(len(positions))[:max_samples]
                positions = positions[indices]

            influences = {
                'text': [],
                'same_layer': [],
                'other_layers': []
            }

            for batch_idx, pos in positions:
                batch_idx, pos = batch_idx.item(), pos.item()

                # Get prediction and label
                pred_logits = shift_logits[batch_idx, pos]
                true_label = shift_labels[batch_idx, pos]

                # Compute loss for this token
                loss = F.cross_entropy(pred_logits.unsqueeze(0), true_label.unsqueeze(0))

                # Compute gradients w.r.t. embeddings
                if embeddings.grad is not None:
                    embeddings.grad.zero_()

                loss.backward(retain_graph=True)

                # Measure gradient magnitude for each input token
                grad_norms = embeddings.grad[batch_idx, :pos+1].norm(dim=-1)  # Only past tokens

                # Categorize influence by token type
                input_tokens = shift_input_ids[batch_idx, :pos+1]

                for i, token_id in enumerate(input_tokens):
                    token_id = token_id.item()
                    influence = grad_norms[i].item()

                    token_layer = self.config.token_id_to_layer(token_id)

                    if token_layer is None:
                        # Text token
                        influences['text'].append(influence)
                    elif token_layer == layer_idx:
                        # Same layer
                        influences['same_layer'].append(influence)
                    else:
                        # Other layer
                        influences['other_layers'].append(influence)

            # Compute normalized influence scores
            total_influence = (
                sum(influences['text']) +
                sum(influences['same_layer']) +
                sum(influences['other_layers'])
            )

            if total_influence > 0:
                results[f'layer_{layer_idx}_text_influence'] = sum(influences['text']) / total_influence
                results[f'layer_{layer_idx}_same_layer_influence'] = sum(influences['same_layer']) / total_influence
                results[f'layer_{layer_idx}_cross_layer_influence'] = sum(influences['other_layers']) / total_influence

        model.train()
        return results

    def interpret(self, metrics: Dict[str, float]) -> str:
        """Interpret token influence metrics."""
        lines = ["=== Token Influence Analysis ==="]

        for layer_idx in range(self.config.num_audio_layers):
            same_layer = metrics.get(f'layer_{layer_idx}_same_layer_influence', float('nan'))
            cross_layer = metrics.get(f'layer_{layer_idx}_cross_layer_influence', float('nan'))

            if np.isnan(same_layer):
                continue

            lines.append(f"\nLayer {layer_idx}:")

            # Check same-layer influence
            if same_layer > self.config.same_layer_influence_threshold:
                status = "✓ Good"
            else:
                status = "✗ Low"
            lines.append(f"  Same-layer: {same_layer:.3f} {status}")

            # Check cross-layer influence
            if cross_layer < self.config.cross_layer_influence_threshold:
                status = "✓ Good isolation"
            else:
                status = "✗ High cross-layer leak"
            lines.append(f"  Cross-layer: {cross_layer:.3f} {status}")

        return "\n".join(lines)


class CrossLayerConfusionMatrix:
    """
    Compute confusion matrix showing if model confuses different layers.

    For each predicted audio token, we check:
    - Is the top prediction from the correct layer?
    - Which layers appear in top-K predictions?

    Good layer isolation means:
    - Diagonal values > 0.8 (predicting correct layer)
    - Off-diagonal values < 0.1 (not confusing layers)

    Interpretation:
    - Diagonal > 0.8: Excellent layer separation
    - Diagonal 0.6-0.8: Good layer separation
    - Diagonal < 0.6: Poor - model is confusing layers
    """

    def __init__(self, config: MetricsConfig):
        self.config = config

    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        top_k: int = 5
    ) -> Dict[str, float]:
        """
        Compute confusion matrix for audio layers.

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            labels: Ground truth labels [batch_size, seq_len]
            attention_mask: Optional mask [batch_size, seq_len]
            top_k: Number of top predictions to consider

        Returns:
            Dictionary with confusion matrix values and metrics
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        if attention_mask is not None:
            shift_mask = attention_mask[..., 1:].contiguous().bool()
        else:
            shift_mask = torch.ones_like(shift_labels, dtype=torch.bool)

        # Flatten batch and sequence dimensions
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))  # [batch*seq, vocab]
        flat_labels = shift_labels.view(-1)  # [batch*seq]
        flat_mask = shift_mask.view(-1)  # [batch*seq]

        valid_mask = (flat_labels != -100) & flat_mask

        # Initialize confusion matrix: rows=true layer, cols=predicted layer
        confusion = np.zeros((self.config.num_audio_layers, self.config.num_audio_layers))

        # Get top-k predictions
        topk_values, topk_indices = torch.topk(flat_logits, k=top_k, dim=-1)  # [batch*seq, top_k]

        for layer_idx in range(self.config.num_audio_layers):
            layer_start, layer_end = self.config.get_layer_range(layer_idx)

            # Mask for this layer
            layer_mask = (flat_labels >= layer_start) & (flat_labels < layer_end) & valid_mask

            if layer_mask.sum() == 0:
                continue

            # Get predictions for this layer
            layer_topk = topk_indices[layer_mask]  # [num_tokens, top_k]

            # Count predictions in each layer
            for pred_layer_idx in range(self.config.num_audio_layers):
                pred_start, pred_end = self.config.get_layer_range(pred_layer_idx)

                # Check if any top-k prediction is in this layer
                in_layer = (layer_topk >= pred_start) & (layer_topk < pred_end)
                count = in_layer.any(dim=-1).sum().item()

                confusion[layer_idx, pred_layer_idx] = count

        # Normalize rows
        row_sums = confusion.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        confusion_normalized = confusion / row_sums

        # Compute metrics
        results = {}

        # Store full confusion matrix
        for i in range(self.config.num_audio_layers):
            for j in range(self.config.num_audio_layers):
                results[f'confusion_{i}_{j}'] = confusion_normalized[i, j]

        # Diagonal values (correct predictions)
        diagonal = np.diag(confusion_normalized)
        results['avg_diagonal'] = np.mean(diagonal)
        results['min_diagonal'] = np.min(diagonal)

        # Off-diagonal values (confusion)
        mask = np.ones_like(confusion_normalized, dtype=bool)
        np.fill_diagonal(mask, False)
        off_diagonal = confusion_normalized[mask]
        results['avg_off_diagonal'] = np.mean(off_diagonal)
        results['max_off_diagonal'] = np.max(off_diagonal)

        return results

    def interpret(self, metrics: Dict[str, float]) -> str:
        """Interpret confusion matrix metrics."""
        lines = ["=== Cross-Layer Confusion Matrix ==="]

        # Print matrix
        lines.append("\nConfusion Matrix (rows=true, cols=pred):")
        lines.append("     " + "  ".join([f"L{i}" for i in range(self.config.num_audio_layers)]))

        for i in range(self.config.num_audio_layers):
            row = [f"{metrics.get(f'confusion_{i}_{j}', 0):.2f}" for j in range(self.config.num_audio_layers)]
            lines.append(f"L{i}   " + "  ".join(row))

        # Interpret diagonal
        avg_diag = metrics.get('avg_diagonal', float('nan'))
        min_diag = metrics.get('min_diagonal', float('nan'))

        if not np.isnan(avg_diag):
            lines.append(f"\nAverage diagonal: {avg_diag:.3f}")
            lines.append(f"Minimum diagonal: {min_diag:.3f}")

            if min_diag > self.config.confusion_diagonal_threshold:
                status = "✓ Excellent layer separation"
            elif min_diag > 0.6:
                status = "✓ Good layer separation"
            else:
                status = "✗ Poor - model confusing layers"

            lines.append(status)

        return "\n".join(lines)


class AttentionMetrics:
    """
    Main class coordinating all attention analysis metrics.
    """

    def __init__(self, config: MetricsConfig):
        self.config = config
        self.layer_perplexity = LayerSpecificPerplexity(config)
        self.output_variance = OutputVarianceTracker(config)
        self.token_influence = TokenInfluenceAnalyzer(config)
        self.confusion_matrix = CrossLayerConfusionMatrix(config)

    def compute_all(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        compute_influence: bool = False,  # Expensive, run less frequently
    ) -> Dict[str, float]:
        """
        Compute all metrics.

        Args:
            model: The model to analyze
            input_ids: Input token IDs
            labels: Labels
            attention_mask: Attention mask
            compute_influence: Whether to compute token influence (expensive)

        Returns:
            Dictionary with all metrics
        """
        metrics = {}

        # Get logits
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # 1. Layer-specific perplexity
        ppl_metrics = self.layer_perplexity.compute(logits, labels, attention_mask)
        metrics.update({f'metrics/{k}': v for k, v in ppl_metrics.items()})

        # 2. Output variance
        var_metrics = self.output_variance.compute()
        metrics.update({f'metrics/{k}': v for k, v in var_metrics.items()})
        self.output_variance.reset()

        # 3. Confusion matrix
        conf_metrics = self.confusion_matrix.compute(logits, labels, attention_mask)
        metrics.update({f'metrics/{k}': v for k, v in conf_metrics.items()})

        # 4. Token influence (expensive, run less frequently)
        if compute_influence:
            infl_metrics = self.token_influence.compute(model, input_ids, labels, attention_mask)
            metrics.update({f'metrics/{k}': v for k, v in infl_metrics.items()})

        return metrics

    def interpret_all(self, metrics: Dict[str, float]) -> str:
        """Generate human-readable interpretation of all metrics."""
        # Remove 'metrics/' prefix for interpretation
        clean_metrics = {k.replace('metrics/', ''): v for k, v in metrics.items()}

        lines = [
            "=" * 60,
            "ATTENTION ANALYSIS REPORT",
            "=" * 60,
            "",
            self.layer_perplexity.interpret(clean_metrics),
            "",
            self.output_variance.interpret(clean_metrics),
            "",
            self.confusion_matrix.interpret(clean_metrics),
            "",
        ]

        # Add token influence if present
        if any('influence' in k for k in clean_metrics):
            lines.extend([
                self.token_influence.interpret(clean_metrics),
                "",
            ])

        # Overall assessment
        lines.append("=" * 60)
        lines.append("OVERALL ASSESSMENT")
        lines.append("=" * 60)

        checks = []

        # Check perplexity variance
        ppl_var = clean_metrics.get('ppl_variance', float('nan'))
        if not np.isnan(ppl_var) and ppl_var < self.config.perplexity_variance_threshold:
            checks.append("✓ Perplexity consistent across layers")
        elif not np.isnan(ppl_var):
            checks.append("✗ Perplexity variance too high")

        # Check output variance
        output_var = clean_metrics.get('avg_output_variance', float('nan'))
        if not np.isnan(output_var) and output_var > self.config.output_variance_threshold:
            checks.append("✓ Attention outputs are focused")
        elif not np.isnan(output_var):
            checks.append("✗ Attention outputs too diffuse")

        # Check confusion diagonal
        min_diag = clean_metrics.get('min_diagonal', float('nan'))
        if not np.isnan(min_diag) and min_diag > self.config.confusion_diagonal_threshold:
            checks.append("✓ Excellent layer separation")
        elif not np.isnan(min_diag):
            checks.append("✗ Poor layer separation")

        lines.extend(checks)
        lines.append("=" * 60)

        return "\n".join(lines)
