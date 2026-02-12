#!/usr/bin/env python3
"""
Inspect learnable RoPE parameters from checkpoint.
Shows raw w values and computed alpha values for each layer.
"""

import torch
from safetensors import safe_open
import math

# Configuration from your model
ALPHA_MIN = 0.1
ALPHA_MAX = 2.0

def sigmoid(x):
    """Sigmoid function for alpha computation"""
    return 1.0 / (1.0 + math.exp(-x))

def compute_alpha(w):
    """Compute constrained alpha from raw w parameter"""
    return ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * sigmoid(w)

# Load checkpoint
checkpoint_path = "checkpoints/checkpoint-5000/model.safetensors"

print("=" * 70)
print("Learnable RoPE Parameters Analysis - Checkpoint 5000")
print("=" * 70)
print()

# Load safetensors
with safe_open(checkpoint_path, framework="pt") as f:
    # Get all keys
    all_keys = f.keys()

    # Find learnable_rope related keys
    rope_keys = [k for k in all_keys if 'learnable_rope' in k]

    if not rope_keys:
        print("❌ No learnable_rope parameters found in checkpoint!")
        print("   Are you sure use_learnable_rope was enabled during training?")
        print()
        print("All available keys:")
        for key in sorted(all_keys):
            print(f"  - {key}")
    else:
        print(f"✅ Found {len(rope_keys)} learnable RoPE parameters\n")

        # Extract layer indices and alpha_weight values
        layer_data = []

        for key in sorted(rope_keys):
            tensor = f.get_tensor(key)

            # Parse layer index from key
            # Expected format: model.learnable_rope_layers.{idx}.alpha_weight
            if 'alpha_weight' in key:
                parts = key.split('.')
                layer_idx = int(parts[2])  # Extract index

                w_value = tensor.item()
                alpha_value = compute_alpha(w_value)

                layer_data.append((layer_idx, w_value, alpha_value))

        # Display results
        print("Layer-wise Learnable RoPE Parameters:")
        print("-" * 70)
        print(f"{'Layer':<10} {'Raw w':<20} {'Alpha (constrained)':<20} {'Change from init'}")
        print("-" * 70)

        # Initial alpha is ~1.05, so initial w ≈ 0.0
        for layer_idx, w, alpha in layer_data:
            w_change = w - 0.0  # Change from initialization
            alpha_change = alpha - 1.05  # Change from initial alpha

            # Visual indicator
            if abs(alpha_change) < 0.01:
                indicator = "≈ (no change)"
            elif alpha_change > 0:
                indicator = f"↑ +{alpha_change:.4f}"
            else:
                indicator = f"↓ {alpha_change:.4f}"

            print(f"{layer_idx:<10} {w:<20.6f} {alpha:<20.6f} {indicator}")

        print("-" * 70)
        print()

        # Statistics
        alphas = [alpha for _, _, alpha in layer_data]
        ws = [w for _, w, _ in layer_data]

        print("Statistics:")
        print(f"  Alpha range: [{min(alphas):.4f}, {max(alphas):.4f}]")
        print(f"  Alpha mean:  {sum(alphas)/len(alphas):.4f}")
        print(f"  Alpha std:   {(sum((a - sum(alphas)/len(alphas))**2 for a in alphas) / len(alphas))**0.5:.4f}")
        print()
        print(f"  Raw w range: [{min(ws):.6f}, {max(ws):.6f}]")
        print(f"  Raw w mean:  {sum(ws)/len(ws):.6f}")
        print()

        # Interpretation
        print("Interpretation:")
        print(f"  α > 1.0 → Faster rotation (stronger positional encoding)")
        print(f"  α ≈ 1.0 → Standard RoPE (no change)")
        print(f"  α < 1.0 → Slower rotation (weaker positional encoding)")
        print()

        # Layer-wise patterns
        if len(layer_data) >= 2:
            early_layers = layer_data[:len(layer_data)//2]
            late_layers = layer_data[len(layer_data)//2:]

            avg_early_alpha = sum(a for _, _, a in early_layers) / len(early_layers)
            avg_late_alpha = sum(a for _, _, a in late_layers) / len(late_layers)

            print("Layer Pattern Analysis:")
            print(f"  Early layers (avg α): {avg_early_alpha:.4f}")
            print(f"  Late layers (avg α):  {avg_late_alpha:.4f}")

            if avg_early_alpha > avg_late_alpha + 0.05:
                print("  → Early layers prefer stronger positional encoding (local patterns)")
            elif avg_late_alpha > avg_early_alpha + 0.05:
                print("  → Late layers prefer stronger positional encoding (global context)")
            else:
                print("  → Similar positional encoding across layers")

print()
print("=" * 70)
