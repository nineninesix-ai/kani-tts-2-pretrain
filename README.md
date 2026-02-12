<div align="center">
  <img src="public/logo.png" alt="Kani TTS Logo" width="150"/>

  [![](https://dcbadge.limes.pink/api/server/https://discord.gg/NzP3rjB4SB?style=flat)](https://discord.gg/NzP3rjB4SB) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

  # KaniTTS2 Pretrain
Pretraining system for KaniTTS2 models using the LFM2 hybrid architecture.
</div>

## Key Features

- **Frame-level Position Encoding**: All 4 tokens within an audio frame share the same position ID, reducing RoPE distance and improving long-form generation
- **Flash Attention 2 Optimized**: 10-20x faster training than eager attention
- **FSDP Multi-GPU Training**: Fully Sharded Data Parallel for efficient distributed training
- **OOP Architecture**: Clean, modular design with configuration-driven approach
- **YAML Configuration**: All settings in easy-to-edit YAML files

## Architecture

### Token Structure

- **Text tokens**: 0-64,399 (base vocabulary)
- **Special tokens**: 64,400-64,409 (10 tokens for SOT, EOT, etc.)
- **Audio tokens**: 64,410-80,537 (4 layers × 4032 = 16,128 tokens)
- **Total vocabulary**: 80,538 tokens

### Frame-level Position Encoding

The key innovation: all 4 tokens in an audio frame share the same position ID.

```python
# Text tokens: positions 0, 1, 2, ..., N
# Audio tokens: positions grouped by frame
# Frame 1: [N+1, N+1, N+1, N+1]  # All 4 tokens share position N+1
# Frame 2: [N+2, N+2, N+2, N+2]  # All 4 tokens share position N+2
```

This reduces the RoPE distance between tokens across frames, improving coherence in long-form generation.

## Quick Start

### 1. Setup Environment

```bash
make setup
```

Or manually:
```bash
./setup.sh
source venv/bin/activate
```

### 2. Configure Training

Edit configuration files in `configs/`:

**`configs/model_config.yaml`** - Model settings:
```yaml
model:
  model_id: "LiquidAI/LFM2-350M"
  first_train: true  # Set false when resuming from checkpoint
  attn_implementation: "flash_attention_2"
```

**`configs/training_config.yaml`** - Training hyperparameters:
```yaml
training:
  num_train_epochs: 3
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 9.0e-4
  save_steps: 5000
```

**`configs/dataset_config.yaml`** - Dataset sources:
```yaml
max_duration_sec: 20
hf_datasets:
  - reponame: "username/dataset-name"
    split: "train"
    # ... see file for full options
```

### 3. Prepare Dataset

```bash
make dataset
```

### 4. Train Model

```bash
make train
```

## Project Structure

```
KaniTTS2-Pretrain/
├── configs/                      # Configuration files
│   ├── model_config.yaml         # Model configuration
│   ├── training_config.yaml      # Training hyperparameters
│   ├── dataset_config.yaml       # Dataset sources
│   └── accelerate_config.yaml    # FSDP multi-GPU config
├── utils/                        # Core utilities (OOP)
│   ├── __init__.py
│   ├── config.py                 # Configuration classes
│   ├── model.py                  # Flash Attention LFM2 model
│   ├── data.py                   # Data collator and loading
│   └── trainer.py                # Trainer class
├── train.py                      # Main training script
├── prepare_dataset.py            # Dataset preparation script
├── upload_to_hf.py              # HuggingFace upload utility
├── dataset_processor.py          # Dataset processing logic
├── Makefile                      # Convenient commands
└── README.md                     # This file
```

## Configuration Reference

### Model Configuration (`configs/model_config.yaml`)

```yaml
model:
  model_id: "LiquidAI/LFM2-350M"           # Base model
  text_vocab_size: 64400                    # Original vocab size
  audio_codebook_size: 4032                 # Tokens per layer
  tokens_per_frame: 4                       # Tokens in one frame
  attn_implementation: "flash_attention_2"  # Attention type
  dtype: "bfloat16"                         # Model precision
  first_train: true                         # Resize embeddings?
```

**CRITICAL**: Set `first_train: false` when resuming from checkpoint!

### Training Configuration (`configs/training_config.yaml`)

```yaml
training:
  # Checkpointing
  output_dir: "./checkpoints"
  save_steps: 5000
  save_total_limit: 10

  # Training
  num_train_epochs: 3
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4

  # Optimizer
  learning_rate: 9.0e-4
  lr_scheduler_type: "cosine"
  warmup_steps: 2000

  # Data
  dataset_path: "./train_dataset"

wandb:
  project: "LFM2-nano-codec-custom-attention"
  name: "train_flash_attention_v1"

# Attention Metrics
metrics:
  enable_metrics: true           # Enable attention analysis
  log_steps: 100                 # Compute metrics every N steps
  influence_steps: 1000          # Compute token influence (expensive)
  print_interpretation: true     # Print results to console
```

## Attention Analysis Metrics

The system includes comprehensive metrics to verify that **frame-level position encoding** achieves layer isolation without explicit attention masking.

### Available Metrics

#### 1. Layer-Specific Perplexity
Measures prediction quality separately for each audio layer.

**Good layer isolation means:**
- Perplexity similar across layers (±10%)
- All layers < 200 PPL

**Thresholds:**
- PPL < 50: Excellent
- PPL 50-100: Good
- PPL 100-200: Acceptable
- PPL > 200: Poor

#### 2. Output Variance
Measures attention focus by tracking GQA layer output statistics.

**Good layer isolation means:**
- Higher variance = more focused attention
- Variance > 1.5 indicates good focus

**Thresholds:**
- Variance > 2.0: Excellent focus
- Variance 1.5-2.0: Good focus
- Variance 1.0-1.5: Acceptable
- Variance < 1.0: Poor (diffuse, likely mixing layers)

#### 3. Token Influence (Gradient-based)
Analyzes which tokens influence predictions using gradients.

**Good layer isolation means:**
- High influence from same-layer tokens (> 0.6)
- Low influence from other-layer tokens (< 0.2)
- Text tokens should have moderate influence

**Note:** This metric is expensive and runs less frequently (every 1000 steps by default).

#### 4. Cross-Layer Confusion Matrix
Shows if the model confuses different audio layers.

**Good layer isolation means:**
- Diagonal values > 0.8 (predicting correct layer)
- Off-diagonal values < 0.1 (not confusing layers)

### Configuration

Enable/disable metrics in `configs/training_config.yaml`:

```yaml
metrics:
  enable_metrics: true           # Set false to disable
  log_steps: 100                 # How often to compute metrics
  influence_steps: 1000          # Token influence interval (expensive)
  print_interpretation: true     # Print human-readable reports
```

### Interpreting Results

During training, you'll see reports like:

```
=== Layer-Specific Perplexity Analysis ===
Layer 0: 45.23 ✓ Excellent
Layer 1: 48.91 ✓ Excellent
Layer 2: 52.34 ✓ Good
Layer 3: 49.12 ✓ Excellent
Variance: 0.0234 ✓ Good layer isolation

=== Output Variance Analysis ===
Average variance: 2.134 ✓ Excellent focus
Range: [1.823, 2.445]

=== Cross-Layer Confusion Matrix ===
     L0    L1    L2    L3
L0   0.87  0.05  0.04  0.04
L1   0.03  0.89  0.05  0.03
L2   0.02  0.04  0.91  0.03
L3   0.03  0.02  0.04  0.91

Average diagonal: 0.890 ✓ Excellent layer separation
```

**Success Indicators:**
- ✓ All perplexities < 100
- ✓ Perplexity variance < 0.1
- ✓ Output variance > 1.5
- ✓ Confusion diagonal > 0.8

These metrics validate that frame-level position encoding successfully achieves layer isolation without explicit masking.

### Wandb Integration

All metrics are automatically logged to Wandb under the `metrics/` namespace:
- `metrics/layer_{0,1,2,3}_ppl`
- `metrics/avg_audio_ppl`
- `metrics/ppl_variance`
- `metrics/avg_output_variance`
- `metrics/confusion_{i}_{j}`
- `metrics/layer_{i}_same_layer_influence`
- `metrics/layer_{i}_cross_layer_influence`

## Makefile Commands

```bash
make help       # Show all commands
make setup      # Setup environment
make dataset    # Prepare dataset
make train      # Train model
make resume CHECKPOINT=./checkpoints/checkpoint-5000
make upload REPO=username/model CHECKPOINT=./checkpoints/checkpoint-5000
make clean      # Clean generated files
make test       # Test Flash Attention
make check-env  # Check environment
```

## Advanced Usage

### Custom Configuration Paths

```bash
python train.py \
  --model-config my_model.yaml \
  --training-config my_training.yaml \
  --dataset-config my_dataset.yaml
```

### Resume Training

```bash
make resume CHECKPOINT=./checkpoints/checkpoint-5000
```

Or directly:
```bash
accelerate launch train.py \
  --config-file configs/accelerate_config.yaml \
  --resume-from ./checkpoints/checkpoint-5000
```

### Using the Trainer Class

```python
from utils import load_configs, KaniTTS2Trainer

# Load configurations
model_config, training_config, _ = load_configs()

# Create trainer
trainer = KaniTTS2Trainer(model_config, training_config)

# Setup and train
trainer.setup()
trainer.train()

# Save model
trainer.save_model("./final_model")

# Or push to Hub
trainer.push_to_hub("username/model-name", private=True)
```

## Multi-GPU Training

Edit `configs/accelerate_config.yaml`:

```yaml
num_processes: 2  # Number of GPUs
fsdp_sharding_strategy: FULL_SHARD
mixed_precision: bf16
```

Training automatically uses FSDP with `accelerate launch`.

## Performance

On 2x H100 80GB GPUs:
- **Speed**: ~1.75 it/s (batch_size=8, gradient_accumulation=4)
- **Memory**: ~60GB per GPU with FSDP
- **GPU Utilization**: 85-92%
- **Speedup**: 10-20x vs eager attention

## Requirements

- Python 3.10+
- CUDA 12.1+
- PyTorch 2.8.0
- Flash Attention 2.8.3
- Transformers 4.56.0
- Accelerate 1.10.1

See [requirements.txt](requirements.txt) for complete list.

## Troubleshooting

### Token Embedding Mismatch

**Error**: `RuntimeError: index out of bounds`

**Fix**: Set `first_train: false` in `configs/model_config.yaml` when resuming from checkpoint.

### Flash Attention Errors

**Error**: `RuntimeError: FlashAttention only support fp16 and bf16`

**Fix**: Ensure `dtype: "bfloat16"` in model config.

### Out of Memory

**Fix**: Reduce `per_device_train_batch_size` or increase `gradient_accumulation_steps` in training config.

### Dataset Not Found

**Error**: `Dataset not found`

**Fix**: Run `make dataset` first to prepare the training dataset.

## Uploading Models

### Using Makefile

```bash
make upload REPO=username/model-name CHECKPOINT=./checkpoints/checkpoint-5000
```

### Using Script

```bash
python upload_to_hf.py \
  --repo username/model-name \
  --checkpoint ./checkpoints/checkpoint-5000 \
  --private
```

### Using Trainer

```python
trainer.push_to_hub("username/model-name", private=True)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{kanitts_2,
  author = {Nineninesix},
  title = {KaniTTS2: Text-to-Speech Model with Frame-level Position Encoding},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/nineninesix/kani-tts-2-pt}},
  note = {Open-source TTS model}
}
```

## License

Apache 2 - see LICENSE file for details
