#!/usr/bin/env python3
"""
Main training script for KaniTTS2-Pretrain.

This script loads configuration from YAML files and runs the training process.

Usage:
    # Train from scratch
    accelerate launch train.py --config-file configs/accelerate_config.yaml

    # Resume from checkpoint
    accelerate launch train.py --config-file configs/accelerate_config.yaml --resume-from checkpoint-5000

    # Custom config paths
    python train.py --model-config configs/model_config.yaml --training-config configs/training_config.yaml
"""

import argparse
from pathlib import Path

from utils import load_configs, KaniTTS2Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train KaniTTS2 model with frame-level position encoding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model configuration YAML file (default: configs/model_config.yaml)"
    )

    parser.add_argument(
        "--training-config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration YAML file (default: configs/training_config.yaml)"
    )

    parser.add_argument(
        "--dataset-config",
        type=str,
        default="configs/dataset_config.yaml",
        help="Path to dataset configuration YAML file (default: configs/dataset_config.yaml)"
    )

    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from (e.g., checkpoints/checkpoint-5000)"
    )

    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Accelerate config file (handled by accelerate launch, included here for compatibility)"
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Print header
    print("\n" + "=" * 70)
    print(" " * 20 + "KaniTTS2-Pretrain")
    print(" " * 10 + "Frame-level Position Encoding for TTS Models")
    print("=" * 70 + "\n")

    # Load configurations
    print("📋 Loading configurations...")
    model_config, training_config, dataset_config = load_configs(
        model_config_path=args.model_config,
        training_config_path=args.training_config,
        dataset_config_path=args.dataset_config,
    )
    print(f"✅ Configurations loaded")
    print(f"   - Model config: {args.model_config}")
    print(f"   - Training config: {args.training_config}")
    print(f"   - Dataset config: {args.dataset_config}")

    # Create trainer
    print("\n🔧 Initializing trainer...")
    trainer = KaniTTS2Trainer(
        model_config=model_config,
        training_config=training_config,
    )

    # Setup all components
    trainer.setup()

    # Start training
    trainer.train(resume_from_checkpoint=args.resume_from)

    # Save final model
    final_output_dir = Path(training_config.output_dir) / "final"
    trainer.save_model(str(final_output_dir))

    print("\n" + "=" * 70)
    print("🎉 Training pipeline completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
