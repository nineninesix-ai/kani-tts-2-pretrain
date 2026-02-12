#!/usr/bin/env python3
"""
Dataset preparation script for KaniTTS2-Pretrain.

This script loads datasets from HuggingFace, processes them, and saves
to disk for training. Configuration is read from configs/dataset_config.yaml.

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --output ./train_dataset
    python prepare_dataset.py --tokenizer-name LiquidAI/LFM2-350M --n-shards 8
"""

import argparse
from pathlib import Path

from dataset_processor import DatasetProcessor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare datasets for KaniTTS2 training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./train_dataset",
        help="Output directory for processed dataset"
    )

    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="LiquidAI/LFM2-350M",
        help="Tokenizer name for processing (default: LiquidAI/LFM2-350M)"
    )

    parser.add_argument(
        "--n-shards",
        type=int,
        default=20,
        help="Number of shards per dataset for parallel processing (default: auto)"
    )

    return parser.parse_args()


def main():
    """Main dataset preparation function."""
    args = parse_args()

    print("\n" + "=" * 70)
    print(" " * 20 + "Dataset Preparation")
    print(" " * 15 + "KaniTTS2-Pretrain")
    print("=" * 70 + "\n")

    # Initialize processor (it will load config internally)
    print(f"\n🔧 Initializing dataset processor...")
    print(f"   - Tokenizer: {args.tokenizer_name}")
    if args.n_shards:
        print(f"   - Shards per dataset: {args.n_shards}")

    processor = DatasetProcessor(
        tokenizer_name=args.tokenizer_name,
        n_shards_per_dataset=args.n_shards
    )

    # Load and process datasets (processor loads config internally)
    print(f"\n📊 Processing datasets...")
    dataset = processor()

    # Save to disk
    output_path = Path(args.output)
    print(f"\n💾 Saving processed dataset to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))

    # Save length statistics if enabled
    if processor.cfg.lenght_statistics:
        statistics_path = output_path / "audio_length_statistics.json"
        print(f"\n📊 Saving audio length statistics...")
        processor.save_length_statistics(str(statistics_path))

    print(f"\n✅ Dataset preparation complete!")
    print(f"   - Total samples: {len(dataset)}")
    print(f"   - Saved to: {output_path}")

    print("\n" + "=" * 70)
    print("🎉 Dataset is ready for training!")
    print("=" * 70)
    print(f"\nNext step: accelerate launch train.py --config-file configs/accelerate_config.yaml\n")


if __name__ == "__main__":
    main()
