#!/usr/bin/env python3
"""
Upload trained model checkpoint to Hugging Face Hub.

Usage:
    python upload_to_hf.py --repo username/model-name --checkpoint ./checkpoints_custom/checkpoint-5000 --private
    python upload_to_hf.py --repo username/model-name --checkpoint ./checkpoints_custom/checkpoint-5000 --public
"""

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer
import yaml


def upload_checkpoint(repo_id: str, checkpoint_path: str, private: bool = True):
    """
    Upload a model checkpoint with tokenizer to Hugging Face Hub.

    Args:
        repo_id: Repository ID (e.g., "username/model-name")
        checkpoint_path: Path to checkpoint directory
        private: Whether to make the repository private
    """
    checkpoint_path = Path(checkpoint_path)

    # Validate checkpoint path
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint path does not exist: {checkpoint_path}")
        sys.exit(1)

    if not checkpoint_path.is_dir():
        print(f"❌ Error: Checkpoint path is not a directory: {checkpoint_path}")
        sys.exit(1)

    # Check for required model files
    required_files = ["config.json", "model.safetensors"]
    missing_files = []
    for file in required_files:
        if not (checkpoint_path / file).exists():
            # Try .bin format
            if file == "model.safetensors" and (checkpoint_path / "pytorch_model.bin").exists():
                continue
            missing_files.append(file)

    if missing_files:
        print(f"⚠️  Warning: Missing model files in checkpoint: {missing_files}")
        print("Continuing anyway...")

    # Check for tokenizer files
    tokenizer_files = ["tokenizer_config.json", "tokenizer.json"]
    has_tokenizer = any((checkpoint_path / f).exists() for f in tokenizer_files)

    if not has_tokenizer:
        print("\n⚠️  No tokenizer found in checkpoint directory")
        print("🔧 Attempting to load tokenizer from base model config...")

        # Try to load model_config.yaml to get base model ID
        try:
            config_path = Path("configs/model_config.yaml")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    base_model_id = config.get('model', {}).get('model_id', 'LiquidAI/LFM2-350M')

                print(f"📝 Loading tokenizer from base model: {base_model_id}")
                tokenizer = AutoTokenizer.from_pretrained(base_model_id)

                # Save tokenizer to checkpoint directory
                print(f"💾 Saving tokenizer to checkpoint directory...")
                tokenizer.save_pretrained(checkpoint_path)
                print("✅ Tokenizer saved to checkpoint")
            else:
                print("⚠️  Could not find model_config.yaml. Skipping tokenizer upload.")
        except Exception as e:
            print(f"⚠️  Failed to load and save tokenizer: {e}")
            print("Continuing without tokenizer...")

    print(f"\n📤 Uploading checkpoint to Hugging Face Hub")
    print(f"   Repository: {repo_id}")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Private: {private}")
    print()

    # Create repository if it doesn't exist
    try:
        api = HfApi()
        create_repo(repo_id, private=private, exist_ok=True)
        print(f"✅ Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        sys.exit(1)

    # Upload all files in checkpoint directory
    try:
        print("\n📁 Uploading files...")

        # List files being uploaded
        files_to_upload = list(checkpoint_path.glob("*"))
        print(f"   Files to upload ({len(files_to_upload)}):")
        for f in files_to_upload:
            if f.is_file():
                print(f"     - {f.name}")

        api.upload_folder(
            folder_path=str(checkpoint_path),
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"\n✅ Upload completed!")
        print(f"🔗 View at: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"\n❌ Error uploading files: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Upload model checkpoint to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload private checkpoint
  python upload_to_hf.py --repo username/model-name --checkpoint ./checkpoints_custom/checkpoint-5000 --private

  # Upload public checkpoint
  python upload_to_hf.py --repo username/model-name --checkpoint ./checkpoints_custom/checkpoint-5000 --public
        """
    )

    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Hugging Face repository ID (e.g., username/model-name)"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory"
    )

    privacy_group = parser.add_mutually_exclusive_group()
    privacy_group.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="Make repository private (default)"
    )
    privacy_group.add_argument(
        "--public",
        action="store_true",
        help="Make repository public"
    )

    args = parser.parse_args()

    # Determine privacy setting
    private = not args.public

    upload_checkpoint(
        repo_id=args.repo,
        checkpoint_path=args.checkpoint,
        private=private
    )


if __name__ == "__main__":
    main()
