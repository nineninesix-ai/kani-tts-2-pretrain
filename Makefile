.PHONY: help setup dataset train resume upload clean test check-env

# Use bash instead of sh for source command
SHELL := /bin/bash

# Default target
help:
	@echo "KaniTTS2-Pretrain Makefile"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup              - Setup virtual environment and install dependencies"
	@echo "  make dataset            - Prepare training dataset from HuggingFace"
	@echo "  make train              - Start training with Flash Attention 2"
	@echo "  make resume CHECKPOINT=path - Resume training from checkpoint"
	@echo "  make upload REPO=user/model CHECKPOINT=path - Upload model to HuggingFace Hub"
	@echo "  make clean              - Remove generated files (checkpoints, datasets, cache)"
	@echo "  make test               - Test Flash Attention installation"
	@echo "  make check-env          - Check environment and dependencies"
	@echo ""
	@echo "Examples:"
	@echo "  make train"
	@echo "  make resume CHECKPOINT=./checkpoints/checkpoint-5000"
	@echo "  make upload REPO=username/model-name CHECKPOINT=./checkpoints/checkpoint-5000"

# Setup environment
setup:
	@echo "🔧 Setting up environment..."
	@sudo ./setup.sh
	@echo "✅ Setup complete!"

login:
	@echo "🔧 HF and Wandb loging"
	@source venv/bin/activate && git config --global credential.helper store
	@source venv/bin/activate && hf auth login
	@source venv/bin/activate && wandb login
	@echo "✅ Complete"

# Prepare dataset
dataset:
	@echo "📊 Preparing training dataset..."
	@source venv/bin/activate && python prepare_dataset.py
	@echo "✅ Dataset prepared in ./train_dataset/"

# Train model
train:
	@echo "🚀 Starting training with Flash Attention 2..."
	@source venv/bin/activate && accelerate launch train.py --config-file configs/accelerate_config.yaml

# Resume training from checkpoint
resume:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "❌ Error: Please specify CHECKPOINT=path"; \
		echo "Example: make resume CHECKPOINT=./checkpoints/checkpoint-5000"; \
		exit 1; \
	fi
	@if [ ! -d "$(CHECKPOINT)" ]; then \
		echo "❌ Error: Checkpoint directory not found: $(CHECKPOINT)"; \
		exit 1; \
	fi
	@echo "🔄 Resuming training from $(CHECKPOINT)..."
	@source venv/bin/activate && accelerate launch train.py \
		--config-file configs/accelerate_config.yaml \
		--resume-from $(CHECKPOINT)

# Upload to Hugging Face Hub
upload:
	@if [ -z "$(REPO)" ] || [ -z "$(CHECKPOINT)" ]; then \
		echo "❌ Error: Please specify REPO and CHECKPOINT"; \
		echo "Example: make upload REPO=username/model-name CHECKPOINT=./checkpoints/checkpoint-5000"; \
		exit 1; \
	fi
	@if [ ! -d "$(CHECKPOINT)" ]; then \
		echo "❌ Error: Checkpoint directory not found: $(CHECKPOINT)"; \
		exit 1; \
	fi
	@echo "📤 Uploading $(CHECKPOINT) to $(REPO)..."
	@source venv/bin/activate && python upload_to_hf.py --repo $(REPO) --checkpoint $(CHECKPOINT) --private

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	@rm -rf checkpoints/
	@rm -rf train_dataset/
	@rm -rf wandb/
	@rm -rf __pycache__/
	@rm -rf utils/__pycache__/
	@rm -rf .pytest_cache/
	@rm -f training_log*.txt
	@echo "✅ Cleaned!"

# Test Flash Attention installation
test:
	@echo "🔍 Testing Flash Attention installation..."
	@source venv/bin/activate && python -c "import torch; import flash_attn; print(f'✅ Flash Attention {flash_attn.__version__} installed'); print(f'✅ PyTorch {torch.__version__} with CUDA {torch.version.cuda}')"

# Check environment and dependencies
check-env:
	@echo "🔍 Checking environment..."
	@echo ""
	@echo "Python version:"
	@source venv/bin/activate && python --version
	@echo ""
	@echo "CUDA devices:"
	@nvidia-smi --query-gpu=index,name,memory.total --format=csv
	@echo ""
	@echo "Installed packages:"
	@source venv/bin/activate && pip list | grep -E "torch|transformers|accelerate|flash-attn|wandb|datasets"
	@echo ""
	@echo "Disk space:"
	@df -h . | tail -1
	@echo ""
	@echo "Memory:"
	@free -h | grep Mem
