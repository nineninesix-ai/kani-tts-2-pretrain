"""
Data utilities for KaniTTS2-Pretrain.
"""

import torch
from datasets import load_from_disk, Dataset
from typing import List, Dict, Any
from pathlib import Path


class DataCollator:
    """
    Custom data collator for padding sequences.

    Pads input_ids, attention_mask, and labels to the same length in a batch.
    """

    def __init__(self, pad_token_id: int):
        """
        Args:
            pad_token_id: Token ID to use for padding
        """
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features.

        Args:
            features: List of dictionaries with 'input_ids', 'attention_mask', 'labels', 'speaker_emb'

        Returns:
            Dictionary with padded tensors and speaker embeddings
        """
        input_ids = [f["input_ids"] for f in features]

        # Get or create attention masks
        if any("attention_mask" not in f for f in features):
            attention_mask = [[1] * len(ids) for ids in input_ids]
        else:
            attention_mask = [f["attention_mask"] for f in features]

        # Get or create labels
        if any("labels" not in f for f in features):
            labels = input_ids
        else:
            labels = [f["labels"] for f in features]

        # Extract speaker embeddings (fixed size, no padding needed)
        speaker_emb = None
        if all("speaker_emb" in f for f in features):
            speaker_emb = torch.tensor(
                [f["speaker_emb"] for f in features],
                dtype=torch.float32
            )  # Shape: [batch_size, speaker_emb_dim]

        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(i, dtype=torch.long) for i in input_ids],
            batch_first=True,
            padding_value=self.pad_token_id
        )

        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(m, dtype=torch.long) for m in attention_mask],
            batch_first=True,
            padding_value=0
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(l, dtype=torch.long) for l in labels],
            batch_first=True,
            padding_value=-100  # Ignore index for loss
        )

        # Adjust labels to account for speaker embedding insertion at position 1
        # Original: [label_0, label_1, label_2, ...]
        # New:      [label_0, -100 (speaker), label_1, label_2, ...]
        if speaker_emb is not None:
            # Insert -100 at position 1 for speaker embedding
            batch_size, seq_len = labels.shape
            labels_adjusted = torch.full(
                (batch_size, seq_len + 1),
                fill_value=-100,
                dtype=torch.long
            )
            # Copy original labels with shift
            labels_adjusted[:, 0] = labels[:, 0]          # First label stays at position 0
            labels_adjusted[:, 2:] = labels[:, 1:]        # Rest shift by +1
            # Position 1 is already -100 (speaker embedding - no loss)
            labels = labels_adjusted

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

        # Add speaker embeddings if present
        if speaker_emb is not None:
            batch["speaker_emb"] = speaker_emb

        return batch


def prepare_dataset(dataset_path: str) -> Dataset:
    """
    Load and prepare the training dataset.

    Args:
        dataset_path: Path to the processed dataset directory

    Returns:
        Loaded dataset

    Raises:
        FileNotFoundError: If dataset path doesn't exist
        ValueError: If dataset is empty
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            f"Please run 'make dataset' or 'python prepare_dataset.py' first."
        )

    print(f"📊 Loading dataset from {dataset_path}")
    dataset = load_from_disk(str(dataset_path))
    dataset = dataset.shuffle()

    if len(dataset) == 0:
        raise ValueError(f"Dataset at {dataset_path} is empty!")

    print(f"✅ Loaded {len(dataset)} samples")

    return dataset
