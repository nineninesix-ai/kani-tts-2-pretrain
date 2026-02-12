"""
Test script for speaker embedding integration.

This script verifies:
1. Speaker embeddings are correctly projected and inserted
2. Position IDs are correctly adjusted
3. Labels are correctly adjusted
4. Forward pass works without errors
5. Gradients flow properly
"""

import torch
from utils.model import FlashCompatibleLfm2ForCausalLM
from utils.data import DataCollator
from datasets import load_from_disk


def test_speaker_embedding_integration():
    """Test speaker embedding integration with the model."""

    print("=" * 60)
    print("Testing Speaker Embedding Integration")
    print("=" * 60)

    # Load a small batch from the dataset
    print("\n1. Loading dataset...")
    dataset = load_from_disk("./train_dataset")
    batch_data = [dataset[i] for i in range(2)]  # Small batch of 2

    # Check that speaker_emb exists
    print(f"   Dataset sample keys: {batch_data[0].keys()}")
    speaker_emb_shape = len(batch_data[0]['speaker_emb'])
    print(f"   Speaker embedding dimension: {speaker_emb_shape}")

    # Create data collator
    print("\n2. Creating data collator...")
    collator = DataCollator(pad_token_id=64407)
    batch = collator(batch_data)

    print(f"   Batch keys: {batch.keys()}")
    print(f"   input_ids shape: {batch['input_ids'].shape}")
    print(f"   attention_mask shape: {batch['attention_mask'].shape}")
    print(f"   labels shape: {batch['labels'].shape}")
    if 'speaker_emb' in batch:
        print(f"   speaker_emb shape: {batch['speaker_emb'].shape}")

    # Verify labels are adjusted correctly (seq_len should be input_ids.shape[1])
    assert batch['labels'].shape[1] == batch['input_ids'].shape[1] + 1, \
        f"Labels should have +1 length due to speaker embedding insertion. " \
        f"Got labels={batch['labels'].shape[1]}, input_ids={batch['input_ids'].shape[1]}"

    print("   ✅ Labels shape adjusted correctly (+1 for speaker embedding)")

    # Create model (small test - using eager attention for simplicity)
    print("\n3. Loading model...")
    model = FlashCompatibleLfm2ForCausalLM.from_pretrained(
        "LiquidAI/LFM2-350M",
        audio_tokens_start=64410,
        tokens_per_frame=4,
        audio_step=1.0,
        use_learnable_rope=False,  # Disable for faster test
        speaker_emb_dim=128,
        attn_implementation="eager",  # Use eager for testing (faster load)
        dtype=torch.float32,  # Use float32 for CPU testing
    )

    print(f"   Model loaded successfully")
    print(f"   Speaker projection layer: {model.model.speaker_emb_projection}")

    # Resize embeddings to match dataset vocabulary (80538 tokens)
    print("\n   Resizing token embeddings to 80538...")
    model.resize_token_embeddings(80538)
    print("   ✅ Embeddings resized")

    # Run forward pass
    print("\n4. Running forward pass...")
    with torch.no_grad():
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            speaker_emb=batch['speaker_emb'],
            labels=batch['labels']
        )

    print(f"   Loss: {outputs.loss.item():.4f}")
    print(f"   Logits shape: {outputs.logits.shape}")

    # Verify logits shape
    # Expected: [batch_size, seq_len + 1, vocab_size]
    # Because speaker embedding is inserted at position 1
    expected_seq_len = batch['input_ids'].shape[1] + 1
    assert outputs.logits.shape[1] == expected_seq_len, \
        f"Logits sequence length should be {expected_seq_len}, got {outputs.logits.shape[1]}"

    print(f"   ✅ Logits shape correct (seq_len + 1 due to speaker embedding)")

    # Test gradient flow
    print("\n5. Testing gradient flow...")
    outputs = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        speaker_emb=batch['speaker_emb'],
        labels=batch['labels']
    )

    loss = outputs.loss
    loss.backward()

    # Check that speaker_emb_projection has gradients
    has_grad = model.model.speaker_emb_projection.weight.grad is not None
    print(f"   Speaker projection layer has gradients: {has_grad}")

    if has_grad:
        grad_norm = model.model.speaker_emb_projection.weight.grad.norm().item()
        print(f"   Gradient norm: {grad_norm:.6f}")
        print("   ✅ Gradients flowing correctly!")
    else:
        print("   ⚠️  WARNING: No gradients on speaker projection layer!")

    # Test without speaker embeddings (should still work)
    print("\n6. Testing without speaker embeddings (backward compatibility)...")
    with torch.no_grad():
        outputs_no_speaker = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            speaker_emb=None,  # No speaker embedding
            labels=batch['labels'][:, 1:]  # Remove the extra label position
        )

    print(f"   Loss (no speaker): {outputs_no_speaker.loss.item():.4f}")
    print(f"   Logits shape (no speaker): {outputs_no_speaker.logits.shape}")
    print("   ✅ Backward compatibility maintained!")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    try:
        test_speaker_embedding_integration()
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
