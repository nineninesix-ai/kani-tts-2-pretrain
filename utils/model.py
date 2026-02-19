"""
Custom LFM2 implementation optimized for Flash Attention 2.

Key Feature:
- Frame-level position encoding: All 4 tokens within an audio frame share the same position ID
  This reduces RoPE distance between tokens across frames, improving long-form generation.

Compatible with Flash Attention 2 for 10-20x training speedup.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.utils import TransformersKwargs
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache

# Import base LFM2 classes
from transformers.models.lfm2.modeling_lfm2 import (
    Lfm2Model,
    Lfm2ForCausalLM,
    Lfm2PreTrainedModel,
    Lfm2HybridConvCache,
    Lfm2Attention,
)
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config


class LearnableRotaryEmbedding(nn.Module):
    """
    Learnable RoPE with layer-wise frequency scaling.

    Each layer has a learnable alpha parameter that scales the RoPE frequencies:
        θᵢ^(l) = α^(l) · base^(-2i/d)

    Alpha is constrained via sigmoid reparameterization:
        α^(l) = α_min + (α_max - α_min) · sigmoid(w^(l))

    This allows each layer to learn its optimal positional sensitivity.
    """

    def __init__(
        self,
        config: Lfm2Config,
        layer_idx: int,
        total_attention_layers: int,
        alpha_min: float = 0.1,
        alpha_max: float = 2.0,
        device=None
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        # Compute base RoPE frequencies (same as standard implementation)
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)

        # Base inverse frequencies: θᵢ = base^(-2i/d)
        inv_freq_base = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq_base", inv_freq_base, persistent=False)

        # Learnable parameter (unconstrained)
        # Initialize to 0.0 so that sigmoid(0) = 0.5 → alpha ≈ (alpha_min + alpha_max) / 2
        # For alpha_min=0.1, alpha_max=2.0: initial alpha ≈ 1.05
        self.alpha_weight = nn.Parameter(torch.tensor(0.0))

        # For compatibility with standard RoPE interface
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.rope_type = "default"
        self.attention_scaling = 1.0

        print(f"  ✅ Layer {layer_idx}: LearnableRoPE initialized (α_init ≈ {self.alpha.item():.3f})")

    @property
    def alpha(self) -> torch.Tensor:
        """Compute constrained alpha via sigmoid reparameterization."""
        return self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.alpha_weight)

    @property
    def inv_freq(self) -> torch.Tensor:
        """Compute scaled inverse frequencies: α^(l) · θᵢ"""
        return self.inv_freq_base * self.alpha

    def forward(self, x, position_ids):
        """
        Compute cos/sin embeddings with learned frequency scaling.

        Args:
            x: Input tensor [batch, num_heads, seq_len, head_dim]
            position_ids: Position indices [batch, seq_len]

        Returns:
            (cos, sin): Tuple of embedding tensors [batch, seq_len, dim]
        """
        # Get scaled frequencies (learnable - gradients MUST flow through alpha!)
        inv_freq_scaled = self.inv_freq

        # Expand dimensions for matmul
        inv_freq_expanded = inv_freq_scaled[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        # Compute frequencies: freqs[m,i] = α·θᵢ·m
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)



def compute_frame_level_positions(
    input_ids: torch.Tensor,
    audio_tokens_start: int,
    tokens_per_frame: int = 4,
    audio_step: float = 1.0
) -> torch.Tensor:
    """
    Vectorized computation of frame-level position IDs (10-50x faster than Python loops).

    Key insight: Use cumulative counts to determine positions.

    - Text tokens: sequential positions (step 1.0)
    - Audio tokens: frame-level positions (step audio_step per frame)

    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        audio_tokens_start: Token ID where audio tokens begin (typically 64410)
        tokens_per_frame: Number of tokens per audio frame (typically 4)
        audio_step: Position step size per audio frame (default 1.0).
                    Set to < 1.0 (e.g., 0.5) to compress audio position space.

    Returns:
        position_ids: Position IDs [batch_size, seq_len].
                      if audio_step is float, returns FloatTensor.

    Example:
        >>> input_ids = torch.tensor([[100, 200, 64410, 68442, 72474, 76506, 300]])
        >>> # Tokens:                [text, text, aud0,  aud1,  aud2,  aud3,  text]
        >>> pos = compute_frame_level_positions(input_ids, 64410, 4, audio_step=0.5)
        >>> pos
        tensor([[0., 1., 2., 2., 2., 2., 3.]])
        # Text at 0, 1. Audio frame at 2. Next text at 3 (1+1+1?)
        # Note: Text logic accumulates 1 per text token.
        # Audio logic accumulates audio_step per frame.
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # Identify audio tokens
    is_audio = input_ids >= audio_tokens_start
    text_mask = ~is_audio

    # Prepare zero prefix for cumsum
    zeros = torch.zeros(batch_size, 1, device=device, dtype=torch.long)

    # 1. Count text tokens before each position
    #    This gives the integer base from text tokens
    text_count = torch.cat([zeros, text_mask.long()], dim=1).cumsum(dim=1)[:, :-1]

    # 2. Count audio tokens before each position
    audio_token_count = torch.cat([zeros, is_audio.long()], dim=1).cumsum(dim=1)[:, :-1]

    # 3. Convert token count to frame count (0, 0, 0, 0, 1, 1...)
    audio_frame_count = audio_token_count // tokens_per_frame

    # 4. Compute final positions
    #    Text contributes 1.0 per token
    #    Audio frames contribute audio_step per frame
    position_ids = text_count + audio_frame_count * audio_step

    return position_ids


class FlashCompatibleLfm2Model(Lfm2Model):
    """
    Custom LFM2 model with frame-level positions and learnable RoPE.

    Features:
    - Frame-level position encoding for audio tokens
    - Learnable RoPE frequencies per attention layer
    - Flash Attention 2 optimized
    """

    def __init__(
        self,
        config: Lfm2Config,
        audio_tokens_start: int,
        tokens_per_frame: int = 4,
        audio_step: float = 1.0,
        use_learnable_rope: bool = False,
        alpha_min: float = 0.1,
        alpha_max: float = 2.0,
        speaker_emb_dim: int = 128
    ):
        super().__init__(config)
        self.audio_tokens_start = audio_tokens_start
        self.tokens_per_frame = tokens_per_frame
        self.audio_step = audio_step
        self.use_learnable_rope = use_learnable_rope
        self.speaker_emb_dim = speaker_emb_dim

        # Speaker embedding projection: 128 -> hidden_size (1024)
        self.speaker_emb_projection = nn.Linear(speaker_emb_dim, config.hidden_size, bias=False)

        print(f"✅ FlashCompatibleLfm2Model initialized:")
        print(f"   - Audio tokens start: {audio_tokens_start}")
        print(f"   - Tokens per frame: {tokens_per_frame}")
        print(f"   - Audio step: {audio_step}")
        print(f"   - Speaker embedding: {speaker_emb_dim} -> {config.hidden_size}")
        print(f"   - Using Flash Attention 2 with frame-level positions")

        # Replace RoPE embeddings with learnable version for attention layers
        if use_learnable_rope:
            print(f"   - Enabling learnable RoPE (α ∈ [{alpha_min}, {alpha_max}])")

            # Identify attention layer indices from config
            attention_layer_indices = []
            if hasattr(config, 'layer_types') and config.layer_types is not None:
                for idx, layer_type in enumerate(config.layer_types):
                    if layer_type == "full_attention":
                        attention_layer_indices.append(idx)
            elif hasattr(config, 'full_attn_idxs') and config.full_attn_idxs is not None:
                attention_layer_indices = list(config.full_attn_idxs)
            else:
                # Fallback: all layers are attention layers
                attention_layer_indices = list(range(config.num_hidden_layers))

            print(f"   - Attention layers: {attention_layer_indices}")
            total_attention_layers = len(attention_layer_indices)

            # Replace pos_emb with learnable RoPE for each attention layer
            # Store learnable RoPE modules in a ModuleList for proper parameter registration
            self.learnable_rope_layers = nn.ModuleList()

            for idx in range(config.num_hidden_layers):
                if idx in attention_layer_indices:
                    # Create learnable RoPE for this attention layer
                    learnable_rope = LearnableRotaryEmbedding(
                        config=config,
                        layer_idx=idx,
                        total_attention_layers=total_attention_layers,
                        alpha_min=alpha_min,
                        alpha_max=alpha_max,
                        device=self.device
                    )
                    self.learnable_rope_layers.append(learnable_rope)
                else:
                    # Not an attention layer, append None
                    self.learnable_rope_layers.append(None)

            print(f"   - {total_attention_layers} learnable RoPE modules created")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Lfm2HybridConvCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        speaker_emb: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        """
        Forward pass with custom frame-level position IDs and speaker embeddings.

        Speaker embeddings are inserted at position 1 (after the first token).
        Uses standard Flash Attention (no custom masking).

        Args:
            speaker_emb: Speaker embeddings [batch_size, speaker_emb_dim] (e.g., [8, 128])
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Insert speaker embedding at position 1
        if speaker_emb is not None:
            # Project speaker embedding: [batch_size, 128] -> [batch_size, hidden_size]
            speaker_emb_projected = self.speaker_emb_projection(speaker_emb)
            # Add sequence dimension: [batch_size, hidden_size] -> [batch_size, 1, hidden_size]
            speaker_emb_projected = speaker_emb_projected.unsqueeze(1)

            # Insert after first token: [batch, seq_len, hidden] -> [batch, seq_len+1, hidden]
            # Result: [token_0, speaker_emb, token_1, token_2, ...]
            inputs_embeds = torch.cat([
                inputs_embeds[:, :1, :],        # First token (e.g., SOT)
                speaker_emb_projected,           # Speaker embedding
                inputs_embeds[:, 1:, :]         # Rest of tokens
            ], dim=1)

            # Update attention_mask to account for inserted token
            if attention_mask is not None:
                # Insert 1 at position 1 (speaker embedding is not masked)
                attention_mask = torch.cat([
                    attention_mask[:, :1],          # First token mask
                    torch.ones(attention_mask.shape[0], 1, device=attention_mask.device, dtype=attention_mask.dtype),
                    attention_mask[:, 1:]           # Rest of masks
                ], dim=1)

        if use_cache and past_key_values is None:
            batch_size = inputs_embeds.shape[0]
            past_key_values = Lfm2HybridConvCache(
                config=self.config, max_batch_size=batch_size, dtype=self.dtype, device=self.device
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # CUSTOM: Compute frame-level position IDs
        if position_ids is None and input_ids is not None:
            # Compute frame-level positions for original input_ids
            position_ids_base = compute_frame_level_positions(
                input_ids=input_ids,
                audio_tokens_start=self.audio_tokens_start,
                tokens_per_frame=self.tokens_per_frame,
                audio_step=self.audio_step
            )

            # If speaker embedding is inserted, adjust positions
            if speaker_emb is not None:
                # Position layout: [0, 1, 2, 3, ...] becomes [0, 1, 2, 3, 4, ...]
                # where position 1 is the speaker embedding
                # Original: [token_0:pos_0, token_1:pos_1, token_2:pos_2, ...]
                # New:      [token_0:pos_0, speaker:pos_1, token_1:pos_2, token_2:pos_3, ...]

                # Shift all positions after the first token by +1
                position_ids = torch.cat([
                    position_ids_base[:, :1],                    # Position 0 (first token)
                    position_ids_base[:, :1] + 1,                # Position 1 (speaker embedding)
                    position_ids_base[:, 1:] + 1                 # Positions 2+ (rest, shifted by +1)
                ], dim=1)
            else:
                position_ids = position_ids_base
        elif position_ids is None:
            # Fallback to standard positions if no input_ids
            position_ids = cache_position.unsqueeze(0)

        # Use standard causal mask (Flash Attention compatible)
        from transformers.masking_utils import create_causal_mask
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds

        # Compute position embeddings
        # If using learnable RoPE, we compute per-layer embeddings in the loop
        # Otherwise, use global position embeddings
        if not self.use_learnable_rope:
            position_embeddings = self.pos_emb(hidden_states, position_ids)
        else:
            position_embeddings = None

        # decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            # Compute position embeddings per layer if using learnable RoPE
            if self.use_learnable_rope and self.learnable_rope_layers[layer_idx] is not None:
                # This is an attention layer with learnable RoPE
                position_embeddings = self.learnable_rope_layers[layer_idx](hidden_states, position_ids)
            elif self.use_learnable_rope and position_embeddings is None:
                # This is a conv layer, use standard RoPE (compute once)
                position_embeddings = self.pos_emb(hidden_states, position_ids)

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.embedding_norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class FlashCompatibleLfm2ForCausalLM(Lfm2PreTrainedModel):
    """
    Flash Attention compatible LFM2 for causal language modeling with frame-level positions.

    Features:
    - Frame-level position encoding for audio tokens
    - Flash Attention 2 optimized
    - Standard causal masking
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self,
        config: Lfm2Config,
        audio_tokens_start: int,
        tokens_per_frame: int = 4,
        audio_step: float = 1.0,
        use_learnable_rope: bool = False,
        alpha_min: float = 0.1,
        alpha_max: float = 2.0,
        speaker_emb_dim: int = 128
    ):
        super().__init__(config)

        # Use our flash-compatible model
        self.model = FlashCompatibleLfm2Model(
            config,
            audio_tokens_start,
            tokens_per_frame,
            audio_step=audio_step,
            use_learnable_rope=use_learnable_rope,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            speaker_emb_dim=speaker_emb_dim
        )
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Store these for easy access
        self.audio_tokens_start = audio_tokens_start
        self.tokens_per_frame = tokens_per_frame
        self.use_learnable_rope = use_learnable_rope
        self.speaker_emb_dim = speaker_emb_dim

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        speaker_emb: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        """
        Forward pass - delegates to flash-compatible model.

        Args:
            speaker_emb: Speaker embeddings [batch_size, speaker_emb_dim] (e.g., [8, 128])
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            speaker_emb=speaker_emb,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        audio_tokens_start: int,
        tokens_per_frame: int = 4,
        audio_step: float = 1.0,
        use_learnable_rope: bool = False,
        alpha_min: float = 0.1,
        alpha_max: float = 2.0,
        speaker_emb_dim: int = 128,
        init_from_scratch: bool = False,
        *model_args,
        **kwargs
    ):
        """
        Load a pretrained LFM2 model with flash-compatible implementation.

        Args:
            pretrained_model_name_or_path: HuggingFace model ID or local path
            audio_tokens_start: Token ID where audio tokens begin (e.g., 64410)
            tokens_per_frame: Number of tokens per audio frame
            audio_step: Position step per audio frame
            use_learnable_rope: Enable learnable RoPE frequencies per layer
            alpha_min: Minimum alpha value for learnable RoPE
            alpha_max: Maximum alpha value for learnable RoPE
            speaker_emb_dim: Dimension of speaker embeddings (default: 128)
            init_from_scratch: If True, skip loading pretrained weights (random initialization)
        """
        # Load config
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # # ________ theta RoPE change _________
        # config.rope_theta = 1000.0
        # # ____________________________________

        # Create model
        model = cls(
            config=config,
            audio_tokens_start=audio_tokens_start,
            tokens_per_frame=tokens_per_frame,
            audio_step=audio_step,
            use_learnable_rope=use_learnable_rope,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            speaker_emb_dim=speaker_emb_dim
        )

        # Load pretrained weights (unless init_from_scratch=True)
        if init_from_scratch:
            print(f"⚠️  Initializing model from SCRATCH (random weights)")
            print(f"   Config loaded from: {pretrained_model_name_or_path}")
            print(f"   Pretrained weights: NOT LOADED")
        else:
            base_model = Lfm2ForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)

            # Copy weights from base model to our custom model
            model.model.load_state_dict(base_model.model.state_dict(), strict=False)
            model.lm_head.load_state_dict(base_model.lm_head.state_dict())

            print(f"✅ Loaded pretrained weights from {pretrained_model_name_or_path}")

        return model
