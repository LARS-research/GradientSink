# coding=utf-8
#
# A minimal, HF-compatible Llama modeling file that adds **V-scale**
#   \tilde V = a(||V||) * V,  a(r) = r^2 / (r^2 + c)
# to control Value-path gradient fan-in without touching softmax / attention logits.
#
# Design goals:
# - Reuse as much as possible from `transformers.models.llama.modeling_llama`.
# - Keep FlashAttention/SDPA backends unchanged.
# - Make the scaling optionally enabled via config attrs (getattr-safe).
# - Support per-layer, per-(KV)-head learnable `c` with dimension-consistent init.

from __future__ import annotations

from typing import Callable, Optional, Union

import torch
from torch import nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple

from transformers.models.llama.configuration_llama import LlamaConfig

# Reuse core Llama components
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


class VRadialScale(nn.Module):
    """Per-(KV)-head V radial scaling: v <- v * (||v||^2 / (||v||^2 + c)).

    Forward is smooth; backward control comes from the Jacobian of this nonlinearity.
    By default, `c` is initialized from HF Llama linear init (std=initializer_range)
    under the common PreNorm assumption Var(x_dim) ~ 1.

    Expected init scale (per KV head):
        E||v||^2 ≈ d_model * d_h * sigma_w^2
    so we set:
        c0 = d_model * d_h * sigma_w^2

    You can make c learnable per head (recommended) or scalar.
    """
    def __init__(
            self,
            config,
            sigma_w: float = 0.02,
            learnable: bool = True,
            mode: str = "full"  # "off", "full", "backward_only"
        ):
        super().__init__()
        self.num_kv_heads = config.num_key_value_heads

        # diagnostic mode
        if mode not in ("off", "full", "backward_only"):
            raise ValueError(f"v_scale_mode={mode!r} not in ('off', 'full', 'backward_only')")
        self.mode = mode

        # Dimension-consistent initialization: c0 is per-head (vector) or scalar.
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        # priori expectations: c0 = d_model * d_head * sigma_w^2
        self.c0_scalar = float(config.hidden_size * self.head_dim * (sigma_w ** 2))

        if learnable:
            self.theta = nn.Parameter(torch.zeros(self.num_kv_heads, dtype=torch.float32))
        else:
            self.register_parameter("theta", None)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        """protect FP32"""
        
        if self.theta is not None:
            theta_key = prefix + 'theta'
            if theta_key in state_dict:
                self.theta.data = self.theta.data.float()

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """v: [B, H_kv, T, D]."""
        if self.mode == "off":
            return v
        
        r2 = (v * v).sum(dim=-1, keepdim=True, dtype=torch.float32)  # [B,H,T,1]

        # protect exp theta
        with torch.autocast(device_type=v.device.type, enabled=False):
            if self.theta is None:
                c = r2.new_full((1, self.num_kv_heads, 1, 1), self.c0_scalar)
            else:
                c = (self.c0_scalar * torch.exp(self.theta.float())).view(1, -1, 1, 1)

            a = r2 / (r2 + c)  # fp32
        
        v_scaled = v * a.to(v.dtype)

        if self.mode == "full":
            return v_scaled

        if self.mode == "backward_only":
            return v.detach() + (v_scaled - v_scaled.detach())

        raise RuntimeError(f"Unhandled v_scale_mode={self.mode!r}")


class VscaleAttention(nn.Module):
    """HF LlamaAttention + optional V radial scaling (per KV head)."""
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        # --- V radial scaling knobs (all getattr-safe) ---
        self.mode = str(getattr(config, "v_scale_mode", "full")).lower()
        self.v_radial = VRadialScale(
            config=config,
            sigma_w=float(getattr(config, "initializer_range", 0.02)),
            learnable=bool(getattr(config, "v_scale_learnable", False)),
            mode=self.mode,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply per-KV-head radial scaling BEFORE cache/repeat_kv.
        value_states = self.v_radial(value_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class VscaleDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = VscaleAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class VscalePreTrainedModel(PreTrainedModel):
    config: LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["VscaleDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]

    _keep_in_fp32_modules = None # ["v_radial"]
    
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": VscaleDecoderLayer,
        "attentions": VscaleAttention,
    }

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        # only when theta is learnable
        if bool(getattr(config, "v_scale_learnable", False)):
            self._keep_in_fp32_modules = ["v_radial"]
        else:
            self._keep_in_fp32_modules = None


class VscaleModel(VscalePreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([VscaleDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    False,
                    output_attentions,
                    cache_position,
                    position_embeddings,
                )
            else:
                outputs = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            hidden_states = outputs[0]

            if output_attentions:
                all_self_attns += (outputs[1],)

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class VscaleForCausalLM(VscalePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"]) }

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = VscaleModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
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


__all__ = [
    "VRadialScale",
    "VscaleAttention",
    "VscaleDecoderLayer",
    "VscaleModel",
    "VscaleForCausalLM",
]
