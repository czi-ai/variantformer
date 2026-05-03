"This module contains the implementation of different transformer layers used in the model."

import torch
import math
import torch.nn as nn
from flash_attn.modules.mha import MHA
from flash_attn.bert_padding import (
    pad_input,
    unpad_input,
)
from typing import Union
from einops import rearrange


# From https://github.com/ofirpress/attention_with_linear_biases/blob/4b92f28a005ead2567abe2359f633e73e08f3833/fairseq/models/transformer.py#L742
def get_alibi_slopes(n):
    """Get the slopes for the alibi attention."""

    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return torch.tensor(
            get_slopes_power_of_2(n)
        )  # In the paper, we only train models that have 2^a heads for some a. This function has

    else:  # some good properties that only occur when the input is a power of 2. To maintain that even
        closest_power_of_2 = 2 ** math.floor(
            math.log2(n)
        )  # when the number of heads is not a power of 2, we use this workaround.
        return torch.tensor(
            get_slopes_power_of_2(closest_power_of_2)
            + get_alibi_slopes(2 * closest_power_of_2)[0::2][
                : n - closest_power_of_2
            ].tolist()
        )


def get_relative_positions(seq_len: int) -> torch.Tensor:
    """Get relative positions for the alibi attention."""
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    return x - y


class ContextFlashAttentionEncoderLayer(nn.Module):
    """Flash Attention Block without dtype conversions"""

    def __init__(
        self,
        d_model,
        nhead,
        hidden_dim=2048,
        dropout=0.1,
        batch_first=True,
        use_alibi=False,
        make_data_kv=False,
        mlp_dout=0.0,
        cross_alibi=False,
        flash_attn_3=False,
    ):
        super().__init__()
        self.mixer = FlashAttLayer(
            d_model, nhead, dropout=dropout, use_alibi=use_alibi, cross_attn=False
        )
        self.crossMHA = FlashAttLayer(
            d_model,
            nhead,
            dropout=dropout,
            use_alibi=cross_alibi,
            cross_attn=True,
            flash_attn_3=flash_attn_3,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.linear_geglu_1 = nn.Linear(d_model, hidden_dim)
        self.dropout = nn.Dropout(mlp_dout)
        self.linear_geglu_2 = nn.Linear(hidden_dim // 2, d_model)
        self.use_alibi = use_alibi
        self.num_heads = nhead
        self.make_data_kv = make_data_kv
        self.activation = nn.GELU()
        if use_alibi:
            self.register_buffer("m", get_alibi_slopes(self.num_heads))

    def forward(
        self,
        src,
        context,
        src_key_padding_mask=None,
        context_padding_mask=None,
        precision=torch.float32,
        unpad_info=None,
        context_unpad_info=None,
        gene_unpad_info=None,
    ):
        res_long = src
        res_short = src
        dtype = src.dtype
        flash_dtype = torch.float16 if precision == torch.float32 else precision

        if context_padding_mask is None and src_key_padding_mask is not None:
            context_padding_mask = src_key_padding_mask.clone()

        # Self-attention block
        x = self.norm1(src)
        qkv = x
        if flash_dtype is not None:
            self.mixer.to(flash_dtype)
            qkv = qkv.to(flash_dtype)

        # Use gene_unpad_info if provided, otherwise fall back to unpad_info
        src_unpad_info = gene_unpad_info if gene_unpad_info is not None else unpad_info

        x = self.mixer(
            qkv,
            src_key_padding_mask=src_key_padding_mask,
            precision=precision,
            unpad_info=src_unpad_info,
        )
        if flash_dtype is not None:
            self.mixer.to(src.dtype)
            x = x.to(src.dtype)

        x += res_short
        res_short = x

        # Cross-attention block
        x = self.norm2(x)

        if self.make_data_kv:
            q, kv = context, x
        else:
            q, kv = x, context
        if flash_dtype is not None:
            self.crossMHA.to(flash_dtype)
            q = q.to(flash_dtype)
            kv = kv.to(flash_dtype)

        x = self.crossMHA(
            q,
            kv,
            src_key_padding_mask=src_key_padding_mask,
            context_key_padding_mask=context_padding_mask,
            precision=precision,
            unpad_info=src_unpad_info,
            context_unpad_info=context_unpad_info,
        )
        if flash_dtype is not None:
            self.crossMHA.to(src.dtype)
            x = x.to(src.dtype)

        x += res_short

        # Feed-forward network with GeGLU
        x = self.norm3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * self.activation(gate)
        x = self.dropout(x)
        x = self.linear_geglu_2(x)
        x += res_long

        return x


class FlashAttentionEncoderLayer(nn.Module):
    "Flash Attention Block"

    def __init__(
        self,
        d_model,
        nhead,
        hidden_dim=2048,
        dropout=0.1,
        batch_first=True,
        use_alibi=False,
        make_data_kv=False,
        mlp_dout=0.0,
    ):
        super().__init__()
        self.mixer = FlashAttLayer(
            d_model, nhead, dropout=dropout, use_alibi=use_alibi, cross_attn=False
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.linear_geglu_1 = nn.Linear(d_model, hidden_dim)
        self.dropout = nn.Dropout(mlp_dout)
        self.linear_geglu_2 = nn.Linear(hidden_dim // 2, d_model)
        self.use_alibi = use_alibi
        self.num_heads = nhead
        self.make_data_kv = make_data_kv
        self.activation = nn.GELU()
        if use_alibi:
            self.register_buffer("m", get_alibi_slopes(self.num_heads))

    def forward(
        self, src, src_key_padding_mask=None, precision=torch.float32, unpad_info=None
    ):
        res_long = src
        res_short = src
        dtype = src.dtype
        flash_dtype = torch.float16 if precision == torch.float32 else precision
        x = self.norm1(src)
        qkv = x
        if flash_dtype is not None:
            self.mixer.to(flash_dtype)
            qkv = qkv.to(flash_dtype)

        x = self.mixer(
            qkv,
            src_key_padding_mask=src_key_padding_mask,
            precision=precision,
            unpad_info=unpad_info,
        )
        if flash_dtype is not None:
            self.mixer.to(src.dtype)
            x = x.to(src.dtype)
        x += res_short
        x = self.norm2(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * self.activation(gate)
        x = self.dropout(x)
        x = self.linear_geglu_2(x)
        x += res_long
        return x


class ContextFlashCrossAttentionEncoderLayer(nn.Module):
    "Flash Attention Block"

    def __init__(
        self,
        d_model,
        nhead,
        hidden_dim=2048,
        dropout=0.1,
        batch_first=True,
        use_alibi=False,
        make_data_kv=False,
        mlp_dout=0.0,
        cross_alibi=False,
        flash_attn_3=False,
    ):
        super().__init__()
        self.crossMHA = FlashAttLayer(
            d_model,
            nhead,
            dropout=dropout,
            use_alibi=cross_alibi,
            cross_attn=True,
            flash_attn_3=flash_attn_3,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear_geglu_1 = nn.Linear(d_model, hidden_dim)
        self.dropout = nn.Dropout(mlp_dout)
        self.linear_geglu_2 = nn.Linear(hidden_dim // 2, d_model)
        self.use_alibi = use_alibi
        self.num_heads = nhead
        self.make_data_kv = make_data_kv
        self.activation = nn.GELU()
        if use_alibi:
            self.register_buffer("m", get_alibi_slopes(self.num_heads))

    def forward(
        self,
        src,
        context,
        context_padding_mask=None,
        src_key_padding_mask=None,
        precision=torch.float32,
        gene_unpad_info=None,
        context_unpad_info=None,
    ):
        res_long = src
        res_short = src
        flash_dtype = torch.float16 if precision == torch.float32 else precision

        x = self.norm1(src)
        if self.make_data_kv:
            q, kv = context, x
        else:
            q, kv = x, context
        if flash_dtype is not None:
            self.crossMHA.to(flash_dtype)
        #  Define padding masks - only if we're not using pre-unpadded data
        if gene_unpad_info is None and src_key_padding_mask is None:
            src_key_padding_mask = torch.zeros(
                src.shape[:2], dtype=torch.bool, device=src.device
            )
        if context_unpad_info is None and context_padding_mask is None:
            context_padding_mask = torch.zeros(
                context.shape[:2], dtype=torch.bool, device=context.device
            )

        #  Perform cross attention
        if flash_dtype is not None:
            q = q.to(flash_dtype)
            kv = kv.to(flash_dtype)

        x = self.crossMHA(
            q,
            kv,
            src_key_padding_mask=src_key_padding_mask,
            context_key_padding_mask=context_padding_mask,
            precision=precision,
            unpad_info=gene_unpad_info,
            context_unpad_info=context_unpad_info,
        )
        if flash_dtype is not None:
            self.crossMHA.to(src.dtype)
            x = x.to(src.dtype)

        # FF layer
        x += res_short
        x = self.norm2(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * self.activation(gate)
        x = self.dropout(x)
        x = self.linear_geglu_2(x)
        x += res_long
        return x


class FlashAttLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        hidden_dim=2048,
        dropout=0.1,
        use_alibi=False,
        cross_attn=False,
        flash_attn_3=False,
    ):
        super().__init__()
        self.cross_attn = cross_attn
        if cross_attn and flash_attn_3:
            raise NotImplementedError("flash-attention-3 is not supported at this time")
        else:
            self.MHA = MHA(
                d_model,
                nhead,
                dropout=dropout,
                use_flash_attn=True,
                use_alibi=use_alibi,
                cross_attn=cross_attn,
            )
        self.use_alibi = use_alibi
        self.causal = False
        # When True, the layer recomputes the full attention matrix during
        # forward and stores it on `self.attn_matrix`. Off by default to keep
        # the FlashAttention fast path unaffected; the LogAttention callback
        # toggles it on for selected layers around the forward pass.
        self.log_attn_matrix = False
        self.attn_matrix = None

    def forward(
        self,
        src,
        cntx=None,
        src_key_padding_mask=None,
        context_key_padding_mask=None,
        precision=torch.float32,
        unpad_info=None,
        context_unpad_info=None,
    ):
        dtype = src.dtype
        flash_dtype = torch.float16 if precision == torch.float32 else precision
        x_short = src
        batch, seqlen = src.shape[:2]
        mixer_kwargs = {}
        if self.cross_attn:
            assert cntx is not None

            # Check if inputs are already unpadded
            if unpad_info is not None:
                # Inputs are already unpadded, use them directly
                src_hidden_states = src
                mixer_kwargs = {
                    "cu_seqlens": unpad_info["cu_seqlens"],
                    "max_seqlen": unpad_info["max_seqlen"],
                }
            elif src_key_padding_mask is not None:
                # Need to unpad the inputs
                assert (
                    context_key_padding_mask is not None
                ), "context_key_padding_mask must be provided if src_key_padding_mask is provided"
                src_key_padding_mask = ~src_key_padding_mask
                (
                    src_hidden_states,
                    src_indices,
                    src_cu_seqlens,
                    src_max_seqlen_in_batch,
                    _,
                ) = unpad_input(src, src_key_padding_mask)
                mixer_kwargs = {
                    "cu_seqlens": src_cu_seqlens,
                    "max_seqlen": src_max_seqlen_in_batch,
                }
            else:
                src_hidden_states = src

            # Handle context unpadding
            if context_unpad_info is not None:
                # Context is already unpadded, use it directly
                cntx_hidden_states = cntx
                mixer_kwargs.update(
                    {
                        "cu_seqlens_k": context_unpad_info["cu_seqlens"],
                        "max_seqlen_k": context_unpad_info["max_seqlen"],
                    }
                )
            elif context_key_padding_mask is not None:
                # Need to unpad the context
                assert (
                    src_key_padding_mask is not None
                ), "src_key_padding_mask must be provided if context_key_padding_mask is provided"
                context_key_padding_mask = ~context_key_padding_mask
                (
                    cntx_hidden_states,
                    cntx_indices,
                    cntx_cu_seqlens,
                    cntx_max_seqlen_in_batch,
                    _,
                ) = unpad_input(cntx, context_key_padding_mask)
                mixer_kwargs.update(
                    {
                        "cu_seqlens_k": cntx_cu_seqlens,
                        "max_seqlen_k": cntx_max_seqlen_in_batch,
                    }
                )
            else:
                cntx_hidden_states = cntx

            if flash_dtype is not None:
                if src_hidden_states is not None:
                    src_hidden_states = src_hidden_states.to(flash_dtype)
                if cntx_hidden_states is not None:
                    cntx_hidden_states = cntx_hidden_states.to(flash_dtype)

            hidden_states = self.MHA(
                src_hidden_states, cntx_hidden_states, **mixer_kwargs
            )

            # Only pad if we're not already working with unpadded data
            if unpad_info is not None:
                # Data was already unpadded, return unpadded result
                x = hidden_states
            elif src_key_padding_mask is not None:
                # We unpadded the data, so we need to pad it back
                hidden_states = pad_input(hidden_states, src_indices, batch, seqlen)
                x = hidden_states
            else:
                x = hidden_states

        else:
            # Check if input is already unpadded
            if unpad_info is not None:
                # Input is already unpadded, use it directly
                hidden_states = src
                mixer_kwargs = {
                    "cu_seqlens": unpad_info["cu_seqlens"],
                    "max_seqlen": unpad_info["max_seqlen"],
                }
                if flash_dtype is not None:
                    if hidden_states is not None:
                        hidden_states = hidden_states.to(flash_dtype)

                hidden_states = self.MHA(hidden_states, **mixer_kwargs)
                # Return unpadded result since input was unpadded
                x = hidden_states
            elif src_key_padding_mask is not None:
                # Need to unpad the input
                src_key_padding_mask = ~src_key_padding_mask
                hidden_states, indices, cu_seqlens, max_seqlen_in_batch, _ = (
                    unpad_input(src, src_key_padding_mask)
                )
                mixer_kwargs = {
                    "cu_seqlens": cu_seqlens,
                    "max_seqlen": max_seqlen_in_batch,
                }
                if flash_dtype is not None:
                    if hidden_states is not None:
                        hidden_states = hidden_states.to(flash_dtype)

                hidden_states = self.MHA(hidden_states, **mixer_kwargs)
                hidden_states = pad_input(hidden_states, indices, batch, seqlen)
                x = hidden_states

            else:
                x = self.MHA(src)

        if self.log_attn_matrix:
            self.attn_matrix = self._compute_logged_attn_matrix(
                src=src,
                cntx=cntx,
                unpad_info=unpad_info,
                context_unpad_info=context_unpad_info,
            )
        return x

    def _compute_logged_attn_matrix(
        self,
        src: torch.Tensor,
        cntx: torch.Tensor | None,
        unpad_info: dict | None,
        context_unpad_info: dict | None,
    ) -> torch.Tensor:
        """Re-pad the unpadded inputs and compute the full attention matrix.

        This is only invoked when ``log_attn_matrix`` is True. It mirrors how
        the model is actually run in this codebase: the modulators always pass
        unpadded tensors through with companion ``unpad_info``/``context_unpad_info``
        dicts.
        """
        assert (
            unpad_info is not None
        ), "log_attn_matrix=True requires unpad_info; logging is only supported on the unpadded code path"
        if self.cross_attn:
            assert (
                context_unpad_info is not None
            ), "log_attn_matrix=True with cross_attn=True requires context_unpad_info"

        max_seqlen_q = unpad_info["max_seqlen"]
        max_seqlen_k = (
            context_unpad_info["max_seqlen"]
            if context_unpad_info is not None
            else unpad_info["max_seqlen"]
        )

        # Reconstruct the padded src and its boolean key-padding mask
        src_mask = torch.ones(src.shape[0], src.shape[1], device=src.device)
        original_src_key_padding_mask = ~pad_input(
            src_mask,
            unpad_info["indices"],
            unpad_info["batch"],
            unpad_info["seqlen"],
        ).bool()[:, :, 0]
        src_padded = pad_input(
            src, unpad_info["indices"], unpad_info["batch"], unpad_info["seqlen"]
        )

        if self.cross_attn:
            context_mask = torch.ones(
                cntx.shape[0], cntx.shape[1], device=cntx.device
            )
            original_context_key_padding_mask = ~pad_input(
                context_mask,
                context_unpad_info["indices"],
                context_unpad_info["batch"],
                context_unpad_info["seqlen"],
            ).bool()[:, :, 0]
            cntx_padded = pad_input(
                cntx,
                context_unpad_info["indices"],
                context_unpad_info["batch"],
                context_unpad_info["seqlen"],
            )
        else:
            cntx_padded = None
            original_context_key_padding_mask = None

        return self.calculate_attention_matrix(
            src_padded,
            cntx_padded,
            original_src_key_padding_mask,
            original_context_key_padding_mask,
            max_seqlen_q,
            max_seqlen_k,
        )

    def calculate_attention_matrix(
        self,
        src: torch.Tensor,
        cntx: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        context_key_padding_mask: torch.Tensor | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        cleanup: bool = True,
    ) -> torch.Tensor:
        """Compute the full attention matrix (with ALiBi) for self/cross attention.

        Args:
            src: Source sequence, shape ``[B, S_q, D]``.
            cntx: Context sequence for cross attention, shape ``[B, S_k, D]``.
                For self-attention this is ignored and ``src`` is used.
            src_key_padding_mask: Boolean padding mask for the query, shape ``[B, S_q]``
                (True = padded position).
            context_key_padding_mask: Boolean padding mask for the key/context,
                shape ``[B, S_k]`` (True = padded position).
            max_seqlen_q / max_seqlen_k: maximum unpadded query/key sequence
                lengths in the batch. Used to align ALiBi distances when
                Q and K have different lengths.
            cleanup: whether to free temporary tensors and call
                ``torch.cuda.empty_cache()`` after the computation.

        Returns:
            Attention probabilities, shape ``[B, H, S_q, S_k]``. For self-attention
            ``S_q == S_k``.
        """
        if self.cross_attn:
            assert cntx is not None, "Context tensor must be provided for cross attention"
        else:
            cntx = src

        delta = (
            max_seqlen_k - max_seqlen_q
            if max_seqlen_q is not None and max_seqlen_k is not None
            else 0
        )

        if self.cross_attn:
            Q = self.MHA.Wq(src)
            KV = self.MHA.Wkv(cntx)
            Q = rearrange(Q, "b s (h d) -> b s h d", d=self.MHA.head_dim)
            KV = rearrange(
                KV, "b s (two h d) -> two b s h d", two=2, d=self.MHA.head_dim
            )
            K, V = KV
        else:
            QKV = self.MHA.Wqkv(src)
            QKV = rearrange(
                QKV, "b s (three h d) -> three b s h d", three=3, d=self.MHA.head_dim
            )
            Q, K, V = QKV

        B, S_q, H, D = Q.shape
        S_k = K.shape[1]

        Q = rearrange(Q, "b s h d -> (b h) s d")
        K = rearrange(K, "b s h d -> (b h) s d")
        V = rearrange(V, "b s h d -> (b h) s d")

        softmax_scale = 1.0 / math.sqrt(D)
        logits = torch.einsum("btd,bsd->bts", Q, K * softmax_scale)

        if self.use_alibi:
            alibi_slopes = (
                get_alibi_slopes(H).to(src.device).repeat(B).to(src.dtype)
            )
            alibi_slopes = alibi_slopes.unsqueeze(1).unsqueeze(2)

            q_idx = torch.arange(S_q, dtype=src.dtype, device=src.device).unsqueeze(1)
            k_idx = torch.arange(S_k, dtype=src.dtype, device=src.device).unsqueeze(0)
            distance = q_idx + delta - k_idx

            bias = alibi_slopes * torch.abs(distance)
            logits = logits - bias
            del distance, bias, q_idx, k_idx, alibi_slopes

        if self.cross_attn:
            key_padding_mask = context_key_padding_mask
        else:
            key_padding_mask = src_key_padding_mask

        if key_padding_mask is not None:
            logits = rearrange(logits, "(b h) sq sk -> b h sq sk", b=B, h=H)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (b, 1, 1, sk)
            logits.masked_fill_(mask, float("-inf"))
            logits = rearrange(logits, "b h sq sk -> (b h) sq sk")

        if self.causal:
            causal_mask = torch.triu(
                torch.ones(S_q, S_k, dtype=logits.dtype, device=logits.device),
                diagonal=1,
            )
            logits = logits.masked_fill(causal_mask.bool(), float("-inf"))
            del causal_mask

        attn = torch.softmax(logits, dim=-1)
        attn = attn.nan_to_num(0)
        attn = rearrange(attn, "(b h) sq sk -> b h sq sk", b=B, h=H, sq=S_q, sk=S_k)

        if src_key_padding_mask is not None:
            attn_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(3)  # (B, 1, S_q, 1)
            attn.masked_fill_(attn_mask, 0.0)

        if cleanup:
            del logits
            del Q, K, V
            if self.cross_attn:
                del KV
            else:
                del QKV
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        attn = attn.detach()
        return attn


class StartToken(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.start_token = nn.Parameter(torch.randn(1, 1, emb_dim))

    def forward(self, x):
        out = torch.ones(x.size(0), 1, x.size(2)).to(x.device)
        out = out * self.start_token
        return out


class MultiRegistry(nn.Module):
    def __init__(self, num_tissues, emb_dim):
        super().__init__()
        self.num_registry_tokens = num_tissues
        self.registry_tokens = nn.Embedding(num_tissues, emb_dim)

    def forward(self, x, tissue_vector):
        registry_embeddings = [
            self.registry_tokens(tissue_vector[i][0]).unsqueeze(0).unsqueeze(1)
            for i in range(x.size(0))
        ]  # (x.size(0), 1, emb_dim)
        registry_embeddings = torch.cat(
            registry_embeddings, dim=0
        )  # (x.size(0), 1, emb_dim)
        # residual = torch.cat((torch.zeros_like(registry_embeddings), x), dim=1) # (x.size(0), 1 + x.size(1), emb_dim)
        combined_x = torch.cat(
            (registry_embeddings, x), dim=1
        )  # (x.size(0), 1 + x.size(1), emb_dim)
        residual = combined_x.clone()
        return combined_x, residual

    def get_registry_tokens(self):
        return self.registry_tokens.weight


class ConcatTissueContext(nn.Module):
    def __init__(self, num_tissues, emb_dim):
        super().__init__()
        self.num_registry_tokens = num_tissues
        self.registry_tokens = nn.Embedding(num_tissues, emb_dim)

    def forward(self, x, tissue_vector, padding_mask):
        registry_embeddings = [
            self.registry_tokens(tissue_vector[i][0]).unsqueeze(0).unsqueeze(1)
            for i in range(x.size(0))
        ]  # (x.size(0), 1, emb_dim)
        registry_embeddings = torch.cat(
            registry_embeddings, dim=0
        )  # (x.size(0), 1, emb_dim)
        combined_x = torch.cat(
            (registry_embeddings, x), dim=1
        )  # (x.size(0), 1 + x.size(1), emb_dim)

        start_mask = torch.zeros(
            (padding_mask.size(0), 1),
            dtype=padding_mask.dtype,
            device=padding_mask.device,
        )
        new_mask = torch.cat((start_mask, padding_mask), dim=1)
        padding_mask = new_mask.clone()
        return combined_x, padding_mask

    def get_registry_tokens(self):
        return self.registry_tokens.weight


class AddContext(nn.Module):
    def __init__(self, num_tissues, emb_dim):
        super().__init__()
        self.num_registry_tokens = num_tissues
        self.registry_tokens = nn.Embedding(num_tissues, emb_dim)

    def forward(self, x, tissue_vector):
        registry_embeddings = [
            self.registry_tokens(tissue_vector[i][0]).unsqueeze(0).unsqueeze(1)
            for i in range(x.size(0))
        ]  # (x.size(0), 1, emb_dim)
        registry_embeddings = torch.cat(
            registry_embeddings, dim=0
        )  # (x.size(0), 1, emb_dim)
        combined_x = x + registry_embeddings  # (x.size(0), x.size(1), emb_dim)
        return combined_x

    def get_registry_tokens(self):
        return self.registry_tokens.weight


class EpigeneticsModulator(nn.Module):
    def __init__(
        self,
        emb_dim,
        num_heads,
        num_layers,
        use_alibi,
        mlp_dout,
        use_context,
        num_ref_cres=None,
        flash_attn_3=False,
    ):
        super().__init__()
        self.use_context = use_context

        if use_context:
            assert (
                num_ref_cres is not None
            ), "num_ref_cres must be provided when use_context is True"
            self.second_level_context_embedding = nn.Embedding(num_ref_cres, emb_dim)
            self.epigenetics_modulator = nn.ModuleList(
                [
                    ContextFlashAttentionEncoderLayer(
                        d_model=emb_dim,
                        nhead=num_heads,
                        batch_first=True,
                        use_alibi=use_alibi,
                        mlp_dout=mlp_dout,
                        flash_attn_3=flash_attn_3,
                    )
                    for _ in range(num_layers - 1)
                ]
            )
        else:
            self.epigenetics_modulator = nn.ModuleList(
                [
                    FlashAttentionEncoderLayer(
                        d_model=emb_dim,
                        nhead=num_heads,
                        batch_first=True,
                        use_alibi=use_alibi,
                        mlp_dout=mlp_dout,
                    )
                    for _ in range(num_layers - 1)
                ]
            )

    def forward(
        self,
        x,
        context=None,
        src_key_padding_mask=None,
        precision=None,
        context_padding_mask=None,
        keep_intermediates_unpadded=False,
    ):
        # Create a list to store all intermediate outputs
        intermediate_outputs = [x.clone()]  # Start with the input
        context_embedding = (
            self.second_level_context_embedding(context) if self.use_context else None
        )

        # Handle unpadding once at the start
        unpad_info = None
        context_unpad_info = None

        if src_key_padding_mask is not None:
            # Unpad the main input once
            batch, seqlen = x.shape[:2]
            src_key_padding_mask_inverted = ~src_key_padding_mask
            x_unpadded, indices, cu_seqlens, max_seqlen_in_batch, _ = unpad_input(
                x, src_key_padding_mask_inverted
            )
            unpad_info = {
                "indices": indices,
                "cu_seqlens": cu_seqlens,
                "max_seqlen": max_seqlen_in_batch,
                "batch": batch,
                "seqlen": seqlen,
            }
            x = x_unpadded

        if (
            self.use_context
            and context_padding_mask is not None
            and context_embedding is not None
        ):
            # Unpad the context once
            context_batch, context_seqlen = context_embedding.shape[:2]
            context_padding_mask_inverted = ~context_padding_mask
            (
                context_unpadded,
                context_indices,
                context_cu_seqlens,
                context_max_seqlen_in_batch,
                _,
            ) = unpad_input(context_embedding, context_padding_mask_inverted)
            context_unpad_info = {
                "indices": context_indices,
                "cu_seqlens": context_cu_seqlens,
                "max_seqlen": context_max_seqlen_in_batch,
                "batch": context_batch,
                "seqlen": context_seqlen,
            }
            context_embedding = context_unpadded

        # Store intermediate outputs in the appropriate format
        unpadded_intermediates = [x.clone()] if keep_intermediates_unpadded else []

        for layer in self.epigenetics_modulator:
            if self.use_context:
                x = layer(
                    x,
                    context_embedding,
                    unpad_info=unpad_info,
                    context_unpad_info=context_unpad_info,
                    precision=precision,
                )
            else:
                x = layer(x, unpad_info=unpad_info, precision=precision)

            # Store intermediate output based on requested format
            if x is not None:
                if keep_intermediates_unpadded:
                    # Store unpadded intermediate
                    unpadded_intermediates.append(x.clone())
                else:
                    # Store padded intermediate (backward compatibility)
                    if unpad_info is not None:
                        x_padded = pad_input(
                            x,
                            unpad_info["indices"],
                            unpad_info["batch"],
                            unpad_info["seqlen"],
                        )
                        if x_padded is not None:
                            intermediate_outputs.append(x_padded.clone())
                    else:
                        intermediate_outputs.append(x.clone())

        # Handle final output
        if x is not None and unpad_info is not None and not keep_intermediates_unpadded:
            x = pad_input(
                x, unpad_info["indices"], unpad_info["batch"], unpad_info["seqlen"]
            )
            # Update the last intermediate output with the final padded result
            if len(intermediate_outputs) > 0 and x is not None:
                intermediate_outputs[-1] = x.clone()

        # Return structured output when keeping intermediates unpadded
        if keep_intermediates_unpadded:
            return {
                "intermediate_outputs": unpadded_intermediates,
                "unpad_info": unpad_info,
                "context_unpad_info": context_unpad_info,
                "final_output": pad_input(
                    x, unpad_info["indices"], unpad_info["batch"], unpad_info["seqlen"]
                )
                if unpad_info is not None and x is not None
                else x,
            }
        else:
            # Backward compatibility: return padded intermediate outputs
            return intermediate_outputs


class GeneModulator(nn.Module):
    """Module for gene modulation using epigenetics context."""

    def __init__(
        self,
        emb_dim,
        num_heads,
        num_layers,
        use_alibi,
        mlp_dout,
        only_cross_attention=True,
        use_res=False,
        cross_alibi=False,
        flash_attn_3=False,
    ):
        super().__init__()
        self.use_res = use_res
        self.only_cross_attention = only_cross_attention
        self.cross_alibi = cross_alibi

        if only_cross_attention:
            self.gene_modulator = nn.ModuleList(
                [
                    ContextFlashCrossAttentionEncoderLayer(
                        d_model=emb_dim,
                        nhead=num_heads,
                        batch_first=True,
                        use_alibi=use_alibi,
                        mlp_dout=mlp_dout,
                        cross_alibi=cross_alibi,
                        flash_attn_3=flash_attn_3,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.gene_modulator = nn.ModuleList(
                [
                    ContextFlashAttentionEncoderLayer(
                        d_model=emb_dim,
                        nhead=num_heads,
                        batch_first=True,
                        use_alibi=use_alibi,
                        mlp_dout=mlp_dout,
                        cross_alibi=cross_alibi,
                        flash_attn_3=flash_attn_3,
                    )
                    for _ in range(num_layers)
                ]
            )

    def forward(
        self,
        g_exp,
        modulator_outputs,
        res=None,
        padding_mask=None,
        src_key_padding_mask=None,
        precision=None,
    ):
        """
        Forward pass through the gene modulator

        Args:
            g_exp: Gene expression embeddings [batch, seq_len, emb_dim]
            modulator_outputs: List of outputs from epigenetics modulator OR dict with unpadded intermediates
            res: Residual connection reference (if None, will use g_exp.clone())
            padding_mask: Padding mask for the context (epigenetics)
            src_key_padding_mask: Padding mask for the source (gene)
            precision: Precision for computation

        Returns:
            Modulated gene expression embeddings
        """
        # Check if modulator_outputs is the new structured format or old list format
        if isinstance(modulator_outputs, dict):
            # New optimized format: use unpadded intermediates directly
            unpadded_intermediates = modulator_outputs["intermediate_outputs"]
            context_unpad_info = modulator_outputs[
                "unpad_info"
            ]  # This is for the context/epigenetics data
            gene_unpad_info = None  # Will be computed for gene data if needed

            # Use the unpadded intermediates directly
            modulator_intermediates = unpadded_intermediates
        else:
            # Old format: list of padded intermediates - need to unpad them
            modulator_intermediates = modulator_outputs
            context_unpad_info = None
            gene_unpad_info = None

        # Handle gene input unpadding once at the start
        if src_key_padding_mask is not None:
            # Unpad the gene input once
            batch, seqlen = g_exp.shape[:2]
            src_key_padding_mask_inverted = ~src_key_padding_mask
            g_exp_unpadded, indices, cu_seqlens, max_seqlen_in_batch, _ = unpad_input(
                g_exp, src_key_padding_mask_inverted
            )
            gene_unpad_info = {
                "indices": indices,
                "cu_seqlens": cu_seqlens,
                "max_seqlen": max_seqlen_in_batch,
                "batch": batch,
                "seqlen": seqlen,
            }
            g_exp = g_exp_unpadded

        # Use provided res or create one from the (potentially unpadded) g_exp
        if res is None and self.use_res and g_exp is not None:
            res = g_exp.clone()  # This will be unpadded if g_exp was unpadded

        # Handle context unpadding if needed (old format)
        if not isinstance(modulator_outputs, dict) and padding_mask is not None:
            # Old format: need to unpad all modulator outputs
            context_batch, context_seqlen = modulator_intermediates[0].shape[:2]
            context_padding_mask_inverted = ~padding_mask
            context_unpad_info = {
                "cu_seqlens": None,
                "max_seqlen": None,
                "batch": context_batch,
                "seqlen": context_seqlen,
            }
            # We need to unpad all modulator outputs
            unpadded_modulator_outputs = []
            for mod_output in modulator_intermediates:
                (
                    mod_unpadded,
                    mod_indices,
                    mod_cu_seqlens,
                    mod_max_seqlen_in_batch,
                    _,
                ) = unpad_input(mod_output, context_padding_mask_inverted)
                unpadded_modulator_outputs.append(mod_unpadded)
            # Store the unpadding info from the first output (they should all be the same)
            context_unpad_info.update(
                {
                    "indices": mod_indices,
                    "cu_seqlens": mod_cu_seqlens,
                    "max_seqlen": mod_max_seqlen_in_batch,
                }
            )
            modulator_intermediates = unpadded_modulator_outputs

        # Process through each layer with corresponding context
        for i, layer in enumerate(self.gene_modulator):
            # Get appropriate context from modulator outputs (use last one if we run out)
            context_idx = min(i, len(modulator_intermediates) - 1)
            context = modulator_intermediates[context_idx]

            # Apply the layer
            g_exp = layer(
                g_exp,
                context,
                gene_unpad_info=gene_unpad_info,
                context_unpad_info=context_unpad_info,
                precision=precision,
            )

            # Apply residual connection if configured
            if self.use_res:
                if res is None:
                    # This should not happen, but just in case
                    raise ValueError(
                        "Residual connection is enabled but no residual reference provided"
                    )
                # Residual is already in the same format as g_exp (unpadded if needed)
                g_exp = g_exp + res

        # Pad the final output once at the end
        if gene_unpad_info is not None:
            g_exp = pad_input(
                g_exp,
                gene_unpad_info["indices"],
                gene_unpad_info["batch"],
                gene_unpad_info["seqlen"],
            )

        return g_exp

    def prepare_input(
        self,
        g_exp,
        gene_pooling,
        start_tkn=None,
        tissue_vector=None,
        padding_mask_gene=None,
    ):
        """
        Prepare the gene expression input by adding start tokens or registry tokens if needed

        Args:
            g_exp: Gene expression embeddings
            gene_pooling: Type of pooling to use ('start_token', 'multi_registry', etc.)
            start_tkn: The start token module to use
            tissue_vector: Tissue vector for multi_registry
            padding_mask_gene: Padding mask for gene expression

        Returns:
            g_exp: Updated gene expression with added tokens if needed
            res: Residual connection reference
            padding_mask_gene: Updated padding mask
        """
        res = g_exp.clone()

        if gene_pooling == "start_token" and start_tkn is not None:
            start_token = start_tkn(g_exp)
            g_exp = torch.cat((start_token, g_exp), dim=1)
            res = torch.cat((torch.zeros_like(start_token), res), dim=1)
            if padding_mask_gene is not None:
                start_mask = torch.zeros(
                    (padding_mask_gene.size(0), 1),
                    dtype=padding_mask_gene.dtype,
                    device=padding_mask_gene.device,
                )
                new_mask = torch.cat((start_mask, padding_mask_gene), dim=1)
                padding_mask_gene = new_mask.clone()

        elif gene_pooling == "multi_registry" and start_tkn is not None:
            g_exp, res = start_tkn(g_exp, tissue_vector)
            if padding_mask_gene is not None:
                start_mask = torch.zeros(
                    (padding_mask_gene.size(0), 1),
                    dtype=padding_mask_gene.dtype,
                    device=padding_mask_gene.device,
                )
                new_mask = torch.cat((start_mask, padding_mask_gene), dim=1)
                padding_mask_gene = new_mask.clone()

        return g_exp, res, padding_mask_gene

    def pool_outputs(self, g_exp, gene_pooling, padding_mask_gene=None):
        """
        Pool the gene expression outputs based on the specified method

        Args:
            g_exp: Gene expression embeddings
            gene_pooling: Type of pooling to use
            padding_mask_gene: Padding mask for gene expression

        Returns:
            Pooled gene expression
        """
        if gene_pooling == "mean" and padding_mask_gene is not None:
            comp_padding_mask_gene = ~padding_mask_gene
            g_exp = (
                g_exp * comp_padding_mask_gene.unsqueeze(-1)
            ) / comp_padding_mask_gene.sum(dim=1)

        elif gene_pooling == "max" and padding_mask_gene is not None:
            inf_padding_mask = torch.zeros(padding_mask_gene.size()).to(
                padding_mask_gene.device
            )
            inf_padding_mask = inf_padding_mask.masked_fill(
                padding_mask_gene, float("-inf")
            )
            g_exp = g_exp + inf_padding_mask.unsqueeze(-1)
            g_exp = g_exp.max(dim=1).values

        elif gene_pooling in ["start_token", "multi_registry"]:
            g_exp = g_exp[:, 0, :]  # Only keep the start/registry token
        else:
            raise ValueError(f"Invalid gene pooling method: {gene_pooling}")

        return g_exp


class TissueExpressionHeads(nn.Module):
    """Module for tissue-specific gene expression prediction."""

    def __init__(
        self,
        emb_dim,
        num_tissues,
        use_bigger_head=False,
        multi_head=True,
        mlp_dout=0.1,
        loss_fn="poisson",
        head_type="mlp",
    ):
        """
        Initialize the tissue expression heads

        Args:
            emb_dim: Embedding dimension
            num_tissues: Number of tissues
            use_bigger_head: Whether to use bigger (more complex) heads
            multi_head: Whether to use a separate head for each tissue
            mlp_dout: Dropout rate for MLPs
            loss_fn: Loss function (affects output activation)
        """
        super().__init__()
        self.multi_head = multi_head
        self.tissue_expressions: Union[nn.ModuleDict, nn.Sequential]

        if head_type == "linear":
            if multi_head:
                self.tissue_expressions = nn.ModuleDict(
                    {
                        str(tissue_id): nn.Sequential(
                            nn.Linear(emb_dim, 1),
                            nn.Softplus() if loss_fn == "poisson" else nn.Identity(),
                        )
                        for tissue_id in range(num_tissues)
                    }
                )
            else:
                self.tissue_expressions = nn.Sequential(
                    nn.Linear(emb_dim, 1),
                    nn.Softplus() if loss_fn == "poisson" else nn.Identity(),
                )
        elif head_type == "mlp":
            # Configure tissue expression heads based on settings
            if use_bigger_head:
                if multi_head:
                    self.tissue_expressions = nn.ModuleDict(
                        {
                            str(tissue_id): nn.Sequential(
                                nn.Linear(emb_dim, emb_dim),
                                nn.LayerNorm(emb_dim),
                                nn.GELU(),
                                nn.Dropout(mlp_dout),
                                nn.Linear(emb_dim, emb_dim),
                                nn.GELU(),
                                nn.Linear(emb_dim, 1),
                                nn.Softplus()
                                if loss_fn == "poisson"
                                else nn.Identity(),
                            )
                            for tissue_id in range(num_tissues)
                        }
                    )
                else:
                    self.tissue_expressions = nn.Sequential(
                        nn.Linear(emb_dim, emb_dim),
                        nn.LayerNorm(emb_dim),
                        nn.GELU(),
                        nn.Dropout(mlp_dout),
                        nn.Linear(emb_dim, emb_dim),
                        nn.GELU(),
                        nn.Linear(emb_dim, 1),
                        nn.Softplus() if loss_fn == "poisson" else nn.Identity(),
                    )
            else:
                if multi_head:
                    self.tissue_expressions = nn.ModuleDict(
                        {
                            str(tissue_id): nn.Sequential(
                                nn.Linear(emb_dim, emb_dim),
                                nn.GELU(),
                                nn.Linear(emb_dim, 1),
                                nn.Softplus()
                                if loss_fn == "poisson"
                                else nn.Identity(),
                            )
                            for tissue_id in range(num_tissues)
                        }
                    )
                else:
                    self.tissue_expressions = nn.Sequential(
                        nn.Linear(emb_dim, emb_dim),
                        nn.GELU(),
                        nn.Linear(emb_dim, 1),
                        nn.Softplus() if loss_fn == "poisson" else nn.Identity(),
                    )
        else:
            raise ValueError(f"Invalid head type: {head_type}")

    def forward(self, g_exp, tissue_vector):
        """
        Predict gene expression for each sample based on its tissue

        Args:
            g_exp: Gene expression embeddings [batch_size, emb_dim]
            tissue_vector: Tissue ids for each sample [batch_size, 1]

        Returns:
            Gene expression predictions [batch_size, 1]
        """
        pred_gene_exp = torch.zeros(g_exp.size(0), 1, device=g_exp.device)

        # Process each sample with the appropriate tissue head
        for g in range(g_exp.size(0)):
            # Verify tissue vector has only one unique value
            assert len(torch.unique(tissue_vector[g])) == 1, "Tissue vector not unique"
            tissue_id = tissue_vector[g][0]
            x_tissue = g_exp[g]

            # Get the appropriate classifier based on multi_head setting
            if self.multi_head:
                tissue_key = str(tissue_id.item())
                # Use getattr to access ModuleDict items safely
                classifier = getattr(self.tissue_expressions, tissue_key)
            else:
                classifier = self.tissue_expressions

            # Apply the classifier
            pred_gene_exp[g] = classifier(x_tissue)

        return pred_gene_exp
