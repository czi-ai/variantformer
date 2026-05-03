"""Unit tests for the attention-logging callback and layer hooks.

These tests run end-to-end on a tiny model with random weights so they do not
depend on the downloaded ``_artifacts`` (model checkpoints, reference genome,
or VCFs). They do require a CUDA GPU because FlashAttention only runs on GPU.
"""

import unittest
from typing import Optional

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from einops import rearrange

from seq2gene.attn_log_callback import (
    LogAttention,
    MatrixLogMetadata,
    create_heatmap,
    process_attention_matrix,
    process_attention_matrix_all_batches,
)
from seq2gene.modules.layers import (
    ContextFlashAttentionEncoderLayer,
    ContextFlashCrossAttentionEncoderLayer,
    EpigeneticsModulator,
    FlashAttentionEncoderLayer,
    FlashAttLayer,
    GeneModulator,
    get_alibi_slopes,
)
from seq2gene.model_combined_modulator import CombinedModulator


CUDA_AVAILABLE = torch.cuda.is_available()
SKIP_CUDA_REASON = "FlashAttention requires CUDA"


# ---------------------------------------------------------------------------
# Tiny model harness


class _TinyModel(pl.LightningModule):
    """Minimal LightningModule with the modulator attributes the callback
    auto-detects.

    It exposes both an ``epigenetics_modulator`` (an ``EpigeneticsModulator``
    wrapping a ``ModuleList`` of self-attention encoder layers) and a
    ``gene_modulator`` (a ``GeneModulator`` wrapping a ``ModuleList`` of
    cross-attention encoder layers), matching the structure the callback
    looks for in the production model.
    """

    def __init__(
        self,
        emb_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        use_alibi: bool = True,
    ):
        super().__init__()
        self.epigenetics_modulator = EpigeneticsModulator(
            emb_dim=emb_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            use_alibi=use_alibi,
            mlp_dout=0.0,
            use_context=False,
        )
        self.gene_modulator = GeneModulator(
            emb_dim=emb_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            use_alibi=use_alibi,
            mlp_dout=0.0,
            only_cross_attention=True,
            use_res=False,
            cross_alibi=False,
        )

    def forward(self, cre, gene, cre_mask, gene_mask, precision=None):
        modulator_outputs = self.epigenetics_modulator(
            cre,
            context=None,
            src_key_padding_mask=cre_mask,
            precision=precision,
            keep_intermediates_unpadded=True,
        )
        g = self.gene_modulator(
            gene,
            modulator_outputs,
            res=None,
            padding_mask=cre_mask,
            src_key_padding_mask=gene_mask,
            precision=precision,
        )
        return g


def _make_padding_mask(batch: int, seqlen: int, n_unpadded: int) -> torch.Tensor:
    """Build a [batch, seqlen] mask with True = padded."""
    mask = torch.ones(batch, seqlen, dtype=torch.bool)
    mask[:, :n_unpadded] = False
    return mask


def _build_inputs(
    batch: int = 2,
    cre_seq: int = 16,
    gene_seq: int = 8,
    cre_unpadded: int = 12,
    gene_unpadded: int = 6,
    emb_dim: int = 64,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
):
    """Build random inputs.

    The model weights remain in fp32; the ``precision`` argument passed to
    the modulator's forward controls when tensors are cast to fp16 for the
    FlashAttention call. So inputs default to fp32 here.
    """
    cre = torch.randn(batch, cre_seq, emb_dim, device=device, dtype=dtype)
    gene = torch.randn(batch, gene_seq, emb_dim, device=device, dtype=dtype)
    cre_mask = _make_padding_mask(batch, cre_seq, cre_unpadded).to(device)
    gene_mask = _make_padding_mask(batch, gene_seq, gene_unpadded).to(device)
    return cre, gene, cre_mask, gene_mask


# ---------------------------------------------------------------------------
# Pure CPU tests (no GPU needed)


class TestLayerAttributes(unittest.TestCase):
    def test_flash_att_layer_has_log_flags(self):
        layer = FlashAttLayer(d_model=64, nhead=4, use_alibi=True)
        self.assertFalse(layer.log_attn_matrix)
        self.assertIsNone(layer.attn_matrix)
        self.assertTrue(hasattr(layer, "calculate_attention_matrix"))
        self.assertTrue(layer.use_alibi)
        self.assertFalse(layer.causal)

    def test_flash_att_layer_default_use_alibi_false(self):
        layer = FlashAttLayer(d_model=64, nhead=4, use_alibi=False)
        self.assertFalse(layer.use_alibi)


class TestProcessAttentionHelpers(unittest.TestCase):
    def test_process_attention_matrix_none(self):
        self.assertIsNone(process_attention_matrix(None))
        self.assertIsNone(process_attention_matrix_all_batches(None))

    def test_process_attention_matrix_first_batch_head_avg(self):
        # [B=2, H=3, S_q=4, S_k=5]
        mat = torch.randn(2, 3, 4, 5)
        out = process_attention_matrix(mat)
        self.assertEqual(out.shape, (4, 5))
        # Should equal the head-averaged first batch element
        torch.testing.assert_close(out, mat[0].mean(dim=0))

    def test_process_attention_matrix_all_batches(self):
        mat = torch.randn(2, 3, 4, 5)
        out = process_attention_matrix_all_batches(mat)
        self.assertEqual(len(out), 2)
        for i, item in enumerate(out):
            self.assertEqual(item.shape, (4, 5))
            torch.testing.assert_close(item, mat[i].mean(dim=0))

    def test_process_attention_matrix_all_batches_keep_heads(self):
        """When ``keep_heads=True`` we must keep the (H, Q, K) shape."""
        mat = torch.randn(2, 3, 4, 5)
        out = process_attention_matrix_all_batches(mat, keep_heads=True)
        self.assertEqual(len(out), 2)
        for i, item in enumerate(out):
            self.assertEqual(item.shape, (3, 4, 5))
            torch.testing.assert_close(item, mat[i])


class TestMatrixLogMetadata(unittest.TestCase):
    def test_defaults(self):
        m = MatrixLogMetadata(
            modulator_name="epigenetics_modulator",
            mha_name="epigenetics_modulator_mixer_3",
            layer_id=3,
            step_number=10,
        )
        self.assertFalse(m.cross_attn)
        self.assertEqual(m.batch_idx, -1)
        self.assertEqual(m.aggregation_op, "mean")


class TestCreateHeatmap(unittest.TestCase):
    def test_create_heatmap_returns_figure(self):
        import matplotlib

        matplotlib.use("Agg")
        mat = np.random.rand(6, 8)
        fig = create_heatmap(mat, title="t", figsize=(4, 3))
        self.assertEqual(fig.axes[0].get_xlabel(), "Key position")
        self.assertEqual(fig.axes[0].get_ylabel(), "Query position")


class TestSetLogFlagsOnEncoderLayers(unittest.TestCase):
    """The flag-setting paths are CPU-only (just attribute toggles)."""

    def setUp(self):
        self.callback = LogAttention(layer_ids=[0])

    def test_context_flash_attention_encoder_layer(self):
        layer = ContextFlashAttentionEncoderLayer(d_model=64, nhead=4)
        self.callback._set_log_flags(layer, True)
        self.assertTrue(layer.mixer.log_attn_matrix)
        self.assertTrue(layer.crossMHA.log_attn_matrix)
        self.callback._set_log_flags(layer, False)
        self.assertFalse(layer.mixer.log_attn_matrix)
        self.assertFalse(layer.crossMHA.log_attn_matrix)

    def test_flash_attention_encoder_layer(self):
        layer = FlashAttentionEncoderLayer(d_model=64, nhead=4)
        self.callback._set_log_flags(layer, True)
        self.assertTrue(layer.mixer.log_attn_matrix)
        # Self-attention layer has no crossMHA
        self.assertFalse(hasattr(layer, "crossMHA"))

    def test_context_flash_cross_attention_encoder_layer(self):
        layer = ContextFlashCrossAttentionEncoderLayer(d_model=64, nhead=4)
        self.callback._set_log_flags(layer, True)
        self.assertTrue(layer.crossMHA.log_attn_matrix)

    def test_unknown_layer_raises(self):
        with self.assertRaises(AttributeError):
            self.callback._set_log_flags(nn.Linear(2, 2), True)


class TestGetModulators(unittest.TestCase):
    def test_dual_modulator_module(self):
        callback = LogAttention()

        class _Dual(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.epigenetics_modulator = EpigeneticsModulator(
                    emb_dim=32, num_heads=2, num_layers=2,
                    use_alibi=False, mlp_dout=0.0, use_context=False,
                )
                self.gene_modulator = GeneModulator(
                    emb_dim=32, num_heads=2, num_layers=2,
                    use_alibi=False, mlp_dout=0.0, only_cross_attention=True,
                )

        epi, gene = callback._get_modulators(_Dual())
        self.assertIsInstance(epi, nn.ModuleList)
        self.assertIsInstance(gene, nn.ModuleList)

    def test_combined_modulator_module(self):
        callback = LogAttention()

        class _Combined(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.combined_modulator = CombinedModulator(
                    emb_dim=32, num_heads=2, num_layers=2,
                    use_alibi=False, mlp_dout=0.0, use_context=False,
                    only_cross_attention=True,
                )

        epi, gene = callback._get_modulators(_Combined())
        self.assertIsInstance(epi, nn.ModuleList)
        self.assertIsInstance(gene, nn.ModuleList)

    def test_missing_attributes_raises(self):
        callback = LogAttention()

        class _Bad(pl.LightningModule):
            pass

        with self.assertRaises(AttributeError):
            callback._get_modulators(_Bad())


# ---------------------------------------------------------------------------
# GPU-required tests: actually run a forward pass and verify capture.


@pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_CUDA_REASON)
class TestRecordAttentionEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.emb_dim = 64
        cls.num_heads = 4
        cls.num_layers = 3
        cls.model = _TinyModel(
            emb_dim=cls.emb_dim,
            num_heads=cls.num_heads,
            num_layers=cls.num_layers,
            use_alibi=True,
        ).to("cuda").eval()

    def _forward(self):
        cre, gene, cre_mask, gene_mask = _build_inputs(
            batch=2,
            cre_seq=16,
            gene_seq=8,
            cre_unpadded=12,
            gene_unpadded=6,
            emb_dim=self.emb_dim,
            device="cuda",
            dtype=torch.float32,
        )
        with torch.no_grad():
            _ = self.model(cre, gene, cre_mask, gene_mask, precision=torch.float16)

    def test_record_attention_populates_dict(self):
        log_attn = LogAttention(layer_ids=[0, 1], log_heatmaps=False)
        with log_attn.record_attention(self.model):
            self._forward()

        self.assertGreater(len(log_attn.attention_matrices), 0)
        # We expect one self-attn entry per epigenetics layer (mixer) and one
        # cross-attn entry per gene layer (crossMHA), for layers 0 and 1.
        expected_epi_keys = {
            "epigenetics_modulator_mixer_0",
            "epigenetics_modulator_mixer_1",
        }
        expected_gene_keys = {
            "gene_modulator_crossMHA_0",
            "gene_modulator_crossMHA_1",
        }
        for k in expected_epi_keys | expected_gene_keys:
            self.assertIn(k, log_attn.attention_matrices, f"missing {k}")
            mats = log_attn.attention_matrices[k]
            self.assertEqual(len(mats), 2)  # batch size = 2
            for m in mats:
                self.assertEqual(m.dim(), 2)  # head-averaged, so 2D
                # softmax rows on unpadded query positions sum to ~1
                # (head-averaged so this is approximate but still close)
                row_sums = m.sum(dim=-1)
                # The first 6 (gene) or 12 (epi) rows should sum to ~1.
                non_zero_rows = row_sums[row_sums > 0]
                self.assertTrue(
                    torch.all(torch.abs(non_zero_rows - 1.0) < 0.05),
                    f"row sums for {k} were not ~1: {non_zero_rows}",
                )

    def test_record_attention_skips_layers_outside_layer_ids(self):
        log_attn = LogAttention(layer_ids=[0], log_heatmaps=False)
        with log_attn.record_attention(self.model):
            self._forward()
        keys = list(log_attn.attention_matrices.keys())
        for k in keys:
            self.assertTrue(k.endswith("_0"), f"unexpected key {k}")

    def test_log_flags_are_reset_after_context_exit(self):
        log_attn = LogAttention(layer_ids=[0, 1], log_heatmaps=False)
        with log_attn.record_attention(self.model):
            self._forward()
        # After exit, no FlashAttLayer should still have log_attn_matrix=True
        for layer in self.model.epigenetics_modulator.epigenetics_modulator:
            self.assertFalse(layer.mixer.log_attn_matrix)
        for layer in self.model.gene_modulator.gene_modulator:
            self.assertFalse(layer.crossMHA.log_attn_matrix)

    def test_recorded_matrices_are_on_cpu(self):
        log_attn = LogAttention(layer_ids=[0], log_heatmaps=False)
        with log_attn.record_attention(self.model):
            self._forward()
        for mats in log_attn.attention_matrices.values():
            for m in mats:
                self.assertEqual(m.device.type, "cpu")

    def test_record_attention_keep_heads_preserves_head_dimension(self):
        """``keep_heads=True`` must keep ``(H, Q, K)`` per-batch entries."""
        log_attn = LogAttention(
            layer_ids=[0], log_heatmaps=False, keep_heads=True
        )
        with log_attn.record_attention(self.model):
            self._forward()

        self.assertGreater(len(log_attn.attention_matrices), 0)
        for k, mats in log_attn.attention_matrices.items():
            for m in mats:
                self.assertEqual(m.dim(), 3, f"{k}: expected (H, Q, K) tensor")
                self.assertEqual(m.shape[0], self.num_heads)
                self.assertEqual(m.device.type, "cpu")

    def test_record_attention_default_averages_heads(self):
        """Default behaviour (``keep_heads=False``) must collapse heads."""
        log_attn = LogAttention(layer_ids=[0], log_heatmaps=False)
        with log_attn.record_attention(self.model):
            self._forward()
        for mats in log_attn.attention_matrices.values():
            for m in mats:
                self.assertEqual(m.dim(), 2)  # (Q, K), heads averaged out

    def test_record_attention_with_only_epigenetics(self):
        log_attn = LogAttention(layer_ids=[0], log_heatmaps=False)
        with log_attn.record_attention(self.model, log_epigenetics=True, log_gene=False):
            self._forward()
        for k in log_attn.attention_matrices:
            self.assertTrue(k.startswith("epigenetics_modulator"))

    def test_disabling_log_does_not_change_forward_output(self):
        """Sanity: log_attn_matrix=False should leave the forward output
        identical, since the recompute happens on a side path."""
        cre, gene, cre_mask, gene_mask = _build_inputs(
            batch=2, emb_dim=self.emb_dim, device="cuda", dtype=torch.float32
        )
        with torch.no_grad():
            out_no_log = self.model(cre, gene, cre_mask, gene_mask, precision=torch.float16)

            log_attn = LogAttention(layer_ids=[0, 1], log_heatmaps=False)
            with log_attn.record_attention(self.model):
                out_with_log = self.model(
                    cre, gene, cre_mask, gene_mask, precision=torch.float16
                )

        # Forward outputs must match exactly: log path is read-only relative
        # to the main computation.
        torch.testing.assert_close(out_no_log, out_with_log)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_CUDA_REASON)
class TestCalculateAttentionMatrixDirect(unittest.TestCase):
    """Test ``FlashAttLayer.calculate_attention_matrix`` directly.

    These run in fp32 to keep the softmax row-sum tolerances tight; the
    method does not depend on FlashAttention so it works on either dtype.
    """

    def test_self_attention_matrix_shape_and_softmax(self):
        torch.manual_seed(0)
        d_model, nhead, B, S = 64, 4, 2, 8
        layer = FlashAttLayer(d_model, nhead, use_alibi=True).cuda()
        src = torch.randn(B, S, d_model, device="cuda")
        attn = layer.calculate_attention_matrix(
            src, max_seqlen_q=S, max_seqlen_k=S
        )
        self.assertEqual(attn.shape, (B, nhead, S, S))
        # Each row sums to 1 (softmax over keys).
        row_sums = attn.float().sum(dim=-1)
        torch.testing.assert_close(
            row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5
        )

    def test_cross_attention_matrix_shape(self):
        torch.manual_seed(1)
        d_model, nhead, B, S_q, S_k = 64, 4, 2, 8, 11
        layer = FlashAttLayer(
            d_model, nhead, use_alibi=True, cross_attn=True
        ).cuda()
        src = torch.randn(B, S_q, d_model, device="cuda")
        cntx = torch.randn(B, S_k, d_model, device="cuda")
        attn = layer.calculate_attention_matrix(
            src, cntx, max_seqlen_q=S_q, max_seqlen_k=S_k
        )
        self.assertEqual(attn.shape, (B, nhead, S_q, S_k))
        row_sums = attn.float().sum(dim=-1)
        torch.testing.assert_close(
            row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5
        )

    def test_padding_mask_zeroes_attention(self):
        torch.manual_seed(2)
        d_model, nhead, B, S = 64, 4, 2, 8
        layer = FlashAttLayer(d_model, nhead, use_alibi=False).cuda()
        src = torch.randn(B, S, d_model, device="cuda")
        # Mark last 3 positions as padded for both batches.
        mask = torch.zeros(B, S, dtype=torch.bool, device="cuda")
        mask[:, -3:] = True
        attn = layer.calculate_attention_matrix(
            src,
            src_key_padding_mask=mask,
            context_key_padding_mask=mask,
            max_seqlen_q=S - 3,
            max_seqlen_k=S - 3,
        )
        # Padded query rows should have all zeros.
        padded_rows = attn[:, :, -3:, :]
        self.assertTrue(torch.all(padded_rows == 0))
        # Padded key columns should have all zeros (because softmax
        # of -inf is 0).
        padded_cols = attn[:, :, :-3, -3:]
        self.assertTrue(torch.all(padded_cols == 0))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_CUDA_REASON)
class TestManualAttentionMatchesFlashAttention(unittest.TestCase):
    """Verify the *manually computed* attention matrix is consistent with
    what FlashAttention actually does.

    The strategy:

    1. Build a ``FlashAttLayer`` with random weights.
    2. Run ``self.MHA(src)`` – this is the real (FlashAttention) output and is
       defined to be ``out_proj(attn @ V)`` where ``attn = softmax(QK^T/sqrt(d)
       + alibi_bias)`` and ``Q,K,V`` come from the same ``Wqkv``/``Wq``/``Wkv``
       projections we use in ``calculate_attention_matrix``.
    3. Recompute QKV ourselves from the layer's projection modules, take our
       captured ``attn`` matrix, multiply by ``V``, reshape, and apply the
       same ``out_proj``.
    4. Assert the two outputs match within a tolerance dominated by fp16
       precision.

    If steps 1 and 3 produce the same tensor, then the attention matrix our
    callback exposes really is the per-head attention probability matrix
    used by FlashAttention. This is the key correctness property the
    callback advertises.
    """

    def _qkv_from_layer(
        self, layer: FlashAttLayer, src: torch.Tensor, cntx: Optional[torch.Tensor]
    ):
        """Replicate flash_attn's QKV projection so our V matches theirs."""
        if layer.cross_attn:
            assert cntx is not None
            Q = layer.MHA.Wq(src)
            KV = layer.MHA.Wkv(cntx)
            Q = rearrange(Q, "b s (h d) -> b s h d", d=layer.MHA.head_dim)
            KV = rearrange(
                KV, "b s (two h d) -> two b s h d", two=2, d=layer.MHA.head_dim
            )
            K, V = KV
        else:
            QKV = layer.MHA.Wqkv(src)
            QKV = rearrange(
                QKV, "b s (three h d) -> three b s h d", three=3, d=layer.MHA.head_dim
            )
            Q, K, V = QKV
        return Q, K, V  # each [B, S_*, H, D]

    def _reconstruct_mha_output(
        self,
        layer: FlashAttLayer,
        attn: torch.Tensor,  # [B, H, S_q, S_k]
        V: torch.Tensor,  # [B, S_k, H, D]
    ) -> torch.Tensor:
        """Apply attn @ V then the layer's out_proj to mimic MHA's full forward."""
        # attn @ V: [B, H, S_q, S_k] @ [B, H, S_k, D] -> [B, H, S_q, D]
        V_h = rearrange(V, "b s h d -> b h s d")
        out_h = torch.einsum("bhqk,bhkd->bhqd", attn, V_h)  # [B, H, S_q, D]
        # FlashAttention's MHA output is shape [B, S_q, D_model] after out_proj
        out = rearrange(out_h, "b h s d -> b s (h d)")
        out = layer.MHA.out_proj(out)
        return out

    @staticmethod
    def _build_layer(use_alibi: bool, cross_attn: bool, d_model=64, nhead=4):
        # dropout=0.0 + .eval() so dropout never fires; with fp16 + dropout
        # active, FlashAttention's randomness would make the comparison
        # non-deterministic. This mirrors how the layer is used at inference.
        layer = (
            FlashAttLayer(
                d_model,
                nhead,
                dropout=0.0,
                use_alibi=use_alibi,
                cross_attn=cross_attn,
            )
            .cuda()
            .to(torch.float16)
        )
        layer.eval()
        return layer

    def test_self_attention_no_alibi_matches_real_mha_output(self):
        """Self-attention without ALiBi: ``out_proj(my_attn @ V)`` must equal
        the real ``MHA(src)`` (which is ``out_proj(FlashAttn(Wqkv(src)))``).
        """
        torch.manual_seed(0)
        d_model, nhead, B, S = 64, 4, 2, 16
        layer = self._build_layer(use_alibi=False, cross_attn=False, d_model=d_model, nhead=nhead)
        src = torch.randn(B, S, d_model, device="cuda", dtype=torch.float16)

        with torch.no_grad():
            real_out = layer.MHA(src)  # FlashAttention output
            attn = layer.calculate_attention_matrix(
                src, max_seqlen_q=S, max_seqlen_k=S
            )
            _, _, V = self._qkv_from_layer(layer, src, None)
            manual_out = self._reconstruct_mha_output(layer, attn, V)

        torch.testing.assert_close(real_out, manual_out, atol=1e-2, rtol=1e-2)

    def test_self_attention_with_alibi_matches_real_mha_output(self):
        """Same equivalence with ALiBi turned on, exercising the ALiBi branch
        of ``calculate_attention_matrix``."""
        torch.manual_seed(1)
        d_model, nhead, B, S = 64, 4, 2, 16
        layer = self._build_layer(use_alibi=True, cross_attn=False, d_model=d_model, nhead=nhead)
        src = torch.randn(B, S, d_model, device="cuda", dtype=torch.float16)

        with torch.no_grad():
            real_out = layer.MHA(src)
            attn = layer.calculate_attention_matrix(
                src, max_seqlen_q=S, max_seqlen_k=S
            )
            _, _, V = self._qkv_from_layer(layer, src, None)
            manual_out = self._reconstruct_mha_output(layer, attn, V)

        torch.testing.assert_close(real_out, manual_out, atol=1e-2, rtol=1e-2)

    def test_cross_attention_no_alibi_matches_real_mha_output(self):
        """Cross-attention with different Q/K lengths."""
        torch.manual_seed(2)
        d_model, nhead, B, S_q, S_k = 64, 4, 2, 8, 13
        layer = self._build_layer(use_alibi=False, cross_attn=True, d_model=d_model, nhead=nhead)
        src = torch.randn(B, S_q, d_model, device="cuda", dtype=torch.float16)
        cntx = torch.randn(B, S_k, d_model, device="cuda", dtype=torch.float16)

        with torch.no_grad():
            real_out = layer.MHA(src, cntx)  # cross-attention output
            attn = layer.calculate_attention_matrix(
                src, cntx, max_seqlen_q=S_q, max_seqlen_k=S_k
            )
            _, _, V = self._qkv_from_layer(layer, src, cntx)
            manual_out = self._reconstruct_mha_output(layer, attn, V)

        torch.testing.assert_close(real_out, manual_out, atol=1e-2, rtol=1e-2)

    def test_cross_attention_with_alibi_matches_real_mha_output(self):
        torch.manual_seed(3)
        d_model, nhead, B, S_q, S_k = 64, 4, 2, 8, 13
        layer = self._build_layer(use_alibi=True, cross_attn=True, d_model=d_model, nhead=nhead)
        src = torch.randn(B, S_q, d_model, device="cuda", dtype=torch.float16)
        cntx = torch.randn(B, S_k, d_model, device="cuda", dtype=torch.float16)

        with torch.no_grad():
            real_out = layer.MHA(src, cntx)
            attn = layer.calculate_attention_matrix(
                src, cntx, max_seqlen_q=S_q, max_seqlen_k=S_k
            )
            _, _, V = self._qkv_from_layer(layer, src, cntx)
            manual_out = self._reconstruct_mha_output(layer, attn, V)

        torch.testing.assert_close(real_out, manual_out, atol=1e-2, rtol=1e-2)

    def test_self_attention_matches_torch_sdpa_no_alibi(self):
        """Cross-check against ``torch.nn.functional.scaled_dot_product_attention``
        as an independent reference for self-attention with no ALiBi."""
        torch.manual_seed(4)
        d_model, nhead, B, S = 64, 4, 2, 12
        layer = (
            FlashAttLayer(d_model, nhead, use_alibi=False, cross_attn=False)
            .cuda()
        )
        src = torch.randn(B, S, d_model, device="cuda")

        with torch.no_grad():
            attn = layer.calculate_attention_matrix(
                src, max_seqlen_q=S, max_seqlen_k=S
            )
            Q, K, V = self._qkv_from_layer(layer, src, None)
            # SDPA expects [B, H, S, D]
            Qh = rearrange(Q, "b s h d -> b h s d")
            Kh = rearrange(K, "b s h d -> b h s d")
            Vh = rearrange(V, "b s h d -> b h s d")
            sdpa_out = F.scaled_dot_product_attention(Qh, Kh, Vh)  # [B, H, S, D]
            # attn @ V (manually) should give the same per-head outputs
            manual_out = torch.einsum("bhqk,bhkd->bhqd", attn, Vh)

        torch.testing.assert_close(manual_out, sdpa_out, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
