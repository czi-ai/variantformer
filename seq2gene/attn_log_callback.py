"""Attention-matrix logging callback for VariantFormer.

This module provides a Lightning callback (``LogAttention``) and a context
manager (``LogAttention.record_attention``) that capture the per-layer,
per-head attention matrices from the transformer layers in VariantFormer's
``epigenetics_modulator``/``gene_modulator`` (or the equivalent
``combined_modulator.cre_layers``/``combined_modulator.gene_layers`` in the
memory-optimized model) during a forward pass.

Internally it works by toggling ``log_attn_matrix`` on the
:class:`~seq2gene.modules.layers.FlashAttLayer` instances of interest before
the forward pass and reading the resulting ``attn_matrix`` tensor afterwards.
The capture path bypasses FlashAttention to recompute a plain
softmax(Q K^T / sqrt(d)) attention matrix (with ALiBi if enabled), so it is
intentionally heavier than the production path; only enable it for the layers
you actually want to inspect.

Typical usage with a one-off forward pass::

    from seq2gene.attn_log_callback import LogAttention

    log_attn = LogAttention(layer_ids=[0, 4, 8])
    with log_attn.record_attention(model, log_epigenetics=True, log_gene=True):
        _ = model(*inputs)
    # log_attn.attention_matrices is now a dict of
    # {f"{modulator}_{mha}_{layer_idx}": [tensor_per_batch_item, ...]}
"""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from typing import Iterable, Optional

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

from seq2gene.modules.layers import (
    ContextFlashAttentionEncoderLayer,
    ContextFlashCrossAttentionEncoderLayer,
    FlashAttentionEncoderLayer,
)


@dataclasses.dataclass
class MatrixLogMetadata:
    """Metadata used to label a captured attention matrix for plotting."""

    modulator_name: str  # one of "epigenetics_modulator" / "gene_modulator"
    mha_name: str  # human-readable mha name (e.g. "epigenetics_modulator_mixer_3")
    layer_id: int
    step_number: int
    cross_attn: bool = False
    batch_idx: int = -1  # -1 means "first batch" / unspecified
    aggregation_op: str = "mean"  # one of "mean", "sum", "max"


def create_heatmap(
    mat,
    title: str,
    figsize: tuple[int, int] = (12, 10),
    cmap: str = "viridis",
) -> plt.Figure:
    """Render an attention matrix as a Matplotlib heatmap.

    Args:
        mat: 2D ``np.ndarray`` of attention weights.
        title: figure title.
        figsize: figure size in inches.
        cmap: matplotlib colormap.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(mat, cmap=cmap, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    plt.tight_layout()
    return fig


def process_attention_matrix(matrix: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Take the first batch element, average over heads, return on CPU."""
    if matrix is None:
        return None
    processed = matrix[0]  # first batch element
    processed = torch.mean(processed, dim=0)  # average over heads
    return processed.detach().cpu()


def process_attention_matrix_all_batches(
    matrix: Optional[torch.Tensor],
    keep_heads: bool = False,
) -> Optional[list[torch.Tensor]]:
    """Return one CPU tensor per batch element.

    If ``keep_heads`` is False (default) the head dimension is averaged out
    and each entry has shape ``(Q, K)``. If True, each entry retains its
    full ``(H, Q, K)`` shape.
    """
    if matrix is None:
        return None
    processed: list[torch.Tensor] = []
    for batch_item in matrix:  # batch_item: (H, Q, K)
        if keep_heads:
            processed.append(batch_item.detach().cpu())
        else:
            processed.append(torch.mean(batch_item, dim=0).detach().cpu())
    return processed


def create_heatmap_from_matrix(
    mlm: MatrixLogMetadata, matrix: Optional[torch.Tensor]
) -> Optional[plt.Figure]:
    """Build a labeled heatmap for one already-processed (head-averaged) matrix."""
    if matrix is None:
        return None
    cmap = "RdBu_r" if mlm.cross_attn else "magma"
    cmat = matrix.to(torch.float32).numpy()
    step_str = str(mlm.step_number).zfill(6)
    fig = create_heatmap(
        cmat,
        title=f"{mlm.modulator_name}_{mlm.mha_name}_L{mlm.layer_id} @ step {step_str}",
        figsize=(14, 12),
        cmap=cmap,
    )
    return fig


_LAYER_TYPES = (
    ContextFlashAttentionEncoderLayer,
    FlashAttentionEncoderLayer,
    ContextFlashCrossAttentionEncoderLayer,
)


class LogAttention(pl.Callback):
    """Lightning callback that captures attention matrices from selected layers.

    It can be used in two modes:

    1. **As a Lightning callback** during ``trainer.predict``: it hooks
       ``on_predict_batch_start``/``on_predict_batch_end`` and captures
       attention every ``freq_steps`` batches.
    2. **As a context manager** via :meth:`record_attention` for a single
       manual forward pass.

    Captured matrices are accumulated on ``self.attention_matrices`` keyed by
    ``f"{modulator_name}_{mha_name}_{layer_idx}"``. Use the
    :func:`create_heatmap` / :func:`create_heatmap_from_matrix` helpers
    (matplotlib + seaborn) if you want to render them, or plot them yourself.

    Args:
        freq_steps: capture every Nth predict batch (only relevant in
            callback mode).
        layer_ids: layer indices (within each modulator's ``ModuleList``) to
            capture. Indices outside the list are silently skipped.
        keep_heads: if True, keep the per-head dimension when accumulating
            matrices (each entry has shape ``(H, Q, K)``); if False (default),
            average over heads (each entry has shape ``(Q, K)``).
    """

    def __init__(
        self,
        freq_steps: int = 100,
        layer_ids: Optional[Iterable[int]] = None,
        keep_heads: bool = False,
    ):
        super().__init__()
        if layer_ids is None:
            layer_ids = [1, 4, 9, 13, 17, 23, 24]
        self.freq_steps = freq_steps
        self.layer_ids = list(layer_ids)
        self.keep_heads = keep_heads
        self.attention_matrices: dict[str, list[torch.Tensor]] = {}

    # ------------------------------------------------------------------ utils
    def _should_process_batch(self, batch_idx: int) -> bool:
        return batch_idx != 0 and batch_idx % self.freq_steps == 0

    def _get_modulators(
        self, pl_module: pl.LightningModule
    ) -> tuple[nn.Module, nn.Module]:
        """Return ``(epigenetics_layers, gene_layers)`` for either model variant.

        Supports both the dual-modulator model
        (``Seq2GenePredictor.epigenetics_modulator`` /
        ``Seq2GenePredictor.gene_modulator``) and the memory-optimized
        ``Seq2GenePredictorCombinedModulator.combined_modulator``.
        """
        if hasattr(pl_module, "combined_modulator"):
            return (
                pl_module.combined_modulator.cre_layers,
                pl_module.combined_modulator.gene_layers,
            )
        if hasattr(pl_module, "epigenetics_modulator") and hasattr(
            pl_module, "gene_modulator"
        ):
            return (
                pl_module.epigenetics_modulator.epigenetics_modulator,
                pl_module.gene_modulator.gene_modulator,
            )
        raise AttributeError(
            "Model must have either 'combined_modulator' or both "
            "'epigenetics_modulator' and 'gene_modulator' attributes"
        )

    def _set_log_flags(
        self,
        layer: nn.Module,
        set_to: bool,
    ) -> None:
        """Toggle ``log_attn_matrix`` on every FlashAttLayer inside ``layer``."""
        if isinstance(layer, ContextFlashAttentionEncoderLayer):
            layer.crossMHA.log_attn_matrix = set_to
            layer.mixer.log_attn_matrix = set_to
        elif isinstance(layer, FlashAttentionEncoderLayer):
            layer.mixer.log_attn_matrix = set_to
        elif isinstance(layer, ContextFlashCrossAttentionEncoderLayer):
            layer.crossMHA.log_attn_matrix = set_to
        else:
            raise AttributeError(f"unknown layer type: {type(layer)}")

    def _process_attention_matrices(
        self,
        layer: nn.Module,
        layer_idx: int,
        modulator_name: str,
    ) -> dict[str, list[torch.Tensor]]:
        """Pull captured matrices off a layer and return them keyed by mha name."""
        matrices: dict[str, list[torch.Tensor]] = {}
        mhas: list[tuple[nn.Module, str]] = []

        if isinstance(
            layer,
            (ContextFlashAttentionEncoderLayer, ContextFlashCrossAttentionEncoderLayer),
        ) and hasattr(layer, "crossMHA"):
            mhas.append((layer.crossMHA, f"{modulator_name}_crossMHA_{layer_idx}"))
        if isinstance(
            layer, (ContextFlashAttentionEncoderLayer, FlashAttentionEncoderLayer)
        ) and hasattr(layer, "mixer"):
            mhas.append((layer.mixer, f"{modulator_name}_mixer_{layer_idx}"))

        for mha, mha_name in mhas:
            if not hasattr(mha, "attn_matrix") or mha.attn_matrix is None:
                continue
            processed_matrices = process_attention_matrix_all_batches(
                mha.attn_matrix,
                keep_heads=self.keep_heads,
            )
            # Free GPU memory from the captured raw tensor immediately.
            mha.attn_matrix = None
            if processed_matrices is not None:
                matrices[mha_name] = processed_matrices

        return matrices

    def _process_modulator(
        self,
        modulator: nn.Module,
        modulator_name: str,
        step_number: int,
        set_flags: bool,
    ) -> None:
        """Walk a modulator's ``ModuleList`` and toggle / extract on selected layers."""
        for layer_idx, layer in enumerate(modulator):
            if layer_idx not in self.layer_ids:
                continue
            self._set_log_flags(layer, set_flags)
            if not set_flags:  # teardown phase: now read & clear
                matrices = self._process_attention_matrices(
                    layer, layer_idx, modulator_name
                )
                if matrices:
                    self.attention_matrices.update(matrices)

    # ------------------------------------------------------------- public API
    def reset(self) -> None:
        """Clear any previously recorded matrices."""
        self.attention_matrices = {}

    @contextmanager
    def record_attention(
        self,
        pl_module: pl.LightningModule,
        step_number: int = 0,
        log_epigenetics: bool = True,
        log_gene: bool = True,
    ):
        """Context manager that records attention for a single forward pass.

        Example::

            log_attn = LogAttention(layer_ids=[0, 4])
            with log_attn.record_attention(model):
                _ = model(*inputs)
            log_attn.attention_matrices  # populated
        """
        epigenetics_modulator, gene_modulator = self._get_modulators(pl_module)
        if log_epigenetics:
            self._process_modulator(
                epigenetics_modulator, "epigenetics_modulator", step_number, True
            )
        if log_gene:
            self._process_modulator(
                gene_modulator, "gene_modulator", step_number, True
            )

        try:
            yield self
        finally:
            if log_epigenetics:
                self._process_modulator(
                    epigenetics_modulator,
                    "epigenetics_modulator",
                    step_number,
                    False,
                )
            if log_gene:
                self._process_modulator(
                    gene_modulator, "gene_modulator", step_number, False
                )

    # --------------------------------------------------------- lightning hooks
    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx):  # noqa: D401
        if not self._should_process_batch(batch_idx):
            return
        epigenetics_modulator, gene_modulator = self._get_modulators(pl_module)
        self._process_modulator(
            epigenetics_modulator,
            "epigenetics_modulator",
            trainer.global_step,
            True,
        )
        self._process_modulator(
            gene_modulator, "gene_modulator", trainer.global_step, True
        )

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):  # noqa: D401
        if not self._should_process_batch(batch_idx):
            return
        epigenetics_modulator, gene_modulator = self._get_modulators(pl_module)
        self._process_modulator(
            epigenetics_modulator,
            "epigenetics_modulator",
            trainer.global_step,
            False,
        )
        self._process_modulator(
            gene_modulator, "gene_modulator", trainer.global_step, False
        )
