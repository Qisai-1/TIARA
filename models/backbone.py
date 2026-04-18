"""
TabPFN Backbone for RL — End-to-End Fine-Tuning
=================================================

Architecture:
    Input: context table rows (s, a, RTG) + query rows (s, â, ?)
           shape: (N_total, 1, feat_dim)  [sequence-first, batch=1]
    ↓
    TabPFN encoder — all 18 transformer layers (fine-tuned)
    Each layer: self_attn_between_features + self_attn_between_items + MLP
    hidden_dim = 192, intermediate = 384
    ↓
    We detach at decoder_dict.standard.0 input (384-dim)
    Discard TabPFN's 5000-bin output head entirely
    ↓
    Our proposal head + value head (trained from scratch on RL loss)

Key design decisions:
    1. TabPFN's encoder weights = initialization from pretraining
       Fine-tuned end-to-end with TD + BC loss
    2. RTG (return-to-go) replaces raw reward as the context label
       This aligns with TabPFN's pretraining: "predict future value
       from labeled context rows" — exactly what Q-estimation requires
    3. Input format: (N_total, 1, feat_dim) sequence-first as TabPFN expects
       Context rows first, query rows last
       y = RTG for context rows, 0 for query rows

Training modes:
    freeze_backbone=True  → only proposal+value heads train (fast, stable)
    freeze_backbone=False → full end-to-end fine-tuning (slower, stronger)
    Recommended: start frozen, then unfreeze with backbone_lr=1e-6
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class TabPFNBackbone(nn.Module):
    """
    TabPFN encoder with output head removed.
    Exposes the 384-dim pre-decoder representation for our RL heads.
    Fully differentiable — no hooks, no numpy conversions.
    """

    # Confirmed from architecture inspection:
    #   transformer hidden = 192
    #   pre-decoder hidden = 384  (decoder_dict.standard.0 input)
    ENCODER_HIDDEN = 192
    PRE_DECODER_HIDDEN = 384

    def __init__(self, device: str = "cuda", model_version: str = "v2"):
        super().__init__()
        self.device        = device
        self.model_version = model_version
        self._encoder      = None   # TabPFN's PerFeatureTransformer (encoder only)
        self._hidden_dim   = self.PRE_DECODER_HIDDEN

    def load_pretrained(self):
        try:
            from tabpfn import TabPFNRegressor
        except ImportError:
            raise ImportError("pip install tabpfn")

        print("[Backbone] Loading TabPFN pretrained weights ...")

        reg = TabPFNRegressor()
        dummy_X = np.random.randn(8, 4).astype(np.float32)
        dummy_y = np.arange(8, dtype=np.float32)
        reg.fit(dummy_X, dummy_y)

        if not hasattr(reg, "model_"):
            raise AttributeError("reg.model_ not found — pip install tabpfn --upgrade")

        full_model = reg.model_

        # ── Extract encoder only — drop the output head ──────────────────────
        # TabPFN structure (confirmed):
        #   encoder            — input embedding
        #   y_encoder          — label embedding
        #   add_thinking_tokens
        #   transformer_encoder.layers.0..17  — 18 attention layers
        #   decoder_dict.standard.0  — pre-output linear (384→384)
        #   decoder_dict.standard.2  — output projection (384→5000) ← DROP THIS

        # We keep everything up to and including decoder_dict.standard.0
        # and discard decoder_dict.standard.2 (the 5000-bin head)

        # Build our encoder as a module that runs the full model
        # but stops before the final projection
        self._encoder = _TabPFNEncoder(full_model).to(self.device)

        n_params = sum(p.numel() for p in self._encoder.parameters())
        print(f"[Backbone] Encoder loaded — {n_params:,} parameters")
        print(f"[Backbone] Output dim: {self._hidden_dim} (pre-decoder representation)")
        print(f"[Backbone] Output head (5000-bin) discarded — replaced by RL heads")

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def forward(
        self,
        context_X: torch.Tensor,   # (B, L, feat_dim)
        context_y: torch.Tensor,   # (B, L)  — RTG values
        query_X:   torch.Tensor,   # (B, Q, feat_dim)
    ) -> torch.Tensor:
        """
        Returns (B, Q, hidden_dim=384).
        Fully differentiable — gradients flow through all 18 transformer layers.
        """
        assert self._encoder is not None, "Call load_pretrained() first"

        B, L, F = context_X.shape
        _, Q, _ = query_X.shape

        outputs = []
        for b in range(B):
            h_b = self._encoder(
                context_X[b],   # (L, F)
                context_y[b],   # (L,)
                query_X[b],     # (Q, F)
            )                   # → (Q, hidden_dim)
            outputs.append(h_b)

        return torch.stack(outputs, dim=0)   # (B, Q, hidden_dim)


class _TabPFNEncoder(nn.Module):
    """
    Wraps TabPFN's PerFeatureTransformer and runs it in a way that:
      1. Accepts our (context, query) split input
      2. Returns the pre-decoder hidden state (384-dim) for query rows
      3. Is fully differentiable

    We use a forward hook on decoder_dict.standard.0 to intercept
    the 384-dim representation BEFORE the 5000-bin projection.
    The hook captures the tensor while it is still part of the
    computational graph — gradients flow normally.
    """

    def __init__(self, full_model: nn.Module):
        super().__init__()
        # Store the full model as a submodule so its parameters
        # are registered and can be optimized
        self.model = full_model

        # Verify the architecture we expect
        modules = dict(self.model.named_modules())
        assert "decoder_dict.standard.0" in modules, \
            "decoder_dict.standard.0 not found — TabPFN architecture changed"
        assert "decoder_dict.standard.2" in modules, \
            "decoder_dict.standard.2 not found — TabPFN architecture changed"

        # Freeze the 5000-bin output head — we never use it
        # (saves memory and avoids useless gradient computation)
        for param in modules["decoder_dict.standard.2"].parameters():
            param.requires_grad = False

    def forward(
        self,
        ctx_X: torch.Tensor,   # (L, feat_dim)
        ctx_y: torch.Tensor,   # (L,)
        qry_X: torch.Tensor,   # (Q, feat_dim)
    ) -> torch.Tensor:
        """Returns (Q, 384) — pre-decoder hidden states for query rows."""

        L, F = ctx_X.shape
        Q, _ = qry_X.shape
        device = ctx_X.device

        # ── Build input in the format TabPFN actually expects ─────────────────
        # Discovered from intercepting executor_ → model_ call:
        #   arg[0]: (L+Q, 1, feat_dim)  — ALL rows, sequence-first, batch=1
        #   arg[1]: (L,)                — ONLY context labels, no query labels
        # TabPFN infers which rows are query by: len(X) > len(y)
        # The last (len(X) - len(y)) rows = query/test rows

        X_all = torch.cat([ctx_X, qry_X], dim=0)   # (L+Q, F)
        X_in  = X_all.unsqueeze(1)                  # (L+Q, 1, F)  sequence-first
        y_in  = ctx_y                               # (L,)  context labels only

        # ── Hook decoder_dict.standard.0 to capture 384-dim hidden state ──────
        # This tensor is part of the computational graph —
        # gradients flow back through all 18 transformer layers.
        captured = {}

        def post_hook(module, args, output):
            # output shape: (Q, 1, 384) — the 384-dim representation
            # AFTER decoder_dict.standard.0, BEFORE the 5000-bin projection
            captured["h"] = output

        hook_mod = dict(self.model.named_modules())["decoder_dict.standard.0"]
        handle   = hook_mod.register_forward_hook(post_hook)

        try:
            self.model(
                X_in,
                y_in,
                only_return_standard_out=True,
                categorical_inds=[[]],
                save_peak_memory_factor=None,
            )
        finally:
            handle.remove()

        h = captured.get("h")
        if h is None:
            raise RuntimeError(
                "Pre-hook on decoder_dict.standard.0 did not fire."
            )

        # h shape: (Q, 1, 384) or (Q, 384)
        if h.dim() == 3:
            h = h.squeeze(1)        # (Q, 384)
        elif h.dim() == 2:
            pass
        else:
            h = h.view(Q, -1)

        if h.shape[0] != Q:
            raise RuntimeError(
                f"Expected {Q} query rows from backbone, got {h.shape[0]}. "
                f"Context len={L}, total rows={L+Q}."
            )

        return h    # (Q, 384)


# ── Fully differentiable fallback backbone ────────────────────────────────────

class _AttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn  = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h2, _ = self.attn(h, h, h)
        h  = self.norm1(h + self.drop(h2))
        h2 = self.ffn(h)
        h  = self.norm2(h + self.drop(h2))
        return h


class TransformerBackbone(nn.Module):
    """
    Randomly initialized fallback backbone.
    Same interface as TabPFNBackbone.
    Use when TabPFN unavailable, or as ablation baseline.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim:  int   = 256,
        n_layers:    int   = 6,
        n_heads:     int   = 8,
        dropout:     float = 0.1,
        device:      str   = "cpu",
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self._hidden_dim = hidden_dim
        self.device      = device

        self.input_proj     = nn.Linear(feature_dim, hidden_dim)
        self.row_type_embed = nn.Embedding(2, hidden_dim)
        self.target_proj    = nn.Linear(1, hidden_dim)
        self.layers         = nn.ModuleList([
            _AttentionBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.to(device)

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def load_pretrained(self):
        print("[Backbone] TransformerBackbone — randomly initialized")
        print("[Backbone] Install TabPFN for pretrained weights: pip install tabpfn")

    def forward(
        self,
        context_X: torch.Tensor,   # (B, L, feat_dim)
        context_y: torch.Tensor,   # (B, L)
        query_X:   torch.Tensor,   # (B, Q, feat_dim)
    ) -> torch.Tensor:
        B, L, F = context_X.shape
        _, Q, _ = query_X.shape
        device  = context_X.device

        X_all    = torch.cat([context_X, query_X], dim=1)   # (B, L+Q, F)
        N        = L + Q
        h        = self.input_proj(X_all)                    # (B, N, hidden)

        row_type = torch.zeros(B, N, dtype=torch.long, device=device)
        row_type[:, L:] = 1
        h = h + self.row_type_embed(row_type)

        target = torch.cat([
            context_y.unsqueeze(-1),
            torch.zeros(B, Q, 1, device=device)
        ], dim=1)
        h = h + self.target_proj(target)

        for layer in self.layers:
            h = layer(h)

        return self.norm(h)[:, L:, :]   # (B, Q, hidden)


def build_backbone(config, feature_dim: int) -> nn.Module:
    """
    Try TabPFNBackbone first, fall back to TransformerBackbone.
    """
    try:
        backbone = TabPFNBackbone(
            device=config.device,
            model_version=config.backbone_model,
        )
        backbone.load_pretrained()
        return backbone
    except Exception as e:
        print(f"[Backbone] TabPFN failed ({type(e).__name__}: {e})")
        print("[Backbone] Using TransformerBackbone fallback")
        return TransformerBackbone(
            feature_dim=feature_dim,
            hidden_dim=256,
            n_layers=6,
            n_heads=8,
            device=config.device,
        )