"""
TabPFN backbone wrapper — batched forward pass.

Critical fix: TabPFN accepts (N_total, batch_size, feat_dim) where
batch_size is the SECOND dimension. We now pass all B items at once
instead of looping over batch, giving B× speedup.

Confirmed input format from interception:
    arg[0]: (L+Q, B, feat_dim)  — ALL rows, sequence-first, batch=B
    arg[1]: (L, B)              — ONLY context labels (no query labels)
    TabPFN infers query rows by: len(X) > len(y) → last Q rows are queries

Output from hook on decoder_dict.standard.0:
    (Q, B, 384) — pre-decoder hidden states for all query rows, all batch items
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class TabPFNBackbone(nn.Module):

    PRE_DECODER_HIDDEN = 384

    def __init__(self, device: str = "cuda", model_version: str = "v2"):
        super().__init__()
        self.device        = device
        self.model_version = model_version
        self._encoder      = None
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
        self._encoder = _TabPFNEncoder(full_model).to(self.device)

        # Confirm hidden dim
        self._hidden_dim = self._probe_hidden_dim()

        n_params = sum(p.numel() for p in self._encoder.parameters())
        print(f"[Backbone] Encoder loaded — {n_params:,} parameters")
        print(f"[Backbone] Output dim: {self._hidden_dim} (pre-decoder representation)")
        print(f"[Backbone] Output head (5000-bin) discarded — replaced by RL heads")

    def _probe_hidden_dim(self) -> int:
        """Probe with B=2 batch to confirm output dim."""
        try:
            B, L, Q, F = 2, 4, 1, 4
            ctx_X = torch.zeros(B, L, F, device=self.device)
            ctx_y = torch.zeros(B, L,    device=self.device)
            qry_X = torch.zeros(B, Q, F, device=self.device)
            with torch.no_grad():
                h = self._encoder(ctx_X, ctx_y, qry_X)
            return h.shape[-1]
        except Exception as e:
            print(f"[Backbone] probe warning: {e}")
            return self.PRE_DECODER_HIDDEN

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def forward(
        self,
        context_X: torch.Tensor,   # (B, L, feat_dim)
        context_y: torch.Tensor,   # (B, L)
        query_X:   torch.Tensor,   # (B, Q, feat_dim)
    ) -> torch.Tensor:
        """
        Batched forward — passes ALL B items through TabPFN at once.
        Returns (B, Q, hidden_dim=384).
        B× faster than the previous per-item loop.
        """
        assert self._encoder is not None, "Call load_pretrained() first"
        return self._encoder(context_X, context_y, query_X)


class _TabPFNEncoder(nn.Module):
    """
    Batched TabPFN encoder.

    Key insight: TabPFN's PerFeatureTransformer processes ALL batch items
    simultaneously when you pass batch_size > 1 in the second dimension:
        X: (N_total, B, feat_dim)
        y: (L, B)
    This is B× more efficient than calling it B times with batch=1.
    """

    def __init__(self, full_model: nn.Module):
        super().__init__()
        self.model = full_model

        modules = dict(self.model.named_modules())
        assert "decoder_dict.standard.0" in modules, \
            "decoder_dict.standard.0 not found"
        assert "decoder_dict.standard.2" in modules, \
            "decoder_dict.standard.2 not found"

        # Freeze the unused 5000-bin output head
        for param in modules["decoder_dict.standard.2"].parameters():
            param.requires_grad = False

    def forward(
        self,
        context_X: torch.Tensor,   # (B, L, feat_dim)
        context_y: torch.Tensor,   # (B, L)
        query_X:   torch.Tensor,   # (B, Q, feat_dim)
    ) -> torch.Tensor:
        """
        Returns (B, Q, 384).

        TabPFN input format:
            X: (N_total, B, feat_dim)  — sequence-first, batch second
            y: (L, B)                  — context labels only (no query labels)
        TabPFN determines query rows by: N_total > L (extra rows = queries)
        """
        B, L, F = context_X.shape
        _, Q, _ = query_X.shape
        device   = context_X.device

        # ── Build batched input ───────────────────────────────────────────────
        # Concatenate context + query along sequence (row) dimension
        # context_X: (B, L, F) → permute → (L, B, F)
        # query_X:   (B, Q, F) → permute → (Q, B, F)
        ctx_perm = context_X.permute(1, 0, 2)   # (L, B, F)
        qry_perm = query_X.permute(1, 0, 2)     # (Q, B, F)

        X_in = torch.cat([ctx_perm, qry_perm], dim=0)  # (L+Q, B, F)

        # Labels: (L, B) — context only, no placeholder for query rows
        y_in = context_y.permute(1, 0)                  # (L, B)

        # ── Hook to capture pre-decoder hidden state ──────────────────────────
        captured = {}

        def post_hook(module, args, output):
            # output shape: (Q, B, 384)
            captured["h"] = output

        hook_mod = dict(self.model.named_modules())["decoder_dict.standard.0"]
        handle   = hook_mod.register_forward_hook(post_hook)

        try:
            self.model(
                X_in,
                y_in,
                only_return_standard_out=True,
                categorical_inds=[[] for _ in range(B)],   # one list per batch item
                save_peak_memory_factor=None,
            )
        finally:
            handle.remove()

        h = captured.get("h")
        if h is None:
            raise RuntimeError(
                "Post-hook on decoder_dict.standard.0 did not capture output. "
                "Check TabPFN version."
            )

        # h shape from hook: (N_total, B, 384) or (Q, B, 384)
        # decoder_dict.standard.0 may output ALL rows or just query rows
        # We always take the LAST Q rows to get query representations
        if h.dim() == 3:
            h = h[-Q:, :, :]        # (Q, B, 384) — last Q rows are queries
            h = h.permute(1, 0, 2)  # (B, Q, 384)
        elif h.dim() == 2:
            # Edge case: (N*B, 384) — reshape and take last Q
            h = h.view(-1, B, h.shape[-1])
            h = h[-Q:, :, :].permute(1, 0, 2)

        if h.shape[0] != B or h.shape[1] != Q:
            raise RuntimeError(
                f"Unexpected hidden state shape: {h.shape}, expected ({B}, {Q}, hidden)"
            )

        return h   # (B, Q, 384)


# ── Fallback backbone ─────────────────────────────────────────────────────────

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
    """Randomly initialized fallback. Same interface as TabPFNBackbone."""

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

    def forward(
        self,
        context_X: torch.Tensor,
        context_y: torch.Tensor,
        query_X:   torch.Tensor,
    ) -> torch.Tensor:
        B, L, F = context_X.shape
        _, Q, _ = query_X.shape
        device  = context_X.device

        X_all    = torch.cat([context_X, query_X], dim=1)
        N        = L + Q
        h        = self.input_proj(X_all)

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

        return self.norm(h)[:, L:, :]


def build_backbone(config, feature_dim: int) -> nn.Module:
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