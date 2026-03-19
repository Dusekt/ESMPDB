"""Loss functions for contrastive training (siamese MSE and triplet)."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from esm_pdb1.model import mean_pool

_mse = torch.nn.MSELoss()


def _nanmean_losses(*losses: torch.Tensor) -> torch.Tensor:
    """Average losses, ignoring any NaN values."""
    valid = [loss for loss in losses if not torch.isnan(loss)]
    if not valid:
        return torch.tensor(float("nan"))
    return torch.stack(valid).mean()


def siamese_mse_loss(
    left_tokens: torch.Tensor,
    right_tokens: torch.Tensor,
    left_mask: torch.Tensor,
    right_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Siamese MSE loss matching AbLangPDB's ``siamese_mse_loss``.

    Pools token-level embeddings, L2-normalises, computes cosine similarity
    (dot product of unit vectors), then MSE against the label values
    (-1, 0.2, or ≥0.5).

    Args:
        left_tokens: (B, N, D') token-level embeddings for the first antibody.
        right_tokens: (B, N, D') token-level embeddings for the second antibody.
        left_mask: (B, N) attention mask for the first antibody.
        right_mask: (B, N) attention mask for the second antibody.
        labels: (B,) target similarity values.

    Returns:
        Scalar MSE loss.
    """
    left_pooled = F.normalize(mean_pool(left_tokens, left_mask), p=2, dim=1)
    right_pooled = F.normalize(mean_pool(right_tokens, right_mask), p=2, dim=1)
    cosine_similarity = (left_pooled * right_pooled).sum(dim=1)
    return _mse(cosine_similarity, labels)


def triplet_loss(
    anchor_tokens: torch.Tensor,
    positive_tokens: torch.Tensor,
    negative_tokens: torch.Tensor,
    anchor_mask: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Triplet margin loss with mean pooling of token-level representations.

    Pools each (B, N, D') tensor to (B, D'), L2-normalises, then computes
    standard triplet margin loss.

    Args:
        anchor_tokens: (B, N, D') token-level anchor embeddings.
        positive_tokens: (B, N, D') token-level positive embeddings.
        negative_tokens: (B, N, D') token-level negative embeddings.
        anchor_mask: (B, N) attention mask for anchor sequences.
        positive_mask: (B, N) attention mask for positive sequences.
        negative_mask: (B, N) attention mask for negative sequences.
        margin: Triplet loss margin.

    Returns:
        Scalar loss.
    """
    anchor_pooled = F.normalize(mean_pool(anchor_tokens, anchor_mask), p=2, dim=1)
    positive_pooled = F.normalize(mean_pool(positive_tokens, positive_mask), p=2, dim=1)
    negative_pooled = F.normalize(mean_pool(negative_tokens, negative_mask), p=2, dim=1)
    return F.triplet_margin_loss(anchor_pooled, positive_pooled, negative_pooled, margin=margin)


# ---------------------------------------------------------------------------
# Evaluation losses (self-comparison and cross-comparison of embedding sets)
# ---------------------------------------------------------------------------


def _masked_loss(
    cos_sim_mat: torch.Tensor,
    label_mat: torch.Tensor,
    mask: torch.Tensor,
    loss_fn: torch.nn.Module,
) -> torch.Tensor:
    if not mask.any():
        return torch.tensor(float("nan"), device=cos_sim_mat.device)
    return loss_fn(cos_sim_mat[mask], label_mat[mask].float())


def self_comparison(
    embeddings: torch.Tensor,
    label_mat: torch.Tensor,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-class and balanced MSE losses for a single embedding set.

    Returns:
        (unbalanced_loss, neg_loss, ag_loss, ep_loss, balanced_loss)
    """
    embeddings = embeddings.to(device)
    label_mat = label_mat.to(device)

    cos_sim = embeddings @ embeddings.t()

    # Upper-triangle mask (no self/duplicate comparisons), excluding non-existent pairs (label ≈ 0)
    base_mask = torch.triu(torch.ones_like(cos_sim, dtype=torch.bool), diagonal=1)
    nonexistent = label_mat.abs() < 0.1
    valid = base_mask & ~nonexistent

    unbalanced = _masked_loss(cos_sim, label_mat, valid, loss_fn)

    neg_mask = base_mask & (label_mat < -0.9)
    neg_loss = _masked_loss(cos_sim, label_mat, neg_mask, loss_fn)

    # Same-antigen-different-epitope: 0.2 in cross-comparison, 0.4 in self-comparison
    ag_mask = base_mask & (label_mat > 0.1) & (label_mat < 0.45)
    ag_loss = _masked_loss(cos_sim, label_mat, ag_mask, loss_fn)

    # Overlapping epitope: ≥0.5 cross-comparison, ≥1.0 self-comparison
    ep_mask = base_mask & (label_mat >= 0.45)
    ep_loss = _masked_loss(cos_sim, label_mat, ep_mask, loss_fn)

    balanced = _nanmean_losses(neg_loss, ag_loss, ep_loss)

    label_mat.cpu()
    return unbalanced, neg_loss, ag_loss, ep_loss, balanced


def cross_comparison(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    label_mat: torch.Tensor,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-class and balanced MSE losses across two embedding sets.

    Returns:
        (unbalanced_loss, neg_loss, ag_loss, ep_loss, balanced_loss)
    """
    embeddings_a = embeddings_a.to(device)
    embeddings_b = embeddings_b.to(device)
    label_mat = label_mat.to(device)

    cos_sim = embeddings_a @ embeddings_b.t()

    base_mask = torch.ones_like(cos_sim, dtype=torch.bool)
    nonexistent = label_mat.abs() < 0.1
    valid = base_mask & ~nonexistent

    unbalanced = _masked_loss(cos_sim, label_mat, valid, loss_fn)

    neg_mask = base_mask & (label_mat < -0.9)
    neg_loss = _masked_loss(cos_sim, label_mat, neg_mask, loss_fn)

    ag_mask = base_mask & (label_mat > 0.1) & (label_mat < 0.45)
    ag_loss = _masked_loss(cos_sim, label_mat, ag_mask, loss_fn)

    ep_mask = base_mask & (label_mat >= 0.45)
    ep_loss = _masked_loss(cos_sim, label_mat, ep_mask, loss_fn)

    balanced = _nanmean_losses(neg_loss, ag_loss, ep_loss)

    label_mat.cpu()
    return unbalanced, neg_loss, ag_loss, ep_loss, balanced
