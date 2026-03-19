"""Evaluation utilities: binary accuracy, nearest-neighbour metrics, and threshold search."""

from __future__ import annotations

from typing import Tuple

import polars as pl
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Binary accuracy + threshold sweep
# ---------------------------------------------------------------------------


def accuracy_and_threshold(
    pos_cos: torch.Tensor,
    neg_cos: torch.Tensor,
    fixed_thresh: float | None = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float, float, float]:
    """Sweep thresholds and return (norm_acc, norm_thresh, total_acc, total_thresh).

    If *fixed_thresh* is provided, only that threshold is evaluated and both
    norm/total use the same threshold.
    """
    if fixed_thresh is not None:
        tp = (pos_cos > fixed_thresh).float().sum()
        tn = (neg_cos < fixed_thresh).float().sum()
        norm_acc = ((tp / max(len(pos_cos), 1)) + (tn / max(len(neg_cos), 1))) / 2.0
        total_acc = (tp + tn) / max(len(pos_cos) + len(neg_cos), 1)
        return norm_acc.item(), fixed_thresh, total_acc.item(), fixed_thresh

    thresholds = torch.linspace(-1, 1, steps=200, device=device)
    tp = (pos_cos.unsqueeze(1).to(device) > thresholds).float().sum(dim=0)
    tn = (neg_cos.unsqueeze(1).to(device) < thresholds).float().sum(dim=0)

    norm_accs = (tp / max(len(pos_cos), 1) + tn / max(len(neg_cos), 1)) / 2.0
    total_accs = (tp + tn) / max(len(pos_cos) + len(neg_cos), 1)

    norm_best, norm_idx = norm_accs.max(0)
    total_best, total_idx = total_accs.max(0)

    return (
        norm_best.item(),
        thresholds[norm_idx].item(),
        total_best.item(),
        thresholds[total_idx].item(),
    )


def binary_comparison_accuracy(
    embeddings_eval: torch.Tensor,
    embeddings_ref: torch.Tensor,
    label_mat: torch.Tensor,
    label_cutoff: float,
    fixed_thresh: float | None = None,
    device: torch.device = torch.device("cpu"),
    is_self: bool = False,
) -> Tuple[float, float, float, float]:
    """Compute binary classification accuracy separating positive/negative pairs.

    Returns:
        (norm_acc, norm_thresh, total_acc, total_thresh)
    """
    if is_self:
        cos_sim = embeddings_eval @ embeddings_eval.t()
    else:
        cos_sim = embeddings_ref @ embeddings_eval.t()

    upper = (
        torch.triu(torch.ones_like(cos_sim, dtype=torch.bool), diagonal=1)
        if is_self
        else torch.ones_like(cos_sim, dtype=torch.bool)
    )

    pos_mask = upper & (label_mat >= label_cutoff)
    neg_mask = upper & (label_mat < label_cutoff)

    pos_cos = cos_sim[pos_mask]
    neg_cos = cos_sim[neg_mask]

    return accuracy_and_threshold(pos_cos, neg_cos, fixed_thresh, device)


# ---------------------------------------------------------------------------
# Nearest-neighbour evaluation
# ---------------------------------------------------------------------------


def _get_ranks(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """For each column, return the rank (1-based, descending) of the row at *indices*."""
    sorted_indices = tensor.argsort(dim=0, descending=True)
    ranks = torch.zeros_like(tensor, dtype=torch.long)
    arange = (
        torch.arange(1, tensor.shape[0] + 1, device=tensor.device).unsqueeze(1).expand_as(tensor)
    )
    ranks.scatter_(0, sorted_indices, arange)
    return ranks[indices, torch.arange(tensor.shape[1], device=tensor.device)]


def nearest_neighbour_metrics(
    df: pl.DataFrame,
    ref_dataset: str,
    eval_dataset: str,
    label_mat: torch.Tensor,
    ag_thresh: float,
    ep_thresh: float,
    ag_cos_cutoff: float,
    ep_cos_cutoff: float,
) -> Tuple[pl.DataFrame, float, float, float, float, float]:
    """Nearest-neighbour classification and ranking metrics.

    Returns:
        (updated_df, avg_ag_acc, avg_ep_acc, avg_actual_best_pred_rank,
         avg_pred_best_act_rank, ag_f1)
    """
    df_ref = df.filter(pl.col("DATASET") == ref_dataset)
    df_eval = df.filter(pl.col("DATASET") == eval_dataset)

    # Embeddings
    ref_embeds = torch.tensor(df_ref["EMBEDDING"].to_list())
    eval_embeds = torch.tensor(df_eval["EMBEDDING"].to_list())

    is_self = ref_dataset == eval_dataset
    if is_self:
        cos_sim = ref_embeds @ ref_embeds.t()
        cos_sim.fill_diagonal_(-2.0)
        label_mat = label_mat.clone()
        label_mat.fill_diagonal_(-2.0)
    else:
        cos_sim = ref_embeds @ eval_embeds.t()

    # Blank out non-existent pairs
    zeros = label_mat == 0
    cos_sim[zeros] = -2.0
    label_mat[zeros] = -2.0

    n_eval = len(df_eval)
    ref_pfams = df_ref["PFAM_PLUS"].to_list()

    # Best actual pair (by label)
    actual_max_vals, actual_max_idx = label_mat.max(dim=0)
    # Best predicted pair (by cosine similarity)
    pred_max_vals, pred_max_idx = cos_sim.max(dim=0)

    # Ranks
    actual_best_pred_rank = _get_ranks(cos_sim, actual_max_idx)
    pred_best_act_rank = _get_ranks(label_mat, pred_max_idx)

    same_ag_possible = [label_mat[:, i].max().item() >= ag_thresh for i in range(n_eval)]
    same_ep_possible = [label_mat[:, i].max().item() >= ep_thresh for i in range(n_eval)]

    nn_guess_label = label_mat[pred_max_idx, torch.arange(n_eval)]
    correct_ag = nn_guess_label >= ag_thresh
    correct_ep = nn_guess_label >= ep_thresh

    # Handle cases where same-ag/ep isn't possible
    for i in range(n_eval):
        if not same_ag_possible[i]:
            correct_ag[i] = pred_max_vals[i].item() < ag_cos_cutoff
        if not same_ep_possible[i]:
            correct_ep[i] = pred_max_vals[i].item() < ep_cos_cutoff

    avg_ag_acc = correct_ag.float().mean().item()
    avg_ep_acc = correct_ep.float().mean().item()

    ep_possible_mask = torch.tensor(same_ep_possible)
    avg_rank1 = (
        actual_best_pred_rank[ep_possible_mask].float().mean().item()
        if ep_possible_mask.any()
        else 0.0
    )
    avg_rank2 = (
        pred_best_act_rank[ep_possible_mask].float().mean().item()
        if ep_possible_mask.any()
        else 0.0
    )

    # F1 score on antigen classification
    nn_guess_pfam = [ref_pfams[idx] for idx in pred_max_idx.tolist()]
    eval_pfams = df_eval["PFAM_PLUS"].to_list()

    all_pfam_labels = set()
    for p in ref_pfams + eval_pfams:
        all_pfam_labels.update(p.split(";"))
    le = LabelEncoder()
    le.fit(list(all_pfam_labels))

    ag_possible_indices = [i for i, v in enumerate(same_ag_possible) if v]
    if ag_possible_indices:
        y_true_raw = [eval_pfams[i] for i in ag_possible_indices]
        y_pred_raw = [nn_guess_pfam[i] for i in ag_possible_indices]

        y_true_final, y_pred_final = [], []
        for yt, yp in zip(y_true_raw, y_pred_raw):
            true_set = set(yt.split(";"))
            pred_set = set(yp.split(";"))
            overlap = true_set & pred_set
            if overlap:
                chosen = list(overlap)[0]
                y_true_final.append(chosen)
                y_pred_final.append(chosen)
            else:
                y_true_final.append(list(true_set)[0])
                y_pred_final.append(list(pred_set)[0])

        y_true_enc = le.transform(y_true_final)
        y_pred_enc = le.transform(y_pred_final)
        ag_f1 = f1_score(y_true_enc, y_pred_enc, average="weighted")
    else:
        ag_f1 = 0.0

    # Build enriched eval DataFrame
    df_eval = df_eval.with_columns(
        pl.Series("CORRECT_NN_PFAM_GUESS", correct_ag.tolist()),
        pl.Series("CORRECT_NN_EP_GUESS", correct_ep.tolist()),
        pl.Series("NN_GUESS_PFAM", nn_guess_pfam),
        pl.Series("NN_ACTUAL_PREDICTED_RANK", actual_best_pred_rank.tolist()),
        pl.Series("NN_GUESS_RANK", pred_best_act_rank.tolist()),
    )

    # Replace eval rows in main df
    df_other = df.filter(pl.col("DATASET") != eval_dataset)
    df = pl.concat([df_other, df_eval], how="diagonal_relaxed")

    return df, avg_ag_acc, avg_ep_acc, avg_rank1, avg_rank2, ag_f1
