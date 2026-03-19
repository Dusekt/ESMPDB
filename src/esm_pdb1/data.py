"""Data loading, tokenisation, and triplet dataset construction using Polars."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoTokenizer

from esm_pdb1.config import DataConfig

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

_TOKENIZER_CACHE: dict[str, AutoTokenizer] = {}


def _get_tokenizer(model_name: str) -> AutoTokenizer:
    if model_name not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(model_name)
    return _TOKENIZER_CACHE[model_name]


def tokenise_sequences(
    sequences: list[str],
    model_name: str,
) -> dict[str, torch.Tensor]:
    """Tokenise a list of amino-acid sequences with the ESM-2 tokenizer.

    Returns:
        Dictionary with ``input_ids`` and ``attention_mask`` tensors.
    """
    tokenizer = _get_tokenizer(model_name)
    encoded = tokenizer(
        sequences,
        add_special_tokens=True,
        padding="longest",
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


# ---------------------------------------------------------------------------
# Antibody DataFrame (Polars)
# ---------------------------------------------------------------------------


def load_antibody_data(
    cfg: DataConfig,
    esm_model_name: str,
) -> pl.DataFrame:
    """Load the per-antibody CSV, tokenise H chains, and return a Polars DataFrame.

    The returned DataFrame carries the original metadata columns plus:
    ``H_INPUT_IDS``, ``H_ATTENTION_MASK`` (each as a list-of-int column).
    """
    csv_path = cfg.resolve("ab_seqs_csv")
    df = pl.read_csv(csv_path)

    # Tokenise heavy chains
    hc_seqs = df["HC_AA"].to_list()
    h_tok = tokenise_sequences(hc_seqs, esm_model_name)

    # Attach token columns (stored as list[int] for Polars compatibility)
    df = df.with_columns(
        pl.Series("H_INPUT_IDS", h_tok["input_ids"].tolist()),
        pl.Series("H_ATTENTION_MASK", h_tok["attention_mask"].tolist()),
    )

    return df


# ---------------------------------------------------------------------------
# Triplet Dataset
# ---------------------------------------------------------------------------


class TripletDataset(Dataset):
    """PyTorch dataset that yields (anchor, positive, negative) H-chain triplets.

    Each item is a tuple of:
        (h_ids_a, h_mask_a, h_ids_p, h_mask_p, h_ids_n, h_mask_n)
    """

    def __init__(
        self,
        h_ids: torch.Tensor,
        h_mask: torch.Tensor,
        triplets: torch.Tensor,
    ) -> None:
        self.h_ids = h_ids
        self.h_mask = h_mask
        self.triplets = triplets  # (N, 3) — [anchor, positive, negative]

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        a, p, n = self.triplets[idx]
        return (
            self.h_ids[a],
            self.h_mask[a],
            self.h_ids[p],
            self.h_mask[p],
            self.h_ids[n],
            self.h_mask[n],
        )


def build_triplet_dataloader(
    df: pl.DataFrame,
    pair_csv_path: Path,
    dataset_name: str,
    batch_size: int,
    num_triplets_per_ab: int,
    fix_labels: bool = False,
) -> DataLoader:
    """Build a triplet DataLoader for one epoch of training.

    Triplets are formed from the pair CSV: positive pairs have label >= 0.2,
    negative pairs have label == -1.  For each anchor that has both positive
    and negative partners, ``num_triplets_per_ab`` triplets are sampled.
    """
    cur_df = df.filter(pl.col("DATASET") == dataset_name)
    column1_vals = cur_df["Column1"].to_list()
    index_mapping = {v: i for i, v in enumerate(column1_vals)}

    # Read pairs CSV, optionally fixing mislabelled pairs
    if fix_labels:
        from esm_pdb1.label_qc import fix_mislabelled_pairs

        pairs_df = fix_mislabelled_pairs(pair_csv_path)
    else:
        pairs_df = pl.read_csv(pair_csv_path)
    pairs_df = pairs_df.select(["I", "J", "LABEL"])
    valid_indices = set(index_mapping.keys())
    pairs_df = pairs_df.filter(pl.col("I").is_in(valid_indices) & pl.col("J").is_in(valid_indices))

    # Build positive/negative adjacency lists (0-based indices into cur_df)
    pos_partners: dict[int, list[int]] = {}
    neg_partners: dict[int, list[int]] = {}

    for row in pairs_df.iter_rows(named=True):
        i_mapped = index_mapping[row["I"]]
        j_mapped = index_mapping[row["J"]]
        label = row["LABEL"]

        if label >= 0.2:
            pos_partners.setdefault(i_mapped, []).append(j_mapped)
            pos_partners.setdefault(j_mapped, []).append(i_mapped)
        elif label == -1.0:
            neg_partners.setdefault(i_mapped, []).append(j_mapped)
            neg_partners.setdefault(j_mapped, []).append(i_mapped)

    # Anchors that have both positive and negative partners
    anchors = [k for k in pos_partners if k in neg_partners]
    target_n = len(cur_df) * num_triplets_per_ab

    triplets: list[list[int]] = []
    while len(triplets) < target_n and anchors:
        random.shuffle(anchors)
        for anchor in anchors:
            if len(triplets) >= target_n:
                break
            pos = random.choice(pos_partners[anchor])
            neg = random.choice(neg_partners[anchor])
            triplets.append([anchor, pos, neg])

    triplet_tensor = torch.tensor(triplets, dtype=torch.long)

    h_ids = torch.tensor(cur_df["H_INPUT_IDS"].to_list(), dtype=torch.long)
    h_mask = torch.tensor(cur_df["H_ATTENTION_MASK"].to_list(), dtype=torch.bool)

    ds = TripletDataset(h_ids, h_mask, triplet_tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def build_eval_dataloader(
    df: pl.DataFrame,
    batch_size: int,
) -> DataLoader:
    """Build a DataLoader over *all* antibodies for evaluation embedding."""
    ds = TensorDataset(
        torch.tensor(df["H_INPUT_IDS"].to_list(), dtype=torch.long),
        torch.tensor(df["H_ATTENTION_MASK"].to_list(), dtype=torch.bool),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def embed_dataloader(
    dataloader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
) -> list[list[float]]:
    """Run the model in eval mode, pool and L2-normalise to get per-antibody embeddings."""
    from esm_pdb1.model import mean_pool

    model.to(device)
    model.eval()
    all_embeds: list[list[float]] = []
    for batch in dataloader:
        h_ids, h_mask = batch[0].to(device), batch[1].to(device)
        token_embeds = model(h_input_ids=h_ids, h_attention_mask=h_mask)
        pooled = mean_pool(token_embeds, h_mask)
        normed = F.normalize(pooled, p=2, dim=1)
        all_embeds.extend(normed.cpu().tolist())
    return all_embeds
