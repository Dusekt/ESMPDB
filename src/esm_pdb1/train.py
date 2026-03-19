"""Training loop and CLI entry point."""

from __future__ import annotations

import argparse
import logging
from time import time

import polars as pl
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from esm_pdb1.config import Config
from esm_pdb1.data import (
    build_eval_dataloader,
    build_pair_dataloader,
    build_triplet_dataloader,
    embed_dataloader,
    load_antibody_data,
)
from esm_pdb1.evaluation import (
    binary_comparison_accuracy,
    nearest_neighbour_metrics,
)
from esm_pdb1.loss import cross_comparison, self_comparison, siamese_mse_loss, triplet_loss
from esm_pdb1.model import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluate one epoch
# ---------------------------------------------------------------------------


def _load_label_mats(cfg: Config) -> dict[str, torch.Tensor]:
    """Load all pre-computed label matrices once."""
    dc = cfg.data
    return {
        "train": torch.load(dc.resolve("train_labels"), weights_only=True),
        "test": torch.load(dc.resolve("test_labels"), weights_only=True),
        "val": torch.load(dc.resolve("val_labels"), weights_only=True),
        "train_test": torch.load(dc.resolve("train_test_labels"), weights_only=True),
        "train_val": torch.load(dc.resolve("train_val_labels"), weights_only=True),
    }


@torch.no_grad()
def evaluate_epoch(
    model: torch.nn.Module,
    epoch: int,
    df: pl.DataFrame,
    cfg: Config,
    label_mats: dict[str, torch.Tensor],
    device: torch.device,
    eval_metrics: list[dict],
) -> tuple[pl.DataFrame, list[dict]]:
    """Embed all antibodies, compute losses and accuracy metrics for this epoch."""
    tc = cfg.train
    loss_fn = torch.nn.MSELoss()

    # Embed everything
    eval_dl = build_eval_dataloader(df, tc.eval_batch_size)
    embeddings = embed_dataloader(eval_dl, model, device)

    # Store embeddings in DataFrame
    if "EMBEDDING" in df.columns:
        df = df.drop("EMBEDDING")
    df = df.with_columns(pl.Series("EMBEDDING", embeddings))

    # Split embeddings by dataset
    train_mask = df["DATASET"] == "TRAIN"
    test_mask = df["DATASET"] == "TEST"
    val_mask = df["DATASET"] == "VAL"

    train_embeds = torch.tensor(df.filter(train_mask)["EMBEDDING"].to_list())
    test_embeds = torch.tensor(df.filter(test_mask)["EMBEDDING"].to_list())
    val_embeds = torch.tensor(df.filter(val_mask)["EMBEDDING"].to_list())

    model.cpu()

    train_losses = self_comparison(train_embeds, label_mats["train"], loss_fn, device)
    test_losses = self_comparison(test_embeds, label_mats["test"], loss_fn, device)
    val_losses = self_comparison(val_embeds, label_mats["val"], loss_fn, device)
    train_test_losses = cross_comparison(
        train_embeds, test_embeds, label_mats["train_test"], loss_fn, device
    )
    train_val_losses = cross_comparison(
        train_embeds, val_embeds, label_mats["train_val"], loss_fn, device
    )

    for dataset_name in ("TRAIN", "TEST", "VAL"):
        row: dict = {"EPOCH": epoch, "DATASET": dataset_name}

        if dataset_name == "TRAIN":
            self_losses = train_losses
            vs_losses = train_losses  # train vs train
            lm = label_mats["train"]
            nn_lm = label_mats["train"]
        elif dataset_name == "TEST":
            self_losses = test_losses
            vs_losses = train_test_losses
            lm = label_mats["test"]
            nn_lm = label_mats["train_test"]
        else:
            self_losses = val_losses
            vs_losses = train_val_losses
            lm = label_mats["val"]
            nn_lm = label_mats["train_val"]

        loss_keys = [
            "UNBALANCED_LOSS",
            "NEGATIVE_PAIR_LOSS",
            "SAME_AG_LOSS",
            "OVERLAPPING_EPITOPE_LOSS",
            "BALANCED_LOSS",
        ]
        for k, v in zip(loss_keys, self_losses):
            row[k] = v.item()
        vs_keys = [f"{k}_VS_TEST" for k in loss_keys]
        for k, v in zip(vs_keys, vs_losses):
            row[k] = v.item()

        # Binary accuracy (self-comparison within each dataset)
        eval_embeds_cur = {"TRAIN": train_embeds, "TEST": test_embeds, "VAL": val_embeds}[
            dataset_name
        ]
        for cutoff, prefix in [(tc.ep_thresh, "EP"), (tc.ag_thresh, "AG")]:
            na, nt, ta, tt = binary_comparison_accuracy(
                eval_embeds_cur.to(device),
                eval_embeds_cur.to(device),
                lm.to(device),
                cutoff,
                device=device,
                is_self=True,
            )
            row[f"NORM_BINARY_ACC_{prefix}"] = na
            row[f"NORM_COS_SIM_THRESH_{prefix}"] = nt
            row[f"TOT_BINARY_ACC_{prefix}"] = ta
            row[f"TOT_COS_SIM_THRESH_{prefix}"] = tt

        # Nearest-neighbour metrics
        ag_cos_cutoff = row.get("TOT_COS_SIM_THRESH_AG", 0.0)
        ep_cos_cutoff = row.get("TOT_COS_SIM_THRESH_EP", 0.0)
        df, avg_ag, avg_ep, rank1, rank2, ag_f1 = nearest_neighbour_metrics(
            df,
            "TRAIN",
            dataset_name,
            nn_lm,
            tc.ag_thresh,
            tc.ep_thresh,
            ag_cos_cutoff,
            ep_cos_cutoff,
        )
        row["NEAREST_NEIGHBOR_ACC_AG"] = avg_ag
        row["NEAREST_NEIGHBOR_ACC_EP"] = avg_ep
        row["ACTUAL_BEST_PRED_RANK"] = rank1
        row["PRED_BEST_ACT_RANK"] = rank2
        row["F1_AG"] = ag_f1

        eval_metrics.append(row)

    # Log summary
    metrics_df = pl.DataFrame(eval_metrics)
    latest = {r["DATASET"]: r for r in eval_metrics if r["EPOCH"] == epoch}
    for ds in ("TRAIN", "TEST", "VAL"):
        r = latest.get(ds)
        if r:
            log.info(
                "Epoch %d  %-5s  bal_loss=%.4f  nn_ag=%.3f  nn_ep=%.3f  f1_ag=%.3f",
                epoch,
                ds,
                r["BALANCED_LOSS"],
                r["NEAREST_NEIGHBOR_ACC_AG"],
                r["NEAREST_NEIGHBOR_ACC_EP"],
                r["F1_AG"],
            )

    # Write metrics
    out_dir = tc.output_dir
    metrics_df.write_csv(out_dir / "eval_metrics.csv")

    # Checkpoint
    if epoch > 0 and epoch % tc.checkpoint_every == 0:
        model.cpu()
        torch.save(model.state_dict(), out_dir / f"model_epoch_{epoch:03d}.pt")
        df.write_parquet(out_dir / f"antibody_data_epoch_{epoch:03d}.parquet")

    return df, eval_metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def training_loop(
    model: torch.nn.Module,
    df: pl.DataFrame,
    cfg: Config,
    device: torch.device,
) -> tuple[pl.DataFrame, list[dict], list[float]]:
    """Run the full training loop.

    Returns:
        (final_df, eval_metrics, step_losses)
    """
    tc = cfg.train
    dc = cfg.data
    label_mats = _load_label_mats(cfg)

    # Optimiser
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tc.learning_rate, weight_decay=tc.weight_decay
    )
    scheduler = None
    if tc.use_scheduler:
        n_train = len(df.filter(pl.col("DATASET") == "TRAIN"))
        import math

        total_steps = math.ceil(
            tc.num_epochs * n_train * tc.num_pairs_per_ab_per_epoch / tc.batch_size
        )
        if tc.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=tc.num_epochs)
        else:
            scheduler = OneCycleLR(
                optimizer,
                max_lr=tc.scheduler_max_lr,
                total_steps=total_steps,
                pct_start=0.25,
                anneal_strategy="cos",
            )

    # Baseline evaluation
    eval_metrics: list[dict] = []
    all_losses: list[float] = []
    df, eval_metrics = evaluate_epoch(model, 0, df, cfg, label_mats, device, eval_metrics)

    pair_csv = dc.resolve("train_pair_csv")
    step = 0

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, tc.num_epochs + 1):
        t0 = time()
        model.to(device)
        model.train()

        if tc.loss_type == "triplet":
            triplet_dl = build_triplet_dataloader(
                df,
                pair_csv,
                "TRAIN",
                tc.batch_size,
                tc.num_pairs_per_ab_per_epoch,
                fix_labels=tc.fix_mislabelled_pairs,
            )

            for batch in triplet_dl:
                step += 1
                optimizer.zero_grad()

                h_a, hm_a, h_p, hm_p, h_n, hm_n = (b.to(device) for b in batch)
                tok_a = model(h_input_ids=h_a, h_attention_mask=hm_a)
                tok_p = model(h_input_ids=h_p, h_attention_mask=hm_p)
                tok_n = model(h_input_ids=h_n, h_attention_mask=hm_n)

                loss = triplet_loss(
                    tok_a,
                    tok_p,
                    tok_n,
                    hm_a,
                    hm_p,
                    hm_n,
                    margin=tc.triplet_margin,
                )
                cur_loss = loss.item()
                all_losses.append(cur_loss)

                loss.backward()
                optimizer.step()
                if scheduler and tc.scheduler_type != "cosine":
                    scheduler.step()

                if step % 50 == 0:
                    log.info(
                        "Epoch %d  step %d  loss=%.4f  (%.1fs)",
                        epoch,
                        step,
                        cur_loss,
                        time() - t0,
                    )
        else:
            # Siamese MSE (pair-based, matches AbLangPDB)
            pair_dl = build_pair_dataloader(
                df,
                pair_csv,
                "TRAIN",
                tc.batch_size,
                tc.num_pairs_per_ab_per_epoch,
                fix_labels=tc.fix_mislabelled_pairs,
            )

            for batch in pair_dl:
                step += 1
                optimizer.zero_grad()

                h1, hm1, h2, hm2, labels = (b.to(device) for b in batch)
                tok1 = model(h_input_ids=h1, h_attention_mask=hm1)
                tok2 = model(h_input_ids=h2, h_attention_mask=hm2)

                loss = siamese_mse_loss(tok1, tok2, hm1, hm2, labels)
                cur_loss = loss.item()
                all_losses.append(cur_loss)

                loss.backward()
                optimizer.step()
                if scheduler and tc.scheduler_type != "cosine":
                    scheduler.step()

                if step % 50 == 0:
                    log.info(
                        "Epoch %d  step %d  loss=%.4f  (%.1fs)",
                        epoch,
                        step,
                        cur_loss,
                        time() - t0,
                    )

        # Per-epoch scheduler step for CosineAnnealingLR
        if scheduler and tc.scheduler_type == "cosine":
            scheduler.step()

        df, eval_metrics = evaluate_epoch(model, epoch, df, cfg, label_mats, device, eval_metrics)
        model.to(device)

        # Early stopping
        if tc.early_stopping_patience > 0:
            val_row = next(
                (r for r in eval_metrics if r["EPOCH"] == epoch and r["DATASET"] == "VAL"),
                None,
            )
            if val_row is not None:
                val_loss = val_row["BALANCED_LOSS"]
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if epochs_without_improvement >= tc.early_stopping_patience:
                    log.info(
                        "Early stopping at epoch %d (no val loss improvement for %d epochs).",
                        epoch,
                        tc.early_stopping_patience,
                    )
                    break

    return df, eval_metrics, all_losses


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train ESM_PDB1 contrastive antibody model.")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file.")
    args = parser.parse_args()

    cfg = Config.from_json(args.config) if args.config else Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Output directory from run_name
    cfg.train.output_dir = cfg.train.output_dir / cfg.train.run_name
    cfg.train.output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    cfg_path = cfg.train.output_dir / "config.json"
    cfg_path.write_text(cfg.model_dump_json(indent=2))
    log.info("Run '%s' on %s — config saved to %s", cfg.train.run_name, device, cfg_path)

    # Data
    df = load_antibody_data(cfg.data, cfg.model.esm_model_name)
    log.info("Loaded %d antibodies", len(df))

    # Model
    model = build_model(cfg.model).to(device)

    # Train
    df, eval_metrics, all_losses = training_loop(model, df, cfg, device)

    # Save final artefacts
    torch.save(model.cpu().state_dict(), cfg.train.output_dir / "model_final.pt")
    pl.DataFrame(eval_metrics).write_csv(cfg.train.output_dir / "eval_metrics_final.csv")
    log.info("Training complete — artefacts in %s", cfg.train.output_dir)


if __name__ == "__main__":
    main()
