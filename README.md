# ESM_PDB1

Contrastive antibody embedding model using **ESM-2** as the protein language model backbone, fine-tuned with **LoRA** on SAbDab structural pairing data.  Heavy-chain only, with a token-wise linear projection and triplet loss — producing residue-level embeddings at inference.

---

## Quick start

```bash
# Install everything (Python ≥ 3.11 required)
uv sync

# Run training with default settings (outputs to outputs/default/)
uv run esm-pdb1-train

# Run with a custom config
uv run esm-pdb1-train --config runs/my_experiment.json
```

---

## Repository layout

```
ESM_PDB1/
├── pyproject.toml                 # Project metadata, dependencies, CLI entry point
├── data/
│   ├── sabdab_df_for_ml_240730.csv          # 1 909 human antibodies + metadata
│   └── pfam_pairs/
│       ├── train_pairs_240730.csv           # Training pair indices & labels
│       ├── {test,val}_pairs_240730.csv      # Held-out pair CSVs
│       ├── {train,test,val}_label_mat_240730.pt        # Self-comparison label matrices
│       └── {train_test,train_val}_label_mat_240730.pt  # Cross-comparison label matrices
└── src/esm_pdb1/
    ├── config.py       # Pydantic configuration (DataConfig, ModelConfig, TrainConfig)
    ├── data.py         # Polars-based data loading, ESM-2 tokenisation, TripletDataset
    ├── model.py        # ESM2Embedder (H-chain ESM-2 + token-wise projection) with LoRA
    ├── loss.py         # Triplet loss with mean pooling, self/cross comparison evaluation losses
    ├── evaluation.py   # Binary accuracy sweep, nearest-neighbour metrics, F1
    ├── label_qc.py     # Detection & correction of ~0.1 % mislabelled pairs
    └── train.py        # Training loop + CLI entry point
```

---

## Architecture

```
  Heavy chain AA ──► ESM-2 + LoRA ──► (B, N, 1280) ──► Linear(1280, D') ──► (B, N, D')
                                                                                │
                                                          ┌─────── training ────┘──── inference ──────┐
                                                          │                                           │
                                                   mean(dim=1)                              raw (B, N, D')
                                                          │                              residue-level output
                                                   L2 normalise
                                                          │
                                                   Triplet Loss
```

- **Backbone**: ESM-2 (`facebook/esm2_t33_650M_UR50D`, 650 M params). Heavy chain only.
- **Token-wise projection**: A shared `Linear(D, D')` layer transforms each residue vector (default D=1280 → D'=256).
- **LoRA**: LoRA adapters (default r=16, α=32) are applied to the `query` and `key` projections in ESM-2's attention layers. The projection layer is trained directly (full gradient).
- **Output**: Residue-level tensor `(B, N, D')`. During training, mean-pooling and L2-normalisation happen inside the loss function. At inference, the full token-level tensor is the output.

### Training objective

Triplet contrastive learning.  Each training step samples an (anchor, positive, negative) triplet:

$$\mathcal{L} = \max\bigl(d(\mathbf{a}, \mathbf{p}) - d(\mathbf{a}, \mathbf{n}) + m,\; 0\bigr)$$

where $d$ is Euclidean distance on mean-pooled, L2-normalised embeddings and $m$ is the margin (default 1.0).

Triplet formation from the pair CSV:

| Pair type | Label range | Role |
|-----------|-------------|------|
| Different antigen family | −1.0 | Negative |
| Same antigen / overlapping epitope | ≥ 0.2 | Positive |

---

## Data

The dataset originates from **SAbDab** (Structural Antibody Database). 1 909 human antibodies with known antigen–antibody crystal structures are split into:

| Split | Count |
|-------|-------|
| Train | 1 517 |
| Test | 190 |
| Val | 202 |

**Pair CSVs** (`train_pairs_240730.csv` etc.) contain columns `I`, `J` (antibody indices referencing `Column1` in the antibody CSV), and `LABEL`. The `I`/`J` naming convention: `train_test_pairs` = train antibodies vs test antibodies; `test_pairs` = test vs test.

**Label matrices** (`.pt` files) are pre-computed pairwise label tensors. Self-comparison matrices (train/test/val) use a doubled scale (−2, 0.4, 1.0+) while cross-comparison matrices (train_test, train_val) use the standard scale (−1, 0.2, 0.5+). Zero entries indicate pairs excluded due to high sequence identity.

### Label QC

~0.1 % of pairs in the CSVs have overlapping PFAM families but are labelled −1. These are automatically corrected to 0.2 during training when `fix_mislabelled_pairs: true` (the default). To inspect or correct manually:

```bash
uv run python -m esm_pdb1.label_qc data/pfam_pairs/train_pairs_240730.csv corrected.csv
```

---

## Configuration

All settings are controlled via a JSON file passed with `--config`. Any field can be omitted to use the default. Example:

```json
{
  "train": {
    "run_name": "esm2_650M_lr1e5",
    "num_epochs": 200,
    "batch_size": 16,
    "learning_rate": 1e-5,
    "triplet_margin": 1.0,
    "checkpoint_every": 20
  },
  "model": {
    "esm_model_name": "facebook/esm2_t33_650M_UR50D",
    "projection_dim": 256,
    "lora_r": 16,
    "lora_alpha": 32
  }
}
```

Key fields:

| Field | Default | Description |
|-------|---------|-------------|
| `train.run_name` | `"default"` | Name for the run — output goes to `outputs/<run_name>/` |
| `train.num_epochs` | 500 | Number of training epochs |
| `train.batch_size` | 16 | Batch size for triplet training |
| `train.learning_rate` | 1e-5 | AdamW learning rate |
| `train.triplet_margin` | 1.0 | Margin for triplet loss |
| `train.checkpoint_every` | 20 | Save model checkpoint every N epochs |
| `train.fix_mislabelled_pairs` | true | Correct the ~0.1 % mislabelled pairs at load time |
| `model.esm_model_name` | `facebook/esm2_t33_650M_UR50D` | HuggingFace model ID |
| `model.projection_dim` | 256 | Output dimension D' of the token-wise linear projection |
| `model.lora_r` | 16 | LoRA rank |

---

## Outputs

Each run writes to `outputs/<run_name>/`:

```
outputs/esm2_650M_lr1e5/
├── config.json                    # Resolved config snapshot
├── eval_metrics.csv               # Per-epoch evaluation metrics (updated live)
├── eval_metrics_final.csv         # Final copy of metrics
├── model_epoch_020.pt             # Periodic checkpoints
├── model_epoch_040.pt
├── model_final.pt                 # Final model weights
└── antibody_data_epoch_020.parquet  # Antibody data + embeddings at checkpoint
```

### Evaluation metrics

Each epoch logs per-dataset (TRAIN / TEST / VAL):

- **BALANCED_LOSS** — mean of negative, same-AG, and overlapping-EP MSE losses
- **NEAREST_NEIGHBOR_ACC_AG / \_EP** — nearest-neighbour classification accuracy for antigen family / epitope
- **F1_AG** — weighted F1 for antigen family prediction via nearest neighbour
- **NORM\_BINARY\_ACC\_EP / \_AG** — binary classification accuracy (positive vs negative pairs) at the optimal cosine similarity threshold
