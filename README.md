# ESM_PDB1

Contrastive antibody embedding model using **ESM-2** as the protein language model backbone, fine-tuned with **LoRA** on SAbDab structural pairing data.  A re-implementation of the AbLangPDB1 training pipeline with a cleaner repo structure, Polars for data handling, and UV for dependency management.

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
    ├── data.py         # Polars-based data loading, ESM-2 tokenisation, PairDataset
    ├── model.py        # ESM2Embedder (dual-chain ESM-2 + Mixer MLP) with LoRA
    ├── loss.py         # Siamese MSE loss, self/cross comparison evaluation losses
    ├── evaluation.py   # Binary accuracy sweep, nearest-neighbour metrics, F1
    ├── label_qc.py     # Detection & correction of ~0.1 % mislabelled pairs
    └── train.py        # Training loop + CLI entry point
```

---

## Architecture

```
  Heavy chain AA ──► ESM-2 ──► [CLS] embedding (1 280-d) ─┐
                                                           ├─► concat (2 560-d) ──► Mixer MLP ──► L2 normalise ──► embedding
  Light chain AA ──► ESM-2 ──► [CLS] embedding (1 280-d) ─┘
```

- **Backbone**: ESM-2 (`facebook/esm2_t33_650M_UR50D`, 650 M params). A single ESM-2 model is shared for both heavy and light chains — each chain is processed independently.
- **Pooling**: By default the `[CLS]` token embedding is used. Alternatively, mean-pooling over residue positions (excluding special tokens) can be enabled via `use_cls: false`.
- **Mixer**: A 6-layer feedforward MLP (Linear → ReLU × 5, then Linear) mapping the concatenated 2 560-d vector back to 2 560-d. Keeps the embedding dimension constant.
- **LoRA**: Only a small subset of parameters are trained. LoRA adapters (default r=16, α=32) are applied to the `query` and `key` projections in ESM-2's attention layers, plus the even-numbered Mixer linear layers. This gives ~2 % trainable parameters.
- **Output**: The final embedding is L2-normalised, so cosine similarity = dot product.

### Training objective

Siamese contrastive learning with MSE loss.  Each training step samples a pair of antibodies and computes:

$$\mathcal{L} = \text{MSE}\bigl(\cos(\mathbf{e}_a, \mathbf{e}_b),\; y\bigr)$$

where $y$ is the structural similarity label:

| Label | Meaning |
|-------|---------|
| −1.0 | Different antigen family (PFAM) |
| 0.2 | Same antigen, different epitope |
| ≥ 0.5 | Overlapping epitope (value = rBSA overlap score) |

Pairs are balanced each epoch: equal counts of negative, same-antigen, and overlapping-epitope pairs.

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
    "checkpoint_every": 20
  },
  "model": {
    "esm_model_name": "facebook/esm2_t33_650M_UR50D",
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
| `train.batch_size` | 16 | Batch size for pair training |
| `train.learning_rate` | 1e-5 | AdamW learning rate |
| `train.checkpoint_every` | 20 | Save model checkpoint every N epochs |
| `train.fix_mislabelled_pairs` | true | Correct the ~0.1 % mislabelled pairs at load time |
| `model.esm_model_name` | `facebook/esm2_t33_650M_UR50D` | HuggingFace model ID |
| `model.use_cls` | true | `true` = CLS pooling, `false` = mean pooling |
| `model.add_mixer` | true | Whether to include the MLP projection head |
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
