"""Training and model configuration, validated with Pydantic."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Paths to input data files (relative to *data_dir*)."""

    data_dir: Path = Field(default=Path("data"), description="Root directory for all data files.")
    ab_seqs_csv: str = "sabdab_df_for_ml_240730.csv"
    train_pair_csv: str = "pfam_pairs/train_pairs_240730.csv"
    train_labels: str = "pfam_pairs/train_label_mat_240730.pt"
    test_labels: str = "pfam_pairs/test_label_mat_240730.pt"
    val_labels: str = "pfam_pairs/val_label_mat_240730.pt"
    train_test_labels: str = "pfam_pairs/train_test_label_mat_240730.pt"
    train_val_labels: str = "pfam_pairs/train_val_label_mat_240730.pt"

    def resolve(self, name: str) -> Path:
        """Return absolute path for a named data file."""
        return self.data_dir / getattr(self, name)


class ModelConfig(BaseModel):
    """ESM-2 backbone + token-wise projection settings."""

    esm_model_name: str = "facebook/esm2_t33_650M_UR50D"
    esm_embed_dim: int = 1280
    projection_dim: int = Field(
        default=256,
        description="Output dimension D' for the token-wise linear projection.",
    )
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.3
    lora_target_modules: list[str] = Field(
        default_factory=lambda: ["query", "key"],
    )


class TrainConfig(BaseModel):
    """Hyper-parameters for the training loop."""

    run_name: str = Field(
        default="default",
        description="Name for this training run. Used as the output sub-directory.",
    )
    batch_size: int = 16
    eval_batch_size: int = 64
    num_epochs: int = 500
    num_pairs_per_ab_per_epoch: int = 10
    loss_type: str = Field(
        default="siamese_mse",
        description="Training loss type: 'siamese_mse' or 'triplet'.",
    )
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    triplet_margin: float = 1.0
    use_scheduler: bool = False
    scheduler_type: str = Field(
        default="onecycle",
        description="LR scheduler type: 'onecycle' or 'cosine'.",
    )
    scheduler_max_lr: float = 5e-4
    early_stopping_patience: int = Field(
        default=0,
        description="Stop after this many epochs without val loss improvement. 0 = disabled.",
    )
    ag_thresh: float = 0.2
    ep_thresh: float = 0.5
    checkpoint_every: int = 20
    fix_mislabelled_pairs: bool = False
    output_dir: Path = Field(default=Path("outputs"))


class Config(BaseModel):
    """Top-level configuration combining data, model, and training."""

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> Config:
        """Load configuration from a JSON file, falling back to defaults for missing fields."""
        with open(path) as f:
            return cls.model_validate(json.load(f))
