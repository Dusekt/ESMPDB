"""ESM-2 based antibody embedder with token-wise projection and LoRA.

The model processes H-chain sequences through ESM-2 and applies a shared
linear projection to every residue, producing (B, N, D') token-level
embeddings.  Mean pooling is deferred to the loss function (training) or
to the caller (inference/evaluation).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModel

from esm_pdb1.config import ModelConfig


def mean_pool(token_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool over non-padding, non-special token positions.

    Args:
        token_embeds: (B, N, D) token-level embeddings.
        attention_mask: (B, N) binary mask (1 = real token, 0 = padding).

    Returns:
        (B, D) pooled embeddings.
    """
    mask = attention_mask.float()
    # Exclude [CLS] (position 0) and [EOS] (last non-pad position)
    mask[:, 0] = 0.0
    lengths = attention_mask.sum(dim=1)
    for i, length in enumerate(lengths):
        mask[i, length - 1] = 0.0
    mask = mask.unsqueeze(-1).expand_as(token_embeds)
    summed = (token_embeds * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class ESM2Embedder(nn.Module):
    """H-chain ESM-2 embedder with token-wise linear projection.

    Processes the heavy chain through ESM-2, then applies a linear
    projection to every residue position.  Returns residue-level
    representations of shape (B, N, D').
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.esm = AutoModel.from_pretrained(cfg.esm_model_name, trust_remote_code=True)
        self.config = self.esm.config  # needed by PEFT
        self.projection = nn.Linear(cfg.esm_embed_dim, cfg.projection_dim)

    def forward(
        self,
        h_input_ids: torch.Tensor,
        h_attention_mask: torch.Tensor,
        **_kwargs,
    ) -> torch.Tensor:
        """Run the H-chain through ESM-2 and project token-wise.

        Returns:
            (B, N, D') residue-level projected embeddings.
        """
        outputs = self.esm(input_ids=h_input_ids, attention_mask=h_attention_mask)
        return self.projection(outputs.last_hidden_state)


def build_model(cfg: ModelConfig, checkpoint_path: str | None = None) -> nn.Module:
    """Construct an ESM2Embedder and apply LoRA (or load a checkpoint).

    Args:
        cfg: Model configuration.
        checkpoint_path: If provided, load weights from this file instead of
            applying fresh LoRA adapters.

    Returns:
        The model (on CPU).
    """
    model = ESM2Embedder(cfg)

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True), strict=False)
        return model

    # Fresh LoRA setup
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.lora_target_modules),
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)

    # Ensure the projection layer is fully trainable (not subject to LoRA)
    for param in model.base_model.model.projection.parameters():
        param.requires_grad = True

    model.print_trainable_parameters()
    return model
