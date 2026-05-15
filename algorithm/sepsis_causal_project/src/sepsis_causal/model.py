from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import NUM_ACTION_COMBOS


SUPPORTED_MODEL_TYPES = ("transformer", "lstm", "gru", "cnn", "mlp")


def action_to_combo_index(actions: torch.Tensor) -> torch.Tensor:
    return actions[..., 0] * 9 + actions[..., 1] * 3 + actions[..., 2]


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        if t > self.pe.size(1):
            raise ValueError(f"Sequence length {t} exceeds positional encoding max_len={self.pe.size(1)}")
        return x + self.pe[:, :t]


class _OutputHeads(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.abx_head = nn.Linear(hidden_size, 2)
        self.fluid_head = nn.Linear(hidden_size, 3)
        self.vaso_head = nn.Linear(hidden_size, 3)
        self.base_outcome = nn.Linear(hidden_size, 1)
        self.combo_embeddings = nn.Parameter(torch.randn(NUM_ACTION_COMBOS, hidden_size) * 0.02)
        self.sepsis_head = nn.Linear(hidden_size, 1)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        abx_logits = self.abx_head(h)
        fluid_logits = self.fluid_head(h)
        vaso_logits = self.vaso_head(h)

        base = self.base_outcome(h).squeeze(-1)
        combo_logits = torch.einsum("bth,ch->btc", h, self.combo_embeddings)
        outcome_logits_all = base.unsqueeze(-1) + combo_logits
        outcome_prob_all = torch.sigmoid(outcome_logits_all)
        sepsis_logits = self.sepsis_head(h).squeeze(-1)
        sepsis_prob = torch.sigmoid(sepsis_logits)

        return {
            "h": h,
            "abx_logits": abx_logits,
            "fluid_logits": fluid_logits,
            "vaso_logits": vaso_logits,
            "outcome_logits_all": outcome_logits_all,
            "outcome_prob_all": outcome_prob_all,
            "sepsis_logits": sepsis_logits,
            "sepsis_prob": sepsis_prob,
        }


class CausalTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.pos_enc = SinusoidalPositionalEncoding(hidden_size, max_len=1024)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.heads = _OutputHeads(hidden_size=hidden_size)

    @staticmethod
    def _causal_mask(t: int, device: torch.device) -> torch.Tensor:
        m = torch.ones((t, t), dtype=torch.bool, device=device)
        return torch.triu(m, diagonal=1)

    @staticmethod
    def action_to_combo_index(actions: torch.Tensor) -> torch.Tensor:
        return action_to_combo_index(actions)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.input_proj(x)
        h = self.pos_enc(h)
        t = h.size(1)
        causal_mask = self._causal_mask(t, h.device)
        key_padding_mask = ~valid_mask
        h = self.encoder(h, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        return self.heads(h)


class CausalLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.dropout = nn.Dropout(dropout)
        self.heads = _OutputHeads(hidden_size=hidden_size)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        del valid_mask
        h = self.input_proj(x)
        h, _ = self.rnn(h)
        h = self.dropout(h)
        return self.heads(h)


class CausalGRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.dropout = nn.Dropout(dropout)
        self.heads = _OutputHeads(hidden_size=hidden_size)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        del valid_mask
        h = self.input_proj(x)
        h, _ = self.rnn(h)
        h = self.dropout(h)
        return self.heads(h)


class _CausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.padding > 0:
            y = y[..., :-self.padding]
        return y


class CausalCNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if kernel_size < 2:
            raise ValueError(f"model.cnn_kernel_size must be >=2, got: {kernel_size}")
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.blocks = nn.ModuleList(
            [_CausalConv1d(hidden_size, kernel_size=kernel_size, dilation=1) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.heads = _OutputHeads(hidden_size=hidden_size)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        del valid_mask
        h = self.input_proj(x)
        h = h.transpose(1, 2)
        for block in self.blocks:
            residual = h
            h = block(h)
            h = F.gelu(h)
            h = self.dropout(h)
            h = h + residual
        h = h.transpose(1, 2)
        return self.heads(h)


class _ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_size: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc1(x)
        y = F.gelu(y)
        y = self.fc2(y)
        y = self.dropout(y)
        return x + y


class CausalMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.blocks = nn.ModuleList(
            [_ResidualMLPBlock(hidden_size=hidden_size, ff_dim=ff_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.heads = _OutputHeads(hidden_size=hidden_size)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        del valid_mask
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.heads(h)


def normalize_model_config(input_dim: int, model_cfg: dict[str, Any]) -> dict[str, Any]:
    model_type = str(model_cfg.get("model_type", "transformer")).lower()
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"model.model_type must be one of {list(SUPPORTED_MODEL_TYPES)}, got: {model_type}"
        )

    hidden_size = int(model_cfg["hidden_size"])
    num_layers = int(model_cfg.get("num_layers", 2))
    num_heads = int(model_cfg.get("num_heads", 8))
    ff_dim = int(model_cfg.get("ff_dim", hidden_size * 2))
    dropout = float(model_cfg.get("dropout", 0.1))
    cnn_kernel_size = int(model_cfg.get("cnn_kernel_size", 3))

    return {
        "model_type": model_type,
        "input_dim": int(input_dim),
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "ff_dim": ff_dim,
        "dropout": dropout,
        "cnn_kernel_size": cnn_kernel_size,
    }


def build_model_from_config(input_dim: int, model_cfg: dict[str, Any]) -> nn.Module:
    cfg = normalize_model_config(input_dim=input_dim, model_cfg=model_cfg)
    model_type = cfg["model_type"]

    if model_type == "transformer":
        if cfg["hidden_size"] % cfg["num_heads"] != 0:
            raise ValueError(
                "model.hidden_size must be divisible by model.num_heads for transformer: "
                f"{cfg['hidden_size']} vs {cfg['num_heads']}"
            )
        return CausalTransformer(
            input_dim=cfg["input_dim"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            ff_dim=cfg["ff_dim"],
            dropout=cfg["dropout"],
        )
    if model_type == "lstm":
        return CausalLSTM(
            input_dim=cfg["input_dim"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )
    if model_type == "gru":
        return CausalGRU(
            input_dim=cfg["input_dim"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )
    if model_type == "cnn":
        return CausalCNN(
            input_dim=cfg["input_dim"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            kernel_size=cfg["cnn_kernel_size"],
        )
    if model_type == "mlp":
        return CausalMLP(
            input_dim=cfg["input_dim"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            ff_dim=cfg["ff_dim"],
            dropout=cfg["dropout"],
        )

    raise ValueError(f"Unsupported model type: {model_type}")


def build_model_from_checkpoint(ckpt: dict[str, Any]) -> nn.Module:
    mcfg = dict(ckpt["model_config"])
    model = build_model_from_config(input_dim=int(mcfg["input_dim"]), model_cfg=mcfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    return model


def _masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    return F.cross_entropy(logits[mask], target[mask])


def _balance_loss(h: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() == 0:
        return torch.tensor(0.0, device=h.device)
    overall = h[mask].mean(dim=0)
    total = 0.0
    groups = 0
    cat_sizes = [2, 3, 3]
    for dim, ncat in enumerate(cat_sizes):
        ad = actions[..., dim]
        for c in range(ncat):
            gmask = mask & (ad == c)
            if gmask.sum() == 0:
                continue
            gmean = h[gmask].mean(dim=0)
            total = total + F.mse_loss(gmean, overall)
            groups += 1
    if groups == 0:
        return torch.tensor(0.0, device=h.device)
    return total / groups


def _smoothness_loss(prob: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pair_mask = mask[:, 1:] & mask[:, :-1]
    if pair_mask.sum() == 0:
        return torch.tensor(0.0, device=prob.device)
    p1 = prob[:, 1:][pair_mask]
    p0 = prob[:, :-1][pair_mask]
    return F.mse_loss(p1, p0)


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    lambda_propensity: float,
    lambda_balance: float,
    lambda_smooth: float,
    lambda_sepsis: float = 1.0,
    sepsis_pos_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    mask = batch["mask"]
    y = batch["y"]
    sepsis_label = batch.get("sepsis_target", batch.get("sepsis_label"))
    actions = batch["actions"]

    combo_idx = action_to_combo_index(actions)
    factual_logits = outputs["outcome_logits_all"].gather(-1, combo_idx.unsqueeze(-1)).squeeze(-1)
    factual_prob = torch.sigmoid(factual_logits)

    if mask.sum() == 0:
        outcome_loss = torch.tensor(0.0, device=factual_logits.device)
    else:
        outcome_loss = F.binary_cross_entropy_with_logits(factual_logits[mask], y[mask])

    prop_loss = (
        _masked_cross_entropy(outputs["abx_logits"], actions[..., 0], mask)
        + _masked_cross_entropy(outputs["fluid_logits"], actions[..., 1], mask)
        + _masked_cross_entropy(outputs["vaso_logits"], actions[..., 2], mask)
    )
    balance_loss = _balance_loss(outputs["h"], actions, mask)
    smooth_loss = _smoothness_loss(factual_prob, mask)
    if sepsis_label is None or mask.sum() == 0:
        sepsis_loss = torch.tensor(0.0, device=factual_logits.device)
    else:
        sepsis_logits = outputs["sepsis_logits"][mask]
        sepsis_target = sepsis_label[mask]
        pos_weight = torch.tensor(float(sepsis_pos_weight), device=sepsis_logits.device)
        sepsis_loss = F.binary_cross_entropy_with_logits(
            sepsis_logits,
            sepsis_target,
            pos_weight=pos_weight,
        )

    total = (
        outcome_loss
        + lambda_propensity * prop_loss
        + lambda_balance * balance_loss
        + lambda_smooth * smooth_loss
        + lambda_sepsis * sepsis_loss
    )

    return {
        "total": total,
        "outcome": outcome_loss,
        "propensity": prop_loss,
        "balance": balance_loss,
        "smooth": smooth_loss,
        "sepsis": sepsis_loss,
        "factual_prob": factual_prob,
        "factual_logits": factual_logits,
        "sepsis_prob": outputs["sepsis_prob"],
        "sepsis_logits": outputs["sepsis_logits"],
    }
