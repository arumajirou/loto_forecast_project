from __future__ import annotations

import math

import torch

_PATCH_APPLIED: bool = False


def _clamp_top_k(factor: float, length: int) -> int:
    """
    Keep Autoformer top_k in a valid range for torch.topk.
    """
    if length <= 0:
        return 0
    k = int(float(factor) * math.log(length))
    return max(1, min(k, length))


def apply() -> None:
    """
    Monkeypatch neuralforecast Autoformer AutoCorrelation aggregators with safe top_k clamping.
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    try:
        from neuralforecast.models.autoformer import AutoCorrelation
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import neuralforecast.models.autoformer.AutoCorrelation. "
            "Check whether neuralforecast is installed."
        ) from e

    def time_delay_agg_training(self, values: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        top_k = _clamp_top_k(float(self.factor), int(length))
        if top_k <= 0:
            return torch.zeros_like(values, dtype=torch.float, device=values.device)

        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)

        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values
        delays_agg = torch.zeros_like(values, dtype=torch.float, device=values.device)

        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * (
                tmp_corr[:, i]
                .unsqueeze(1)
                .unsqueeze(1)
                .unsqueeze(1)
                .repeat(1, head, channel, length)
            )

        return delays_agg

    def time_delay_agg_inference(self, values: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]

        init_index = (
            torch.arange(length, device=values.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch, head, channel, 1)
        )

        top_k = _clamp_top_k(float(self.factor), int(length))
        if top_k <= 0:
            return torch.zeros_like(values, dtype=torch.float, device=values.device)

        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights = torch.topk(mean_value, top_k, dim=-1)[0]
        delay = torch.topk(mean_value, top_k, dim=-1)[1]

        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values, dtype=torch.float, device=values.device)

        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(
                1, head, channel, length
            )
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (
                tmp_corr[:, i]
                .unsqueeze(1)
                .unsqueeze(1)
                .unsqueeze(1)
                .repeat(1, head, channel, length)
            )

        return delays_agg

    def time_delay_agg_full(self, values: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]

        init_index = (
            torch.arange(length, device=values.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch, head, channel, 1)
        )

        top_k = _clamp_top_k(float(self.factor), int(length))
        if top_k <= 0:
            return torch.zeros_like(values, dtype=torch.float, device=values.device)

        weights = torch.topk(corr, top_k, dim=-1)[0]
        delay = torch.topk(corr, top_k, dim=-1)[1]

        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values, dtype=torch.float, device=values.device)

        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))

        return delays_agg

    AutoCorrelation.time_delay_agg_training = time_delay_agg_training
    AutoCorrelation.time_delay_agg_inference = time_delay_agg_inference
    AutoCorrelation.time_delay_agg_full = time_delay_agg_full
    _PATCH_APPLIED = True
