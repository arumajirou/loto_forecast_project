from __future__ import annotations

from typing import Any

from .base import Collector
from ..utils import to_mb


class NvmlCollector(Collector):
    name = "nvml"

    def __init__(self) -> None:
        import pynvml

        self._nvml = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def snapshot(self) -> dict[str, Any]:
        n = self._nvml
        util = n.nvmlDeviceGetUtilizationRates(self._handle)
        mem = n.nvmlDeviceGetMemoryInfo(self._handle)
        return {
            "gpu_util": float(getattr(util, "gpu", 0.0)),
            "gpu_mem_used": int(getattr(mem, "used", 0)),
        }

    def diff(self, start: dict[str, Any], end: dict[str, Any]) -> dict[str, Any]:
        return {
            "gpu_util_avg": float((start["gpu_util"] + end["gpu_util"]) / 2.0),
            "gpu_mem_used_mb_avg": float((to_mb(start["gpu_mem_used"]) + to_mb(end["gpu_mem_used"])) / 2.0),
        }

    def sample_metrics(self, snap: dict[str, Any]) -> list[tuple[str, float, str, str]]:
        return [
            ("gpu.util_percent", float(snap["gpu_util"]), "%", "gpu"),
            ("gpu.mem_used_mb", to_mb(snap["gpu_mem_used"]), "mb", "gpu"),
        ]
