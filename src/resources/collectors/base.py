from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Collector(ABC):
    name: str = "collector"

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def diff(self, start: dict[str, Any], end: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def sample_metrics(self, snap: dict[str, Any]) -> list[tuple[str, float, str, str]]:
        return []
