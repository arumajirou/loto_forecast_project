from __future__ import annotations

from typing import Any

from .base import Collector
from ..utils import to_mb


class PsutilCollector(Collector):
    name = "psutil"

    def __init__(self) -> None:
        import psutil

        self._psutil = psutil
        self._proc = psutil.Process()

    def snapshot(self) -> dict[str, Any]:
        p = self._proc
        ps = self._psutil
        cpu_times = p.cpu_times()
        mi = p.memory_info()
        io = p.io_counters() if hasattr(p, "io_counters") else None
        net = ps.net_io_counters()
        vm = ps.virtual_memory()
        return {
            "cpu_user": float(getattr(cpu_times, "user", 0.0)),
            "cpu_system": float(getattr(cpu_times, "system", 0.0)),
            "rss": int(getattr(mi, "rss", 0)),
            "vms": int(getattr(mi, "vms", 0)),
            "threads": int(p.num_threads()),
            "io_read": int(getattr(io, "read_bytes", 0) if io else 0),
            "io_write": int(getattr(io, "write_bytes", 0) if io else 0),
            "net_sent": int(getattr(net, "bytes_sent", 0)),
            "net_recv": int(getattr(net, "bytes_recv", 0)),
            "host_mem_used": int(getattr(vm, "used", 0)),
            "host_mem_avail": int(getattr(vm, "available", 0)),
        }

    def diff(self, start: dict[str, Any], end: dict[str, Any]) -> dict[str, Any]:
        return {
            "cpu_user_ms": int((end["cpu_user"] - start["cpu_user"]) * 1000.0),
            "cpu_system_ms": int((end["cpu_system"] - start["cpu_system"]) * 1000.0),
            "rss_start_mb": to_mb(start["rss"]),
            "rss_end_mb": to_mb(end["rss"]),
            "rss_peak_mb": max(to_mb(start["rss"]), to_mb(end["rss"])),
            "io_read_bytes_delta": int(end["io_read"] - start["io_read"]),
            "io_write_bytes_delta": int(end["io_write"] - start["io_write"]),
            "net_sent_bytes_delta": int(end["net_sent"] - start["net_sent"]),
            "net_recv_bytes_delta": int(end["net_recv"] - start["net_recv"]),
        }

    def sample_metrics(self, snap: dict[str, Any]) -> list[tuple[str, float, str, str]]:
        return [
            ("process.rss_mb", to_mb(snap["rss"]), "mb", "process"),
            ("process.vms_mb", to_mb(snap["vms"]), "mb", "process"),
            ("host.mem_used_mb", to_mb(snap["host_mem_used"]), "mb", "host"),
            ("host.mem_available_mb", to_mb(snap["host_mem_avail"]), "mb", "host"),
        ]
