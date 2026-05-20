from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import psutil
from loguru import logger

_NVIDIA_SMI_AVAILABLE: bool | None = None


@dataclass
class ResourceSample:
    ts: float
    cpu_percent: float
    process_cpu_percent: float
    system_cpu_percent: float
    mem_percent: float
    rss_mb: float
    gpu_util: float | None
    gpu_mem_mb: float | None
    gpu_name: str | None
    pid: int

    def to_dict(self) -> dict:
        return asdict(self)


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if cleaned == "" or cleaned.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _run_nvidia_smi(query: str) -> list[dict[str, Any]] | None:
    global _NVIDIA_SMI_AVAILABLE
    if _NVIDIA_SMI_AVAILABLE is False:
        return None
    cmd = ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=2.0, check=True)
    except (FileNotFoundError, subprocess.SubprocessError):
        _NVIDIA_SMI_AVAILABLE = False
        return None
    _NVIDIA_SMI_AVAILABLE = True
    rows: list[dict[str, Any]] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        rows.append({"raw": line, "parts": parts})
    return rows


def _sample_gpu(pid: int) -> tuple[float | None, float | None, str | None]:
    gpu_rows = _run_nvidia_smi("gpu=index,uuid,name,utilization.gpu,memory.used")
    if not gpu_rows:
        return None, None, None

    gpu_map: dict[str, dict[str, str]] = {}
    ordered: list[dict[str, str]] = []
    for row in gpu_rows:
        parts = row["parts"]
        if len(parts) < 5:
            continue
        item = {
            "index": parts[0],
            "uuid": parts[1],
            "name": parts[2],
            "util": parts[3],
            "mem": parts[4],
        }
        gpu_map[item["uuid"]] = item
        ordered.append(item)

    app_rows = _run_nvidia_smi("compute-apps=pid,gpu_uuid,used_gpu_memory") or []
    matched = [row["parts"] for row in app_rows if len(row["parts"]) >= 3 and row["parts"][0] == str(pid)]
    if matched:
        names: list[str] = []
        utils: list[float] = []
        mems: list[float] = []
        for _, gpu_uuid, used_mem in matched:
            gpu = gpu_map.get(gpu_uuid)
            if gpu is not None:
                if gpu["name"] and gpu["name"] not in names:
                    names.append(gpu["name"])
                util = _to_float(gpu["util"])
                if util is not None:
                    utils.append(util)
            mem = _to_float(used_mem)
            if mem is not None:
                mems.append(mem)
        return (max(utils) if utils else None, sum(mems) if mems else None, ",".join(names) if names else None)

    first = ordered[0] if ordered else None
    if first is None:
        return None, None, None
    return _to_float(first["util"]), _to_float(first["mem"]), first["name"] or None


def sample_resources(process: psutil.Process | None = None) -> ResourceSample:
    p = process or psutil.Process(os.getpid())
    mem = p.memory_info().rss / (1024**2)
    pid = p.pid
    process_cpu_percent = p.cpu_percent(interval=None)
    system_cpu_percent = psutil.cpu_percent(interval=None)
    gpu_util, gpu_mem_mb, gpu_name = _sample_gpu(pid)
    return ResourceSample(
        ts=time.time(),
        cpu_percent=system_cpu_percent,
        process_cpu_percent=process_cpu_percent,
        system_cpu_percent=system_cpu_percent,
        mem_percent=psutil.virtual_memory().percent,
        rss_mb=mem,
        gpu_util=gpu_util,
        gpu_mem_mb=gpu_mem_mb,
        gpu_name=gpu_name,
        pid=pid,
    )


class ResourceMonitor:
    """Lightweight resource monitor for long runs."""

    def __init__(self, interval_sec: float = 5.0):
        self.interval_sec = max(0.1, float(interval_sec))
        self.samples: list[ResourceSample] = []
        self._running = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = psutil.Process(os.getpid())

    def _capture_once(self) -> ResourceSample:
        s = sample_resources(self._process)
        with self._lock:
            self.samples.append(s)
        gpu_part = (
            f" gpu={s.gpu_util:.1f}% gpu_mem={s.gpu_mem_mb:.1f}MB"
            if s.gpu_util is not None or s.gpu_mem_mb is not None
            else ""
        )
        logger.info(
            f"resource proc_cpu={s.process_cpu_percent:.1f}% sys_cpu={s.system_cpu_percent:.1f}% "
            f"mem={s.mem_percent:.1f}% rss={s.rss_mb:.1f}MB{gpu_part}"
        )
        return s

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self._capture_once()
            self._stop_event.wait(self.interval_sec)

    def start(self) -> None:
        if self._running:
            return
        self.samples = []
        self._process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="resource-monitor", daemon=True)
        self._thread.start()

    def run_for(self, duration_sec: float) -> list[ResourceSample]:
        self.samples = []
        start = time.time()
        self.start()
        while self._running and (time.time() - start) < duration_sec:
            time.sleep(min(0.1, self.interval_sec))
        self.stop()
        return self.samples

    def stop(self, capture_final: bool = True) -> list[ResourceSample]:
        if not self._running:
            if capture_final and not self.samples:
                self._capture_once()
            return list(self.samples)
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_sec * 2.0))
            self._thread = None
        self._running = False
        if capture_final:
            self._capture_once()
        return list(self.samples)

    def to_dicts(self) -> list[dict]:
        with self._lock:
            return [s.to_dict() for s in self.samples]


def generate_run_id(prefix: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(prefix)).strip("_") or "run"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"{safe}_{stamp}"
