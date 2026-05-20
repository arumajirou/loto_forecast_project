from __future__ import annotations

import contextlib
import os
import threading

import psutil

from loto_forecast.infra.db import get_session
from loto_forecast.infra.orm_models import ResourceSample


def _try_init_nvml():
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None


def start_resource_logger(task_id: str, interval_s: float = 1.0) -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()
    proc = psutil.Process(os.getpid())
    pynvml = _try_init_nvml()

    def loop() -> None:
        with contextlib.suppress(Exception):
            proc.cpu_percent(interval=None)

        while not stop_event.is_set():
            cpu = None
            rss_mb = None
            vms_mb = None
            try:
                cpu = proc.cpu_percent(interval=None)
                mem = proc.memory_info()
                rss_mb = float(mem.rss) / (1024 * 1024)
                vms_mb = float(mem.vms) / (1024 * 1024)
            except Exception:
                pass

            gpu_util = None
            gpu_mem_mb = None
            gpu_temp_c = None
            if pynvml is not None:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_util = float(util.gpu)
                    gpu_mem_mb = float(meminfo.used) / (1024 * 1024)
                    gpu_temp_c = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                except Exception:
                    pass

            with get_session() as s:
                s.add(
                    ResourceSample(
                        task_id=task_id,
                        cpu_percent=cpu,
                        rss_mb=rss_mb,
                        vms_mb=vms_mb,
                        gpu_util=gpu_util,
                        gpu_mem_mb=gpu_mem_mb,
                        gpu_temp_c=gpu_temp_c,
                    )
                )
                s.commit()

            stop_event.wait(max(0.2, float(interval_s)))

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    return stop_event, th
