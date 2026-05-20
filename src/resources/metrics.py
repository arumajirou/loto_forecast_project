from __future__ import annotations


def default_metric_defs() -> list[dict[str, object]]:
    return [
        {
            "metric_key": "process.rss_mb",
            "scope": "process",
            "unit": "mb",
            "description": "Process RSS memory",
            "source_library": "psutil",
            "source_method": "Process.memory_info.rss",
            "recommended_interval_sec": 1,
        },
        {
            "metric_key": "process.vms_mb",
            "scope": "process",
            "unit": "mb",
            "description": "Process VMS memory",
            "source_library": "psutil",
            "source_method": "Process.memory_info.vms",
            "recommended_interval_sec": 1,
        },
        {
            "metric_key": "host.mem_used_mb",
            "scope": "host",
            "unit": "mb",
            "description": "Host used memory",
            "source_library": "psutil",
            "source_method": "virtual_memory.used",
            "recommended_interval_sec": 1,
        },
        {
            "metric_key": "host.mem_available_mb",
            "scope": "host",
            "unit": "mb",
            "description": "Host available memory",
            "source_library": "psutil",
            "source_method": "virtual_memory.available",
            "recommended_interval_sec": 1,
        },
        {
            "metric_key": "gpu.util_percent",
            "scope": "gpu",
            "unit": "%",
            "description": "GPU utilization",
            "source_library": "pynvml",
            "source_method": "nvmlDeviceGetUtilizationRates",
            "recommended_interval_sec": 1,
        },
        {
            "metric_key": "gpu.mem_used_mb",
            "scope": "gpu",
            "unit": "mb",
            "description": "GPU memory used",
            "source_library": "pynvml",
            "source_method": "nvmlDeviceGetMemoryInfo",
            "recommended_interval_sec": 1,
        },
        {
            "metric_key": "db.query_time_ms_total",
            "scope": "db",
            "unit": "ms",
            "description": "Total query elapsed time",
            "source_library": "sqlalchemy",
            "source_method": "before/after_cursor_execute",
            "recommended_interval_sec": 1,
        },
    ]
