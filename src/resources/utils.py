from __future__ import annotations

import datetime as dt
import inspect
import os
import platform
import socket
import traceback
from typing import Any


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def to_mb(v: float) -> float:
    return float(v) / (1024.0 * 1024.0)


def host_name() -> str:
    return socket.gethostname()


def summarize_exception(exc: BaseException, limit: int = 400) -> tuple[str, str]:
    et = type(exc).__name__
    msg = str(exc)
    if len(msg) > limit:
        msg = msg[: limit - 3] + "..."
    return et, msg


def stack_summary(limit: int = 800) -> str:
    text = traceback.format_exc()
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def safe_ident(value: str) -> str:
    if not value.replace("_", "").isalnum():
        raise ValueError(f"unsafe identifier: {value}")
    return value


def resolve_function_identity(func) -> tuple[str, str]:
    fqn = f"{func.__module__}.{func.__qualname__}"
    path = inspect.getsourcefile(func) or ""
    return fqn, path


def _read_text_if_exists(path: str, limit: int = 4000) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read(limit)
    except Exception:
        return ""


def detect_execution_environment() -> dict[str, Any]:
    platform_system = str(platform.system() or "").lower()
    platform_release = str(platform.release() or "")
    platform_version = str(platform.version() or "")
    platform_name = str(platform.platform() or "")

    has_wsl_env = bool(os.getenv("WSL_DISTRO_NAME") or os.getenv("WSL_INTEROP"))
    has_wsl_kernel_hint = False
    if platform_system == "linux":
        for probe_path in ("/proc/sys/kernel/osrelease", "/proc/version"):
            text = _read_text_if_exists(probe_path).lower()
            if "microsoft" in text or "wsl" in text:
                has_wsl_kernel_hint = True
                break

    is_wsl = bool(platform_system == "linux" and (has_wsl_env or has_wsl_kernel_hint))
    if is_wsl:
        execution_os = "wsl"
    elif platform_system == "linux":
        execution_os = "native_linux"
    elif platform_system == "windows":
        execution_os = "windows"
    else:
        execution_os = platform_system or "unknown"

    host_os_hint = "windows" if is_wsl else execution_os
    return {
        "execution_os": execution_os,
        "host_os_hint": host_os_hint,
        "is_wsl": is_wsl,
        "platform_system": platform_system,
        "platform_release": platform_release,
        "platform_version": platform_version,
        "platform_name": platform_name,
    }
