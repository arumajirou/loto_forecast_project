from __future__ import annotations

import datetime
import sys
from pathlib import Path

from loguru import logger

from ..config.settings import settings


def setup_logging(run_id: str | None = None) -> Path:
    settings.log_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = run_id or "adhoc"
    logfile = settings.log_dir / f"{stamp}_{tag}.log"

    logger.remove()
    logger.add(sys.stderr, level=settings.log_level)
    # enqueue=False avoids multiprocessing semaphore restrictions in sandboxed environments.
    logger.add(str(logfile), level=settings.log_level, rotation="20 MB", retention="14 days", enqueue=False)
    return logfile
