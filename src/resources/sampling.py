from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .utils import now_utc


class ResourceSampler:
    def __init__(self, run_ctx, interval_sec: float, buffer_size: int) -> None:
        self.run_ctx = run_ctx
        self.interval_sec = interval_sec
        self.buffer_size = buffer_size
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._buf: list[dict] = []

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_sec * 2.0))
        self._flush()

    def _snapshot_parallel(self):
        out = {}
        with ThreadPoolExecutor(max_workers=self.run_ctx.cfg.parallel_snapshot_workers) as ex:
            fut_map = {ex.submit(c.snapshot): c for c in self.run_ctx.collectors}
            for fut in as_completed(fut_map):
                c = fut_map[fut]
                try:
                    key = getattr(c, "name", c.__class__.__name__.lower())
                    out[key] = fut.result()
                except Exception:
                    continue
        return out

    def _loop(self) -> None:
        while not self._stop.is_set():
            sampled_at = now_utc()
            snaps = self._snapshot_parallel()
            for c in self.run_ctx.collectors:
                key = getattr(c, "name", c.__class__.__name__.lower())
                s = snaps.get(key)
                if not s:
                    continue
                for metric_key, value, unit, scope in c.sample_metrics(s):
                    self._buf.append(
                        {
                            "run_id": self.run_ctx.run_id,
                            "span_id": self.run_ctx.current_span_id,
                            "sampled_at": sampled_at,
                            "scope": scope,
                            "metric_key": metric_key,
                            "metric_value": float(value),
                            "unit": unit,
                        }
                    )
            if len(self._buf) >= self.buffer_size:
                self._flush()
            self._stop.wait(self.interval_sec)

    def _flush(self) -> None:
        if not self._buf:
            return
        rows = self._buf
        self._buf = []
        self.run_ctx.db_writer.insert_metric_samples(rows)
