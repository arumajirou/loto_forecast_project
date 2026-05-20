from __future__ import annotations

import sys


def run_test() -> int:
    try:
        import jax
        import jax.numpy as jnp
        import jaxlib
    except Exception as e:
        print(f"[JAX Test] import error: {e}", file=sys.stderr)
        return 2

    print(f"[JAX Test] jax.__version__ = {jax.__version__}")
    print(f"[JAX Test] jaxlib.__version__ = {jaxlib.__version__}")
    devices = jax.devices()
    print(f"[JAX Test] devices = {[str(d) for d in devices]}")

    has_gpu = any(d.platform == "gpu" for d in devices)
    print(f"[JAX Test] has_gpu = {has_gpu}")
    if not has_gpu:
        print("[JAX Test] GPU is NOT visible to JAX.")
        return 3

    try:
        x = jnp.ones((1_000_000,), dtype=jnp.float32)
        y = jnp.full((1_000_000,), 2.0, dtype=jnp.float32)
        z = x + y
        s = float(z.sum().block_until_ready())
        if s <= 0:
            print("[JAX Test] Compute test = FAILED (unexpected sum)")
            return 1
        print("[JAX Test] Compute test = SUCCESS")
        return 0
    except Exception as e:
        print(f"[JAX Test] runtime error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run_test())
