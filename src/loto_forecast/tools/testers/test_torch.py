from __future__ import annotations

import sys


def run_test() -> int:
    try:
        import torch
    except Exception as e:
        print(f"[PyTorch Test] import error: {e}", file=sys.stderr)
        return 2

    print(f"[PyTorch Test] torch.__version__ = {torch.__version__}")
    print(f"[PyTorch Test] torch.version.cuda = {torch.version.cuda}")
    print(f"[PyTorch Test] cuda.is_available = {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("[PyTorch Test] GPU/CUDA is NOT available.")
        return 3

    try:
        idx = 0
        print(f"[PyTorch Test] device_count = {torch.cuda.device_count()}")
        print(f"[PyTorch Test] device[0] = {torch.cuda.get_device_name(idx)}")

        a = torch.ones((1024, 1024), device=f"cuda:{idx}", dtype=torch.float32)
        b = torch.full((1024, 1024), 2.0, device=f"cuda:{idx}", dtype=torch.float32)
        c = a @ b
        s = float(c.sum().item())
        torch.cuda.synchronize(idx)

        if s <= 0:
            print("[PyTorch Test] Compute test = FAILED (unexpected sum)")
            return 1

        print("[PyTorch Test] Compute test = SUCCESS")
        return 0
    except Exception as e:
        print(f"[PyTorch Test] runtime error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run_test())
