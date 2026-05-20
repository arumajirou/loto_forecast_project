import os
import subprocess
import sys
from pathlib import Path

# 実行ディレクトリを基準にパスを解決
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TORCH_TEST_PATH = os.path.join(BASE_DIR, "testers", "test_torch.py")
JAX_TEST_PATH = os.path.join(BASE_DIR, "testers", "test_jax.py")


def _validate_tester_files() -> bool:
    missing: list[str] = []
    for p in [TORCH_TEST_PATH, JAX_TEST_PATH]:
        if not os.path.exists(p):
            missing.append(p)
    if not missing:
        return True

    print("[致命的エラー] 診断テストファイルが見つかりません。再インストールは実行しません。")
    for p in missing:
        print(f"  - missing: {p}")
    print("testers/test_torch.py と testers/test_jax.py を配置して再実行してください。")
    return False


def _site_packages_dir() -> Path:
    return Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"


def _build_jax_env() -> dict:
    """JAX が古い /usr/bin/ptxas を拾わないように実行環境を補正する。"""
    env = dict(os.environ)
    sp = _site_packages_dir()
    cuda_nvcc_root = sp / "nvidia" / "cuda_nvcc"
    ptxas_bin_dir = cuda_nvcc_root / "bin"
    ptxas = ptxas_bin_dir / "ptxas"

    if ptxas.exists():
        env["PATH"] = f"{ptxas_bin_dir}{os.pathsep}{env.get('PATH', '')}"
        print(f"[JAX Env] ptxas = {ptxas}")

    if cuda_nvcc_root.exists():
        xla_flag = f"--xla_gpu_cuda_data_dir={cuda_nvcc_root}"
        cur = str(env.get("XLA_FLAGS", "")).strip()
        if xla_flag not in cur.split():
            env["XLA_FLAGS"] = f"{xla_flag} {cur}".strip()
        print(f"[JAX Env] XLA_FLAGS = {env['XLA_FLAGS']}")

    return env


def run_isolated_command(command_list, description, extra_env: dict | None = None) -> int:
    """
    メタ認知・進捗管理: 外部コマンドを分離された環境で実行し、クラッシュを安全に捕捉する。
    """
    print(f"\n--- 実行中(Executing): {description} ---")
    try:
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        result = subprocess.run(command_list, text=True, capture_output=True, env=env)
        print(result.stdout)

        if result.returncode != 0:
            print(f"[{description} 失敗(Failed)] 戻り値(Return code): {result.returncode}")
            if result.stderr:
                print(f"標準エラー(STDERR):\n{result.stderr}")
        return int(result.returncode)

    except Exception as e:
        print(f"[{description} 致命的エラー(Fatal Error)] プロセス実行中に異常発生: {e}")
        return 99


def reinstall_libraries():
    """
    要件定義に基づく環境修復: RTX 50シリーズ対応の最新環境を再構築する。
    """
    print("\n環境の修復を開始します(Starting environment repair)...")

    # 依存関係の競合をなくすため、関連パッケージを完全にアンインストール
    # 注意: nvidia/nccl/lib は cu12/cu13 パッケージでパス衝突するため両方いったん削除する。
    uninstall_cmd = [
        sys.executable,
        "-m",
        "pip",
        "uninstall",
        "-y",
        "torch",
        "torchvision",
        "torchaudio",
        "jax",
        "jaxlib",
        "jax-cuda12-pjrt",
        "jax-cuda12-plugin",
        "jax-cuda13-pjrt",
        "jax-cuda13-plugin",
        "nvidia-nccl-cu11",
        "nvidia-nccl-cu12",
        "nvidia-nccl-cu13",
        "nvidia-cudnn-cu11",
        "nvidia-cudnn-cu12",
        "nvidia-cudnn-cu13",
    ]
    run_isolated_command(uninstall_cmd, "既存ライブラリの削除(Uninstalling old libraries)")

    # JAX を先に入れてから Torch を入れる。
    # 理由: Torch(cu130) 側の NCCL (cu13) を最後に配置し、libtorch_cuda の undefined symbol を回避する。
    install_jax_cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir", "jax[cuda12]"]
    jax_rc = run_isolated_command(install_jax_cmd, "JAX (CUDA 12/13互換版) のインストール")

    # RTX 5070 Ti (sm_120) 対応のため Torch は cu130 Nightly を利用
    install_torch_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--pre",
        "--upgrade",
        "torch",
        "torchvision",
        "torchaudio",
        "--index-url",
        "https://download.pytorch.org/whl/nightly/cu130",
    ]
    torch_rc = run_isolated_command(install_torch_cmd, "PyTorch (cu130対応最新版) のインストール")

    return (torch_rc == 0) and (jax_rc == 0)


def main():
    if not _validate_tester_files():
        return

    print("初回テストを実行します(Running initial tests)...")
    jax_env = _build_jax_env()

    # サブプロセスとしてテストを実行（コアダンプが発生しても管理プロセスは生存する）
    torch_rc = run_isolated_command([sys.executable, TORCH_TEST_PATH], "PyTorch 診断テスト")
    jax_rc = run_isolated_command([sys.executable, JAX_TEST_PATH], "JAX 診断テスト", extra_env=jax_env)

    if torch_rc == 0 and jax_rc == 0:
        print("\n全てのテストが正常に完了しました(All tests passed successfully). 環境は健全です.")
        return

    # 3: importは成功したがGPUが見えない。再インストールでは解決しにくい。
    if torch_rc == 3 or jax_rc == 3:
        print("\n[停止] GPUがライブラリから見えていません。再インストールはスキップします。")
        print("nvidia-smi, ドライバ, コンテナ/WSLのGPUパススルー, CUDA runtime権限を確認してください。")
        return

    print("\n[警告] テストに失敗しました(Test failed). パッケージの再構築へ移行します(Proceeding to rebuild packages).")
    repair_success = reinstall_libraries()

    if repair_success:
        print("\n再インストールが完了しました(Reinstallation complete). 再テストを実行します(Running tests again)...")
        torch_rc_retry = run_isolated_command([sys.executable, TORCH_TEST_PATH], "PyTorch 診断テスト (再)")
        jax_rc_retry = run_isolated_command([sys.executable, JAX_TEST_PATH], "JAX 診断テスト (再)", extra_env=jax_env)

        if torch_rc_retry == 0 and jax_rc_retry == 0:
            print("\n環境の修復と再テストに成功しました(Environment repaired successfully).")
        elif torch_rc_retry == 3 or jax_rc_retry == 3:
            print("\n[停止] 再インストール後もGPU非認識です。ドライバ/CUDAランタイムを確認してください。")
        else:
            print(
                "\n[致命的エラー] 修復後もテストに失敗しました(Failed even after repair). NVIDIAドライバやCUDA Toolkit自体のバージョン確認が必要です."
            )
    else:
        print("\n[致命的エラー] パッケージの再インストールに失敗しました(Failed to reinstall packages).")


if __name__ == "__main__":
    main()
