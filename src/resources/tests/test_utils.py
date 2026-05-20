from resources import utils


def test_detect_execution_environment_windows(monkeypatch) -> None:
    monkeypatch.setattr(utils.platform, "system", lambda: "Windows")
    monkeypatch.setattr(utils.platform, "release", lambda: "11")
    monkeypatch.setattr(utils.platform, "version", lambda: "10.0.22631")
    monkeypatch.setattr(utils.platform, "platform", lambda: "Windows-11-10.0.22631")
    monkeypatch.setenv("WSL_DISTRO_NAME", "", prepend=False)
    monkeypatch.setenv("WSL_INTEROP", "", prepend=False)
    monkeypatch.setattr(utils, "_read_text_if_exists", lambda path, limit=4000: "")

    env = utils.detect_execution_environment()
    assert env["execution_os"] == "windows"
    assert env["is_wsl"] is False


def test_detect_execution_environment_wsl(monkeypatch) -> None:
    monkeypatch.setattr(utils.platform, "system", lambda: "Linux")
    monkeypatch.setattr(utils.platform, "release", lambda: "5.15.153.1-microsoft-standard-WSL2")
    monkeypatch.setattr(utils.platform, "version", lambda: "#1 SMP")
    monkeypatch.setattr(utils.platform, "platform", lambda: "Linux-5.15.153.1-microsoft-standard-WSL2")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setenv("WSL_INTEROP", "/run/WSL/123_interop")
    monkeypatch.setattr(utils, "_read_text_if_exists", lambda path, limit=4000: "")

    env = utils.detect_execution_environment()
    assert env["execution_os"] == "wsl"
    assert env["is_wsl"] is True
    assert env["host_os_hint"] == "windows"


def test_detect_execution_environment_native_linux(monkeypatch) -> None:
    monkeypatch.setattr(utils.platform, "system", lambda: "Linux")
    monkeypatch.setattr(utils.platform, "release", lambda: "6.8.0")
    monkeypatch.setattr(utils.platform, "version", lambda: "#1 SMP PREEMPT_DYNAMIC")
    monkeypatch.setattr(utils.platform, "platform", lambda: "Linux-6.8.0")
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.delenv("WSL_INTEROP", raising=False)
    monkeypatch.setattr(utils, "_read_text_if_exists", lambda path, limit=4000: "Linux version 6.8.0")

    env = utils.detect_execution_environment()
    assert env["execution_os"] == "native_linux"
    assert env["is_wsl"] is False
