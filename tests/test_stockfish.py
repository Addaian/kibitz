"""
Unit tests for kibitz.stockfish: which engine is used, and the download.

Nothing here touches the network; `_fetch` is replaced by archives built in
memory, laid out like the official release.
"""

import hashlib
import io
import tarfile
import zipfile

import pytest

from kibitz import stockfish as S


def tar_release(name: str, engine: bytes) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for member, data in ((f"stockfish/{name}", engine), ("stockfish/Copying.txt", b"GPL")):
            info = tarfile.TarInfo(member)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def zip_release(name: str, engine: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"stockfish/{name}", engine)
        z.writestr("stockfish/Copying.txt", b"GPL")
    return buf.getvalue()


@pytest.fixture
def no_installed_engine(monkeypatch, tmp_path):
    monkeypatch.delenv("STOCKFISH_PATH", raising=False)
    monkeypatch.setattr(S.shutil, "which", lambda name: None)
    monkeypatch.setenv("KIBITZ_HOME", str(tmp_path))


def serve(monkeypatch, system, machine, archive):
    """Pretend to be `system`/`machine`, with `archive` as its release asset."""
    asset, _ = S.asset_for(system, machine)
    monkeypatch.setattr(S.platform, "system", lambda: system)
    monkeypatch.setattr(S.platform, "machine", lambda: machine)
    monkeypatch.setitem(S.ASSETS, S.asset_key(system, machine),
                        (asset, hashlib.sha256(archive).hexdigest()))
    fetched = []
    monkeypatch.setattr(S, "_fetch", lambda url: fetched.append(url) or archive)
    return fetched


@pytest.mark.parametrize("system, machine, asset", [
    ("Darwin", "arm64", "stockfish-macos-universal.tar.gz"),
    ("Darwin", "x86_64", "stockfish-macos-universal.tar.gz"),
    ("Linux", "x86_64", "stockfish-linux-x86-64-universal.tar.gz"),
    ("Linux", "aarch64", "stockfish-linux-arm64-universal.tar.gz"),
    ("Windows", "AMD64", "stockfish-windows-x86-64-universal.zip"),
    ("Windows", "ARM64", "stockfish-windows-arm64-universal.zip"),
])
def test_each_platform_gets_its_release_asset(system, machine, asset):
    assert S.asset_for(system, machine)[0] == asset


def test_an_unsupported_platform_is_refused():
    with pytest.raises(RuntimeError, match="STOCKFISH_PATH"):
        S.asset_for("Linux", "riscv64")


def test_stockfish_path_wins(monkeypatch):
    monkeypatch.setenv("STOCKFISH_PATH", "/somewhere/stockfish")
    monkeypatch.setattr(S.shutil, "which", lambda name: "/usr/bin/stockfish")
    assert S.locate() == "/somewhere/stockfish"


def test_an_engine_on_path_is_used_as_it_is(monkeypatch):
    monkeypatch.delenv("STOCKFISH_PATH", raising=False)
    monkeypatch.setattr(S.shutil, "which", lambda name: "/usr/bin/stockfish")
    monkeypatch.setattr(S, "_fetch", lambda url: pytest.fail("downloaded"))
    assert S.locate() == "/usr/bin/stockfish"


def test_without_one_the_release_is_downloaded_once(monkeypatch, tmp_path, no_installed_engine):
    fetched = serve(monkeypatch, "Linux", "x86_64",
                    tar_release("stockfish-linux-x86-64-universal", b"ENGINE"))
    first = S.locate()
    assert S.locate() == first
    assert len(fetched) == 1
    assert fetched[0].endswith("/sf_19/stockfish-linux-x86-64-universal.tar.gz")
    assert first == str(tmp_path / "sf_19" / "stockfish-linux-x86-64-universal")
    with open(first, "rb") as f:
        assert f.read() == b"ENGINE"
    assert S.Path(first).stat().st_mode & 0o111


def test_windows_gets_the_exe_from_the_zip(monkeypatch, tmp_path, no_installed_engine):
    serve(monkeypatch, "Windows", "AMD64",
          zip_release("stockfish-windows-x86-64-universal.exe", b"ENGINE.EXE"))
    path = S.locate()
    assert path.endswith("stockfish-windows-x86-64-universal.exe")
    with open(path, "rb") as f:
        assert f.read() == b"ENGINE.EXE"


def test_a_download_that_fails_the_checksum_is_not_used(monkeypatch, tmp_path, no_installed_engine):
    serve(monkeypatch, "Linux", "x86_64", tar_release("stockfish-linux-x86-64-universal", b"ENGINE"))
    monkeypatch.setattr(S, "_fetch", lambda url: tar_release("stockfish-linux-x86-64-universal",
                                                              b"TAMPERED"))
    with pytest.raises(RuntimeError, match="checksum"):
        S.locate()
    assert not any(tmp_path.rglob("stockfish-*"))
