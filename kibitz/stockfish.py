"""
Finding Stockfish, or fetching it. README §9.

An engine on $STOCKFISH_PATH or PATH is used as it is. Otherwise the official
Stockfish 19 build for this machine is downloaded once, checked against the
release's SHA-256 and kept in the Kibitz cache ($KIBITZ_HOME, default
~/.cache/kibitz). Stockfish is GPL-3: Kibitz fetches the official release
rather than bundling it.
"""

from __future__ import annotations

import hashlib
import io
import os
import platform
import shutil
import ssl
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

RELEASE = "sf_19"
URL = "https://github.com/official-stockfish/Stockfish/releases/download/{release}/{asset}"

# (system, machine) -> (asset, sha256), from the release page. The "universal"
# builds pick the fastest instruction set the CPU has at run time.
ASSETS = {
    ("Darwin", "arm64"): ("stockfish-macos-universal.tar.gz",
                          "a1f0e3bcc5a6927a11fe6fc8e54a779754645f3c2bae2cf13420fd1957adaa77"),
    ("Darwin", "x86_64"): ("stockfish-macos-universal.tar.gz",
                           "a1f0e3bcc5a6927a11fe6fc8e54a779754645f3c2bae2cf13420fd1957adaa77"),
    ("Linux", "x86_64"): ("stockfish-linux-x86-64-universal.tar.gz",
                          "9defc0d4e55d49c65a6d042f3e571a39fcea499ade6dbe741b53b8c65e03611f"),
    ("Linux", "arm64"): ("stockfish-linux-arm64-universal.tar.gz",
                         "fe26cfd1d9db4c8af3d21e24d9ff34cacb31c1f940085a7583da11796f2bac01"),
    ("Windows", "x86_64"): ("stockfish-windows-x86-64-universal.zip",
                            "3c8bf1f9ea66a09350a40df4f632288285ac206d99f33ab5842c408fc30b48a7"),
    ("Windows", "arm64"): ("stockfish-windows-arm64-universal.zip",
                           "8372ad3f0d7276deb2c70f801f541ec7db463219fc6d9c7592864e542aa4f401"),
}

# platform.machine() names the same CPU differently on each OS.
_MACHINE = {"amd64": "x86_64", "x86_64": "x86_64", "aarch64": "arm64", "arm64": "arm64"}


def cache_dir() -> Path:
    return Path(os.environ.get("KIBITZ_HOME") or Path.home() / ".cache" / "kibitz")


def asset_key(system: str, machine: str) -> tuple[str, str]:
    return system, _MACHINE.get(machine.lower(), machine)


def asset_for(system: str, machine: str) -> tuple[str, str]:
    key = asset_key(system, machine)
    if key not in ASSETS:
        raise RuntimeError(f"no Stockfish download for {system} {machine}; install Stockfish "
                           "and put it on PATH, or set $STOCKFISH_PATH")
    return ASSETS[key]


def locate() -> str:
    """$STOCKFISH_PATH, else `stockfish` on PATH, else the downloaded build."""
    if os.environ.get("STOCKFISH_PATH"):
        return os.environ["STOCKFISH_PATH"]
    if found := shutil.which("stockfish"):
        return found
    asset, digest = asset_for(platform.system(), platform.machine())
    binary = cache_dir() / RELEASE / (asset.split(".")[0] + (".exe" if asset.endswith(".zip") else ""))
    if not binary.exists():
        download(asset, digest, binary)
    return str(binary)


def _fetch(url: str) -> bytes:
    # certifi's CA bundle: python.org's macOS Python has none of its own until
    # "Install Certificates.command" is run.
    import certifi
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=context) as response:
        return response.read()


def download(asset: str, digest: str, binary: Path) -> None:
    """Fetch `asset`, check it, and unpack its engine to `binary`."""
    print(f"Stockfish not found; downloading {asset} (~80 MB) to {binary.parent}",
          file=sys.stderr)
    data = _fetch(URL.format(release=RELEASE, asset=asset))
    if hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError(f"{asset} does not match the release checksum; not using it")
    member = f"stockfish/{binary.name}"
    if asset.endswith(".zip"):
        engine = zipfile.ZipFile(io.BytesIO(data)).read(member)
    else:
        engine = tarfile.open(fileobj=io.BytesIO(data)).extractfile(member).read()
    binary.parent.mkdir(parents=True, exist_ok=True)
    partial = binary.with_suffix(".partial")
    partial.write_bytes(engine)
    partial.chmod(0o755)
    partial.replace(binary)
