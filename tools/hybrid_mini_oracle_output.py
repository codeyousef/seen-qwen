"""Fail-closed atomic output promotion for the hybrid mini-model oracle."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import os
from pathlib import Path
import shutil
import tempfile


Replace = Callable[[Path, Path], None]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_file(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def promote_oracle_output(
    output: Path,
    assets: Mapping[str, bytes],
    *,
    replace: Replace = os.replace,
) -> None:
    """Atomically replace ``output`` while preserving it on promotion failure."""

    if output == Path("/") or not output.name or output.is_symlink():
        raise ValueError("unsafe output directory")
    if output.exists() and not output.is_dir():
        raise ValueError("output path must be a directory")
    if not assets:
        raise ValueError("oracle output must contain at least one asset")
    for name in assets:
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError(f"unsafe oracle asset name: {name!r}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    backup: Path | None = None
    try:
        for name, data in assets.items():
            _write_file(temporary / name, data)
        _fsync_directory(temporary)

        if output.exists():
            backup = Path(
                tempfile.mkdtemp(prefix=f".{output.name}.previous.", dir=output.parent)
            )
            backup.rmdir()
            replace(output, backup)
        try:
            replace(temporary, output)
        except BaseException:
            if backup is not None and not output.exists():
                replace(backup, output)
                backup = None
                _fsync_directory(output.parent)
            raise

        _fsync_directory(output.parent)
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
            _fsync_directory(output.parent)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary, ignore_errors=True)
        # A non-null backup contains the caller's prior output. Never delete it
        # if restoration itself failed.
