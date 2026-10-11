"""Bounded fail-closed atomic file output for project oracle tooling."""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import tempfile


Replace = Callable[[str | Path, str | Path], None]
SyncDirectory = Callable[[Path], None]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_atomic_file(
    output: Path,
    data: bytes,
    *,
    maximum_bytes: int,
    replace: Replace = os.replace,
    sync_directory: SyncDirectory = _fsync_directory,
) -> None:
    output = output.absolute()
    if maximum_bytes < 1 or len(data) > maximum_bytes:
        raise ValueError("atomic output is empty or exceeds its byte bound")
    if output == Path("/") or not output.name or output.is_symlink():
        raise ValueError("unsafe output file")
    if output.exists() and not output.is_file():
        raise ValueError("output path must be a regular file")

    for parent in output.parents:
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            raise ValueError("unsafe output parent directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink() or output.parent.resolve(strict=True) != output.parent:
        raise ValueError("unsafe output parent directory")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        dir=output.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as temporary:
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        replace(temporary_name, output)
        sync_directory(output.parent)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
