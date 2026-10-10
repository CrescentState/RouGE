"""Startup guards and diagnostics for the GPU model server."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import IO


class ApiAlreadyRunningError(RuntimeError):
    """Raised before model loading when another API owns the GPU lock."""


class ApiInstanceLock:
    """Cross-platform process lock preventing duplicate model residency."""

    def __init__(self, path: str | Path | None = None):
        default_path = Path(tempfile.gettempdir()) / "rouge-model-server.lock"
        self.path = Path(path or os.getenv("ROUGE_API_LOCK_PATH", default_path))
        self._stream: IO[str] | None = None

    def acquire(self) -> "ApiInstanceLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+", encoding="utf-8")
        try:
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                if self.path.stat().st_size == 0:
                    stream.write("0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            try:
                stream.seek(0)
                owner = stream.read().strip() or "unknown"
            except OSError:
                # The held byte-range lock also blocks reading the PID back
                # on Windows; report the conflict without the owner PID.
                owner = "unknown (lock file unreadable while held)"
            stream.close()
            raise ApiAlreadyRunningError(
                "Another RouGE API process is already loading or hosting the models "
                f"(recorded PID: {owner}). Use the existing server instead of starting "
                "a second copy."
            ) from exc

        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()))
        stream.flush()
        self._stream = stream
        return self

    def close(self) -> None:
        if self._stream is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._stream.seek(0)
                msvcrt.locking(self._stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        finally:
            self._stream.close()
            self._stream = None


def gpu_allocation_message(
    engine_name: str,
    *,
    free_vram_mb: int | None,
    context_size: int,
    batch_size: int,
) -> str:
    free_text = "unknown" if free_vram_mb is None else f"{free_vram_mb} MB"
    return (
        f"Failed to initialize the {engine_name} llama.cpp context. Free GPU memory: "
        f"{free_text}; context: {context_size}; batch: {batch_size}. Check that another "
        "RouGE/LLM process is not using the GPU. If only one server is running, retry "
        "with ROUGE_CONTEXT_SIZE=2048 and/or ROUGE_BATCH_SIZE=256."
    )
