from __future__ import annotations

import pytest

from Setup.startup import (
    ApiAlreadyRunningError,
    ApiInstanceLock,
    gpu_allocation_message,
)


def test_instance_lock_rejects_duplicate_server(tmp_path):
    lock_path = tmp_path / "api.lock"
    first = ApiInstanceLock(lock_path).acquire()
    try:
        with pytest.raises(ApiAlreadyRunningError, match="Another RouGE API process"):
            ApiInstanceLock(lock_path).acquire()
    finally:
        first.close()

    replacement = ApiInstanceLock(lock_path).acquire()
    replacement.close()


def test_gpu_allocation_message_contains_recovery_settings():
    message = gpu_allocation_message(
        "INT8", free_vram_mb=256, context_size=4096, batch_size=512
    )

    assert "Free GPU memory: 256 MB" in message
    assert "ROUGE_CONTEXT_SIZE=2048" in message
    assert "ROUGE_BATCH_SIZE=256" in message
