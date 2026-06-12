"""Windows-safe multiprocessing configuration."""

from __future__ import annotations

import atexit
import multiprocessing
import sys


def _patch_multiprocess_resource_tracker() -> None:
    """Work around multiprocess ResourceTracker shutdown bug on Python 3.12+."""
    try:
        from multiprocess import resource_tracker

        tracker = resource_tracker._resource_tracker

        def _safe_stop(self: object) -> None:
            try:
                lock = getattr(self, "_lock", None)
                if lock is not None and not hasattr(lock, "_recursion_count"):
                    lock._recursion_count = 0  # type: ignore[attr-defined]
                original_stop = getattr(type(self), "_stop_locked", None)
                if callable(original_stop):
                    original_stop(self)
            except Exception:
                return

        tracker._stop = _safe_stop.__get__(tracker, type(tracker))  # type: ignore[method-assign]
    except Exception:
        pass


def configure_multiprocessing() -> None:
    """Prevent recursive process spawning and noisy shutdown errors on Windows."""
    if sys.platform == "win32":
        multiprocessing.freeze_support()
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        _patch_multiprocess_resource_tracker()
        atexit.register(_patch_multiprocess_resource_tracker)
