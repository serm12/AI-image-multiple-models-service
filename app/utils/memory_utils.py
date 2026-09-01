"""Helpers for returning unused image-processing memory to the OS."""

import ctypes
import gc
import sys
from typing import Dict


def release_process_memory(*, clear_image_cache: bool = False) -> Dict[str, int | bool]:
    """Collect unreachable objects and ask glibc to release free heap pages.

    ``malloc_trim`` is Linux/glibc-specific, so failures are intentionally ignored.
    The function is safe to call after a task and from the periodic cleanup loop.
    """
    cache_cleared = False
    if clear_image_cache:
        try:
            from app.utils.watermark_utils import clear_logo_cache

            clear_logo_cache()
            cache_cleared = True
        except Exception:
            pass

    collected = gc.collect()
    trimmed = False
    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL("libc.so.6")
            malloc_trim = libc.malloc_trim
            malloc_trim.argtypes = [ctypes.c_size_t]
            malloc_trim.restype = ctypes.c_int
            trimmed = bool(malloc_trim(0))
        except (AttributeError, OSError):
            pass

    return {
        "collected": collected,
        "malloc_trimmed": trimmed,
        "image_cache_cleared": cache_cleared,
    }
