from __future__ import annotations

import ctypes
import os


def _try_preload_cuda_symbols() -> None:
    """Best-effort preload for CUDA JIT-link symbols used by cuVS.

    Some environments import `torch` (or other CUDA consumers) before importing
    our Rust extension, which can lead to missing JIT-link symbols at dlopen()
    time for cuVS. Preloading the relevant CUDA libs with RTLD_GLOBAL makes the
    symbols available regardless of import order.
    """

    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    libs = ("libnvJitLink.so.13", "libcudart.so.13")

    # Prefer loading by absolute path from LD_LIBRARY_PATH to avoid accidentally
    # pulling an incompatible system CUDA library when multiple installations exist.
    ld_paths = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    for lib in libs:
        loaded = False
        for d in ld_paths:
            cand = os.path.join(d, lib)
            if os.path.exists(cand):
                try:
                    ctypes.CDLL(cand, mode=mode)
                    loaded = True
                    break
                except OSError:
                    # Try next candidate directory.
                    continue
        if loaded:
            continue

        # Fallback to standard loader search.
        try:
            ctypes.CDLL(lib, mode=mode)
        except OSError:
            pass


_try_preload_cuda_symbols()

