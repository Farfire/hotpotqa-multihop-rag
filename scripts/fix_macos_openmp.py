#!/usr/bin/env python3
"""One-time macOS repair: make faiss and torch share a single OpenMP runtime.

    .venv/bin/python scripts/fix_macos_openmp.py

The problem
-----------
The `torch` and `faiss-cpu` wheels each bundle their own private copy of
`libomp.dylib`. Load both into one process and whichever initialises second
aborts the interpreter:

    OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib
    already initialized.

Anything that encodes with torch *and* searches with FAISS in the same process
hits this, so it takes down `evaluate.py` and `ask.py`.

The fix
-------
Replace faiss's copy with a symlink to torch's, so only one runtime is ever
loaded. Verified on this project: FAISS search results after the symlink are
identical to a NumPy brute-force ranking (same top-10, scores within 5e-7).

`KMP_DUPLICATE_LIB_OK=TRUE` is the workaround you will find first when you
search this error. It does not work here — it converts the clean abort into a
silent segfault partway through encoding.

Re-run this script after any `pip install` that reinstalls torch or faiss-cpu.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def package_dir(name: str) -> Path | None:
    """Locate an installed package without importing it.

    `find_spec` resolves the path but never runs the module. That matters here:
    once the symlink this script manages is dangling — after renaming the
    project directory, say — `import faiss` fails outright, and a repair tool
    that cannot run while the thing is broken is no repair tool at all.
    """
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).parent


def find_libomp() -> tuple[Path | None, Path | None]:
    """Locate faiss's bundled libomp and torch's."""
    faiss_dir = package_dir("faiss")
    torch_dir = package_dir("torch")
    return (
        faiss_dir / ".dylibs" / "libomp.dylib" if faiss_dir else None,
        torch_dir / "lib" / "libomp.dylib" if torch_dir else None,
    )


def main() -> int:
    if sys.platform != "darwin":
        print("Not macOS — nothing to do.")
        return 0

    faiss_lib, torch_lib = find_libomp()

    if torch_lib is None or not torch_lib.exists():
        print("No torch libomp found — nothing to link to.")
        return 0
    if faiss_lib is None:
        print("faiss is not installed — nothing to patch.")
        return 0

    if faiss_lib.is_symlink() and faiss_lib.resolve() == torch_lib.resolve():
        print("Already patched — faiss and torch share one libomp.")
        return 0

    # Keep the original so the change is reversible. Note `exists()` follows
    # symlinks, so a dangling link reads as missing — check for the link itself.
    backup = faiss_lib.with_suffix(".dylib.bak")
    if faiss_lib.is_symlink() or faiss_lib.exists():
        if not backup.exists() and not faiss_lib.is_symlink():
            faiss_lib.rename(backup)
            print(f"backed up faiss libomp -> {backup.name}")
        else:
            faiss_lib.unlink()  # stale or dangling link; the backup is the original

    faiss_lib.symlink_to(torch_lib)
    print(f"linked {faiss_lib} -> {torch_lib}")
    print("Done. torch and faiss now share one OpenMP runtime.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
