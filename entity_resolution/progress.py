"""One place for the progress-bar look (tqdm): every slow CPU / GPU loop uses bar()."""
import sys

from tqdm import tqdm


def bar(iterable=None, desc="", total=None, unit="it", leave=True):
    """tqdm bar on stderr, refreshed at most every 2 s (keeps logs small on long runs)."""
    if total is None and iterable is not None and hasattr(iterable, "__len__"):
        total = len(iterable)
    return tqdm(iterable, desc=f"    {desc}", total=total, unit=unit, leave=leave,
                mininterval=2.0, smoothing=0.05, dynamic_ncols=True, file=sys.stderr)
