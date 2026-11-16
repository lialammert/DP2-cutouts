# progress.py
# AstroDeep Project
# Time-stamped log lines and a progress counter for the long loops (targets of a patch, empty-sky windows).
# Every line is flushed, so that it reaches a log file at once even when stdout is redirected (block-buffered).

import datetime
import time


def log(msg):
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def _fmt(seconds):
    if seconds != seconds or seconds == float("inf"):          # NaN or inf
        return "?"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def elapsed_since(t0):
    """Time since t0 (a time.monotonic() value), formatted."""
    return _fmt(time.monotonic() - t0)


class Progress:
    """Logs `label: i / n (pct), elapsed, rate, ETA` about `n_reports` times over the loop, and at the end.

    Call update() once per item done.
    """

    def __init__(self, n, label, n_reports=20):
        self.n = int(n)
        self.label = label
        self.every = max(1, self.n // n_reports) if self.n else 1
        self.i = 0
        self.t0 = time.monotonic()
        log(f"{label}: 0 / {self.n:,}")

    def update(self, k=1):
        self.i += k
        if self.i % self.every == 0 or self.i == self.n:
            dt = time.monotonic() - self.t0
            rate = self.i / dt if dt > 0 else float("inf")
            eta = (self.n - self.i) / rate if rate > 0 else float("nan")
            pct = 100.0 * self.i / self.n if self.n else 100.0
            log(f"{self.label}: {self.i:,} / {self.n:,} ({pct:.0f}%), {_fmt(dt)} elapsed, "
                f"{rate:.2f}/s, ETA {_fmt(eta)}")

    def elapsed(self):
        return _fmt(time.monotonic() - self.t0)
