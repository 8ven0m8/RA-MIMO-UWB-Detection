"""progress.py -- log-friendly progress reporting.

Everything here prints whole lines (no carriage-return redraws), so output
stays readable when stdout is redirected to a file, e.g.
`nohup python scripts/run_all.py > train.log &` followed by `tail -f`.
"""

from __future__ import annotations

import time

BAR_WIDTH = 30


def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def text_bar(done: int, total: int, width: int = BAR_WIDTH) -> str:
    frac = done / total if total else 1.0
    filled = int(round(frac * width))
    return f"[{'#' * filled}{'.' * (width - filled)}] {frac * 100:5.1f}%"


class EpochHeartbeat:
    """Prints a deep model's training state at most once every `interval`
    seconds (plus once at the end), so a long fit isn't silent between
    the runner's per-run progress lines. Throttled by wall-clock time, not
    epoch count, so fast and slow models produce a similar volume of log."""

    def __init__(self, label: str, max_epochs: int, patience: int, interval: float = 30.0):
        self.label = label
        self.max_epochs = max_epochs
        self.patience = patience
        self.interval = interval
        self.start = time.perf_counter()
        self._last_print = self.start

    def update(self, epoch: int, val_loss: float, best_val: float, epochs_no_improve: int) -> None:
        now = time.perf_counter()
        if now - self._last_print < self.interval:
            return
        self._last_print = now
        self._print(epoch, val_loss, best_val, epochs_no_improve, now)

    def finish(self, epoch: int, best_val: float, epochs_no_improve: int) -> None:
        now = time.perf_counter()
        if now - self.start < self.interval:
            return  # fit was short enough that the runner's line covers it
        reason = "early stop" if epochs_no_improve >= self.patience else "max epochs"
        print(
            f"      {self.label} done: {epoch + 1} epochs ({reason}), best val {best_val:.4f}, "
            f"{fmt_duration(now - self.start)}",
            flush=True,
        )

    def _print(self, epoch, val_loss, best_val, epochs_no_improve, now) -> None:
        print(
            f"      {self.label} epoch {epoch + 1}/{self.max_epochs} "
            f"{text_bar(epoch + 1, self.max_epochs, width=20)} | val {val_loss:.4f} best {best_val:.4f} "
            f"| no-improve {epochs_no_improve}/{self.patience} | {fmt_duration(now - self.start)}",
            flush=True,
        )
