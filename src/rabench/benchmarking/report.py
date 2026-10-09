"""
report.py

Turns the raw per-fold benchmark DataFrame (from runner.py) into:
  * a summary table (mean +/- std per model, across folds)
  * bar charts comparing accuracy (R2) and efficiency (train time, model
    size) across models -- the core artifact for your Objective #2
    (deep-vs-classical trade-off analysis).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def summarize(results_df: pd.DataFrame) -> pd.DataFrame:
    agg = results_df.groupby(["task", "model_name", "family"]).agg(
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        mse_mean=("mse", "mean"),
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
        train_time_mean=("train_time_sec", "mean"),
        predict_time_mean=("predict_time_sec", "mean"),
        model_size_bytes=("model_size_bytes", "mean"),
        param_count=("param_count", "mean"),
        n_runs=("mae", "size"),
    ).reset_index()
    return agg.sort_values(["task", "r2_mean"], ascending=[True, False])


def _fmt_actual(v: float) -> str:
    """Formats a raw value with precision scaled to its magnitude, for
    display under a percentage label."""
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _label_bars_pct_of_max(ax, container, values, unit: str = "") -> None:
    """Writes each bar's value as a percentage of the largest value in
    `values`, with the actual number below it, centered inside the bar --
    makes relative differences (e.g. "this model trains in 38% of the
    slowest model's time") readable at a glance while still keeping the
    real number on the chart."""
    values = list(values)
    max_val = max(values) if values else 0
    labels = [
        f"{(v / max_val * 100):.0f}%\n{_fmt_actual(v)}{unit}" if max_val > 0 else f"0%\n{_fmt_actual(v)}{unit}"
        for v in values
    ]
    ax.bar_label(container, labels=labels, label_type="center", color="white", fontweight="bold", fontsize=8)


def plot_accuracy_comparison(summary_df: pd.DataFrame, task: str, output_path: str) -> None:
    df = summary_df[summary_df["task"] == task].sort_values("r2_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4C72B0" if f == "classical" else "#C44E52" for f in df["family"]]
    bars = ax.bar(df["model_name"], df["r2_mean"], yerr=df["r2_std"], color=colors, capsize=4)
    ax.bar_label(
        bars,
        labels=[f"{v * 100:.1f}%\n{v:.3f}" for v in df["r2_mean"]],
        label_type="center",
        color="white",
        fontweight="bold",
        fontsize=8,
    )
    ax.set_ylabel("R² (mean across folds)")
    ax.set_title(f"{task}: accuracy comparison (blue=classical, red=deep)")
    ax.set_ylim(bottom=min(0, df["r2_mean"].min() - 0.1))
    plt.xticks(rotation=20)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_efficiency_comparison(summary_df: pd.DataFrame, task: str, output_path: str) -> None:
    df = summary_df[summary_df["task"] == task].sort_values("train_time_mean")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    colors = ["#4C72B0" if f == "classical" else "#C44E52" for f in df["family"]]
    time_bars = axes[0].bar(df["model_name"], df["train_time_mean"], color=colors)
    _label_bars_pct_of_max(axes[0], time_bars, df["train_time_mean"], unit="s")
    axes[0].set_ylabel("Mean training time (s)")
    axes[0].set_title(f"{task}: training time (% of slowest)")
    axes[0].tick_params(axis="x", rotation=20)

    size_kb = df["model_size_bytes"].fillna(0) / 1024.0
    size_bars = axes[1].bar(df["model_name"], size_kb, color=colors)
    _label_bars_pct_of_max(axes[1], size_bars, size_kb, unit="KB")
    axes[1].set_ylabel("Model size (KB)")
    axes[1].set_title(f"{task}: model size (% of largest)")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_full_report(results_df: pd.DataFrame, output_dir: str, task: Optional[str] = None) -> pd.DataFrame:
    """Writes summary.csv + accuracy/efficiency plots to output_dir.
    Returns the summary DataFrame."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize(results_df)
    summary.to_csv(output_dir / "summary.csv", index=False)

    tasks = [task] if task else summary["task"].unique().tolist()
    for t in tasks:
        plot_accuracy_comparison(summary, t, output_dir / f"{t}_accuracy.png")
        plot_efficiency_comparison(summary, t, output_dir / f"{t}_efficiency.png")

    return summary
