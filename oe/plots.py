"""Plots for fits: data with the fitted curve on top and the residuals underneath."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_fit(curve: dict, title: str, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = curve["panels"]
    ratios = []
    for _ in panels:
        ratios += [3, 1]
    fig, axes = plt.subplots(
        2 * len(panels), 1, figsize=(10, 3.2 + 2.6 * len(panels)), sharex=True, gridspec_kw={"height_ratios": ratios}
    )
    axes = np.atleast_1d(axes)
    marker = "o" if curve.get("markers") else "."
    size = 5 if curve.get("markers") else 3
    for i, panel in enumerate(panels):
        top, bottom = axes[2 * i], axes[2 * i + 1]
        top.plot(panel["t"], panel["data"], marker, ms=size, ls="none", color="#2a6fdb", label="measured")
        top.plot(panel["fit_t"], panel["fit"], "-", lw=1.4, color="#e4572e", label="fit")
        top.set_ylabel(panel["label"])
        bottom.plot(panel["t"], panel["residual"], marker, ms=size - 1, ls="none", color="#555555")
        bottom.axhline(0, color="#999999", lw=0.8)
        bottom.set_ylabel("residual")
        top.legend(loc="best", fontsize=8)
        for ax in (top, bottom):
            ax.grid(alpha=0.3)
    axes[-1].set_xlabel(curve.get("xlabel", "time in the fit window (s)"))
    axes[0].set_title(title, fontsize=10)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
