from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_violin(
    ax: plt.Axes, labels: Sequence[str], data: list[np.ndarray], title: str, color: str
) -> None:
    bar_locs = np.arange(len(labels))
    vp = ax.violinplot(data, positions=bar_locs, showmeans=True, showmedians=False)
    for i, group in enumerate(data):
        jitter = np.random.normal(0, 0.05, size=len(group))
        ax.scatter(
            np.full(len(group), bar_locs[i]) + jitter,
            group,
            color="black",
            s=10,
            alpha=0.6,
        )
    for pc in vp["bodies"]:
        pc.set_facecolor(color)
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)
    ax.set_title(title)
    ax.set_xticks(bar_locs)
    ax.set_xticklabels(labels)
