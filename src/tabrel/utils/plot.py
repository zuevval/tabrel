from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp


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


def calc_diffs(y: np.ndarray, r: np.ndarray, threshold: float, plot: bool, out_fname: str | None) -> float:
    """
    @param y: array of shape (n,)
    @param r: array of shape (n, n)
    @param threshold: if r[i, j] > threshold, i,j considered adjacent

    @returns two-sided KS test p-value
    """
    y_2d = y[np.newaxis, :]
    y_diffs = y_2d.T - y_2d

    adj_diffs = y_diffs[np.where(np.tril(r > threshold).astype(int))].flatten()
    backgnd_diffs = y_diffs[
        np.where(np.tril(1 - (r > threshold).astype(int)))
    ].flatten()

    if plot:
        plt.figure(figsize=(10, 10))
        _, (ax_bckgnd, ax_adj) = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax_bckgnd.hist(backgnd_diffs, density=True)
        ax_bckgnd.set_title("non-related")
        ax_adj.hist(adj_diffs, density=True)
        ax_adj.set_title("related")
        if out_fname:
            plt.savefig(out_fname)

    res = ks_2samp(backgnd_diffs, adj_diffs)
    return res.pvalue  # type:ignore
