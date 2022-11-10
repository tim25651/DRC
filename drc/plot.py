from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


def scatter(x: Sequence[float], y: Sequence[float]):
    """Simple scatter plot

    Args:
        x (Sequence[float]): X values
        y (Sequence[float]): Corresponding Y values
    """
    plt.scatter(x, y)


def plot(x: Sequence[float], y: Sequence[float]):
    """Simple black plot

    Args:
        x (Sequence[float]): X values
        y (Sequence[float]): Corresponding Y values
    """
    plt.plot(x, y, color="black")


def errorbars(x: Sequence[float], y: Sequence[float], err: Sequence[float]):
    """Plot bold black errorbars without connecting line

    Args:
        x (Sequence[float]): X values
        y (Sequence[float]): Mean values
        err (Sequence[float]): Standard error of the means
    """
    plt.errorbar(
        x,
        y,
        err,
        color="black",
        marker="o",
        capsize=4,
        linewidth=0,
        elinewidth=2,
    )


def axes(ax: Axes):
    """Inverse X-axis, remove right and upper spines, make lines and ticks bold

    Args:
        ax (Axes): Axes from current plot
    """
    ax.invert_xaxis()
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.tick_params(width=2)


def yticks(y: Sequence[float]):
    """Plot bold ticks on Y-axis with 0.5 step size"""
    max_val = np.ceil(max(y) * 2).astype(int)
    min_val = np.floor(min(y) * 2).astype(int)
    y_ticks = [x / 2 for x in range(min_val, max_val + 1)][1:-1]
    plt.yticks(y_ticks, weight="bold")


def labels(title: str, xlabel: str, ylabel: str):
    """Add bold labels to the plot and its axis

    Args:
        title (str): Title of plot
        xlabel (str): Label on the X-axis
        ylabel (str): Label on the Y-axis
    """
    plt.title(title, weight="bold", size="18")
    plt.xlabel(xlabel, weight="bold", size="16")
    plt.ylabel(ylabel, weight="bold", size="16")


def xticks(x: Sequence[float]):
    """Plot ticks on Y axis with logarithmic scaling and labels on every log step

    Args:
        x (Sequence[float]): Logarithmic X values
        scaling (float, optional): Logarithmic shift of the X values to the standard unit. Defaults to 1e-6.
    """
    min_val = 10 ** np.floor(-max(x))
    max_val = 10 ** np.ceil(-min(x))

    x_ticks = [f * min_val for f in (0.7, 0.8, 0.9, 1.0)]
    x_tick_labels = ["", "", "", f"{min_val:.2f}"]
    while min_val <= max_val:
        for f in range(2, 10):
            x_ticks.append(f * min_val)
            x_tick_labels.append("")
        min_val = 10 * min_val
        x_ticks.append(min_val)
        label = f"{min_val:.2f}" if min_val < 1 else f"{min_val:.0f}"
        x_tick_labels.append(label)
        if (max_val - min_val) < 1e-15:
            break
    log_x_ticks = -np.log10(x_ticks)
    plt.xticks(log_x_ticks, x_tick_labels, weight="bold")
