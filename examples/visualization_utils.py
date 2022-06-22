from typing import Dict, List, Tuple

import matplotlib as mplt
from matplotlib import pyplot as plt

Name = str
History = Dict[str, List[float]]

def histories_plot(
    histories: Dict[Name, History],
    names: List[str] = None,
    figsize: Tuple[int] = (10, 4),
):
    """Helper function for plotting multiple named training histories."""

    if names is None:
        # Create list of measurements (metrics and losses)
        # and make sure all histories are the same
        measurements = None
        for name in histories:
            keys = histories[name].keys()

            if measurements is None:
                measurements = set(keys)
            else:
                if len(keys) != len(measurements):
                    raise Exception("histories do not share the same measurements")
                measurements.update(keys)
    else:
        measurements = names

    # Prepare figure and grid (row)
    fig = plt.figure(figsize=figsize)
    gs = mplt.gridspec.GridSpec(1, len(measurements))

    # On each subplot, plot a measurement from all histories
    for subplot_index, measurement in enumerate(sorted(measurements)):
        ax = fig.add_subplot(gs[subplot_index])

        min_ = None
        max_ = None

        for name in histories:
            history = histories[name][measurement]
            ax.plot(range(len(history)), history, label=name)

            min_ = min(history) if min_ is None else min(min_, min(history))
            max_ = max(history) if max_ is None else max(max_, max(history))

        # Automatically adjust plot limits
        scale = abs(max_ - min_) * 0.1 + 0.1

        ax.set_ylim(
            bottom=min_ + scale if min_ < 0 else min_ - scale,
            top=max_ - scale if max_ < 0 else max_ + scale,
        )
        ax.grid()
        ax.legend()
        ax.set_title(measurement)

    plt.show()
