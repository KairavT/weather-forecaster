"""Charts of the results.

Colours, line widths and fonts are fixed here so that every figure in the repo matches. The
figures are built on matplotlib's Figure directly, without pyplot, so they need no display.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"  # the model the chart is about
ORANGE = "#eb6834"  # the model it is compared with
FONTS = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


@dataclass(frozen=True)
class Panel:
    """One field's scores: the stepper and persistence at every lead time, and the rest."""

    title: str
    stepper: list[float]
    persistence: list[float]
    weekly_climatology: float
    cnn: dict[int, float] = field(default_factory=dict)  # lead time in days -> RMSE


def crossing(lead_days: np.ndarray, values: np.ndarray, threshold: float) -> float | None:
    """Lead time at which values first rise above threshold, or None if they never do.

    Scores are only known every 6 hours, so the crossing is read off the straight line
    between the last score below the threshold and the first one above it.
    """
    above = np.flatnonzero(np.asarray(values) > threshold)
    if len(above) == 0:
        return None
    k = int(above[0])
    if k == 0:
        return float(lead_days[0])
    fraction = (threshold - values[k - 1]) / (values[k] - values[k - 1])
    return float(lead_days[k - 1] + fraction * (lead_days[k] - lead_days[k - 1]))


def _text(target, x, y, text, **style):
    style = {"fontsize": 10, "color": SECONDARY_INK, "family": FONTS} | style
    return target.text(x, y, text, **style)


def _dot(axes, x, y, color, size=7):
    """A marker with a ring in the background colour, so it stays readable on top of a line."""
    axes.plot(
        [x],
        [y],
        marker="o",
        markersize=size,
        color=color,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        linestyle="none",
        zorder=5,
        clip_on=False,
    )


def _style_axes(axes) -> None:
    axes.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(AXIS)
    axes.grid(axis="y", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.tick_params(colors=MUTED, labelsize=9.5, length=0, pad=5)
    for label in (*axes.get_xticklabels(), *axes.get_yticklabels()):
        label.set_family(FONTS)


def lead_time_chart(lead_hours: list[int], panels: list[Panel]) -> Figure:
    """Error against lead time, one panel per field.

    Every forecast starts from the true state, so each curve starts at zero error. A dot marks
    where the stepper stops beating weekly climatology, the strongest no-model baseline.
    """
    days = np.concatenate([[0.0], np.asarray(lead_hours) / 24])
    figure = Figure(figsize=(10.8, 4.5), dpi=200, facecolor=SURFACE)
    grid = figure.add_gridspec(
        1, len(panels), left=0.045, right=0.915, top=0.70, bottom=0.13, wspace=0.55
    )

    for column, panel in enumerate(panels):
        axes = figure.add_subplot(grid[0, column])
        _style_axes(axes)
        stepper = np.concatenate([[0.0], panel.stepper])
        persistence = np.concatenate([[0.0], panel.persistence])

        axes.axhline(
            panel.weekly_climatology, color=SECONDARY_INK, linewidth=1.1, linestyle=(0, (4, 3))
        )
        axes.plot(days, persistence, color=MUTED, linewidth=1.6, solid_capstyle="round")
        axes.plot(days, stepper, color=BLUE, linewidth=2.2, solid_capstyle="round", zorder=4)
        for lead, value in panel.cnn.items():
            _dot(axes, lead, value, ORANGE, size=8)

        top = 1.08 * max(persistence.max(), stepper.max(), panel.weekly_climatology)
        axes.set_xlim(0, days[-1])
        axes.set_ylim(0, top)
        axes.set_xticks(np.arange(0, days[-1] + 1, 2))
        axes.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        axes.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.10g}"))
        axes.set_xlabel("Lead time (days)", color=MUTED, fontsize=9.5, family=FONTS, labelpad=6)
        axes.set_title(panel.title, loc="left", color=INK, fontsize=11, family=FONTS, pad=10)

        # the two curves are named where they end. The dashed baseline is named in the legend
        # only, because persistence crosses it too often to leave room for a label.
        end = days[-1] + 0.25
        _text(axes, end, stepper[-1], "Stepper", va="center", clip_on=False)
        _text(axes, end, persistence[-1], "Persistence", va="center", clip_on=False)
        if panel.cnn:
            lead, value = max(panel.cnn.items())
            _text(axes, lead + 0.3, value - 0.012 * top, "CNN", va="top")

        level = crossing(days, stepper, panel.weekly_climatology)
        if level is not None:
            _dot(axes, level, panel.weekly_climatology, BLUE)
            _text(
                axes,
                level - 0.2,
                panel.weekly_climatology + 0.02 * top,
                f"level at {level:.1f} days",
                ha="right",
                va="bottom",
                fontsize=9.5,
            )

    _text(figure, 0.045, 0.93, "Forecast error by lead time", fontsize=14, color=INK, weight="bold")
    _text(
        figure,
        0.045,
        0.875,
        "Root mean squared error on the 2017 to 2018 test years. Lower is better.",
        fontsize=10.5,
    )
    handles = [
        Line2D([], [], color=BLUE, linewidth=2.2, label="Stepper"),
        Line2D([], [], color=MUTED, linewidth=1.6, label="Persistence"),
        Line2D(
            [],
            [],
            color=SECONDARY_INK,
            linewidth=1.1,
            linestyle=(0, (4, 3)),
            label="Weekly climatology",
        ),
    ]
    if any(panel.cnn for panel in panels):
        handles.append(
            Line2D(
                [],
                [],
                color=ORANGE,
                marker="o",
                markersize=8,
                markeredgecolor=SURFACE,
                linestyle="none",
                label="CNN, one network per lead time",
            )
        )
    figure.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.038, 0.855),
        ncols=len(handles),
        frameon=False,
        handlelength=1.8,
        columnspacing=1.8,
        prop={"family": FONTS, "size": 10},
        labelcolor=SECONDARY_INK,
    )
    return figure
