import numpy as np
import pytest

from weather_forecaster.plots import (
    Panel,
    atlantic_view,
    crossing,
    forecast_maps,
    lead_time_chart,
    rollout_gain_chart,
)


def test_crossing_is_read_off_the_line_between_two_scores():
    days = np.array([0.0, 1.0, 2.0, 3.0])
    assert crossing(days, np.array([0.0, 2.0, 4.0, 6.0]), 3.0) == 1.5


def test_crossing_is_none_when_the_threshold_is_never_reached():
    assert crossing(np.arange(3.0), np.array([0.0, 1.0, 2.0]), 5.0) is None


def test_lead_time_chart_draws_one_panel_per_field(tmp_path):
    panels = [
        Panel("A", [1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 5.0, 6.0], 3.5, {1: 3.0}),
        Panel("B", [0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], 0.9),
    ]
    figure = lead_time_chart([6, 12, 18, 24], panels)
    assert [axes.get_title(loc="left") for axes in figure.axes] == ["A", "B"]
    figure.savefig(tmp_path / "chart.png")
    assert (tmp_path / "chart.png").stat().st_size > 10_000


def test_atlantic_view_puts_greenwich_in_the_middle_and_closes_the_map():
    longitude = np.array([0.0, 90.0, 180.0, 270.0])
    east, centred = atlantic_view(longitude, np.array([[0.0, 1.0, 2.0, 3.0]]))
    assert east.tolist() == [-180.0, -90.0, 0.0, 90.0, 180.0]
    assert centred.tolist() == [[2.0, 3.0, 0.0, 1.0, 2.0]]


def test_forecast_maps_draw_truth_above_forecast_for_every_lead_time(tmp_path):
    longitude = np.linspace(0, 360, 16, endpoint=False)
    latitude = np.linspace(-78.75, 78.75, 8)
    waves = (
        np.sin(np.deg2rad(longitude))[None, None, :] * np.cos(np.deg2rad(latitude))[None, :, None]
    )
    truth = 1000 * waves * np.array([1.0, 2.0])[:, None, None]
    figure = forecast_maps(
        longitude,
        latitude,
        truth,
        0.8 * truth,
        (waves[0] > 0).astype(np.float32),
        headings=["After 1 day", "After 3 days"],
        notes=["error 1", "error 2"],
        band=300,
        title="A forecast",
        subtitle="Difference from usual",
        scale_label="units",
    )
    assert len(figure.axes) == 2 * 2 + 1  # two rows of maps and the colour scale
    figure.savefig(tmp_path / "maps.png")
    assert (tmp_path / "maps.png").stat().st_size > 10_000


def test_rollout_gain_chart_draws_one_line_per_field(tmp_path):
    gains = {"Z500": [5.0, 10.0, 20.0, 15.0], "T850": [2.0, -1.0, 4.0, 6.0]}
    figure = rollout_gain_chart([6, 12, 18, 24], gains)
    (axes,) = figure.axes
    assert [line.get_label() for line in axes.get_lines() if line.get_label() in gains] == [
        "Z500",
        "T850",
    ]
    assert axes.get_ylim()[0] < -1 and axes.get_ylim()[1] > 20
    figure.savefig(tmp_path / "gain.png")
    assert (tmp_path / "gain.png").stat().st_size > 10_000


def test_rollout_gain_chart_refuses_more_fields_than_colours():
    with pytest.raises(ValueError, match="at most"):
        rollout_gain_chart([6], {name: [1.0] for name in "ABCD"})
