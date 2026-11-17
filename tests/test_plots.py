import numpy as np

from weather_forecaster.plots import Panel, crossing, lead_time_chart


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
