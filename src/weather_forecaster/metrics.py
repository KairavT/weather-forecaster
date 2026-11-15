"""Forecast scores used by WeatherBench.

Grid cells near the poles cover less area than cells at the equator, so every score weights
each latitude row by the cosine of its latitude.
"""

from __future__ import annotations

import numpy as np


def latitude_weights(latitude: np.ndarray) -> np.ndarray:
    """Area weight of each latitude row, scaled so the weights average 1."""
    weights = np.cos(np.deg2rad(latitude))
    return weights / weights.mean()


def spatial_mse(forecast: np.ndarray, truth: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    """Area-weighted mean squared error over the grid, one value per time step.

    Inputs have shape (..., latitude, longitude) and may broadcast against each other.
    """
    weights = latitude_weights(latitude)[:, None]
    error = np.asarray(forecast, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return (weights * error**2).mean(axis=(-2, -1))


def rmse(forecast: np.ndarray, truth: np.ndarray, latitude: np.ndarray) -> float:
    """Root of the weighted squared error averaged over the grid and over all time steps.

    This is what the WeatherBench scoring code computes: average first, then take the root.
    """
    return float(np.sqrt(spatial_mse(forecast, truth, latitude).mean()))


def mean_rmse(forecast: np.ndarray, truth: np.ndarray, latitude: np.ndarray) -> float:
    """Average of the per-time-step RMSEs, the formula printed in the WeatherBench paper.

    Never larger than rmse(), because here the root is taken before averaging over time.
    """
    return float(np.sqrt(spatial_mse(forecast, truth, latitude)).mean())


def acc(
    forecast: np.ndarray, truth: np.ndarray, climatology: np.ndarray, latitude: np.ndarray
) -> float:
    """Anomaly correlation coefficient.

    Correlates how the forecast and the truth each depart from climatology. 1 is a perfect
    forecast, 0 is no better than forecasting the long-term average.
    """
    weights = latitude_weights(latitude)[:, None]
    forecast_anomaly = np.asarray(forecast, dtype=np.float64) - climatology
    truth_anomaly = np.asarray(truth, dtype=np.float64) - climatology
    forecast_anomaly = forecast_anomaly - forecast_anomaly.mean()
    truth_anomaly = truth_anomaly - truth_anomaly.mean()
    covariance = (weights * forecast_anomaly * truth_anomaly).sum()
    scale = np.sqrt((weights * forecast_anomaly**2).sum() * (weights * truth_anomaly**2).sum())
    return float(covariance / scale)
