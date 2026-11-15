import numpy as np
import pytest

from weather_forecaster.dataset import fit_normalizer


def test_normalized_training_data_has_zero_mean_and_unit_spread():
    rng = np.random.default_rng(0)
    fields = np.stack(
        [rng.normal(54000, 3000, (50, 8, 16)), rng.normal(270, 15, (50, 8, 16))], axis=1
    ).astype(np.float32)
    normalized = fit_normalizer(fields).normalize(fields)
    assert normalized.mean(axis=(0, 2, 3)) == pytest.approx([0, 0], abs=1e-3)
    assert normalized.std(axis=(0, 2, 3)) == pytest.approx([1, 1], abs=1e-3)


def test_denormalize_undoes_normalize():
    rng = np.random.default_rng(1)
    fields = rng.normal(270, 15, (10, 2, 8, 16)).astype(np.float32)
    normalizer = fit_normalizer(fields)
    assert normalizer.denormalize(normalizer.normalize(fields)) == pytest.approx(fields, abs=1e-2)
