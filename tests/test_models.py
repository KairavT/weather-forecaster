import torch

from weather_forecaster.models import SimpleCNN


def test_output_has_the_same_shape_as_the_input():
    model = SimpleCNN(channels=2)
    assert model(torch.randn(3, 2, 32, 64)).shape == (3, 2, 32, 64)


def test_parameter_count_matches_the_published_baseline():
    # Rasp et al. 2020 report 313,858 parameters for their CNN
    model = SimpleCNN(channels=2)
    assert sum(p.numel() for p in model.parameters()) == 313_858


def test_rotating_the_globe_rotates_the_forecast():
    # longitude wraps, so shifting the input east must shift the output by the same amount
    torch.manual_seed(0)
    model = SimpleCNN(channels=2).eval()
    x = torch.randn(1, 2, 32, 64)
    shifted = torch.roll(x, shifts=17, dims=-1)
    assert torch.allclose(model(shifted), torch.roll(model(x), shifts=17, dims=-1), atol=1e-5)
