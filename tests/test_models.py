import torch

from weather_forecaster.models import ResNet, SimpleCNN


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


def test_resnet_maps_many_input_channels_to_fewer_outputs():
    model = ResNet(in_channels=9, out_channels=3, width=16, blocks=2)
    assert model(torch.randn(2, 9, 32, 64)).shape == (2, 3, 32, 64)


def test_an_untrained_resnet_predicts_no_change():
    model = ResNet(in_channels=9, out_channels=3, width=16, blocks=2)
    assert torch.count_nonzero(model(torch.randn(2, 9, 32, 64))) == 0


def test_rotating_the_globe_rotates_the_resnet_output():
    torch.manual_seed(0)
    model = ResNet(in_channels=4, out_channels=4, width=16, blocks=2).eval()
    torch.nn.init.normal_(model.head.conv.weight, std=0.1)  # the head starts at zero
    x = torch.randn(1, 4, 32, 64)
    shifted = torch.roll(x, shifts=17, dims=-1)
    assert torch.allclose(model(shifted), torch.roll(model(x), shifts=17, dims=-1), atol=1e-5)
