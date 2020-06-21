import pytest
import torch

from fourier_feature_transform import GaussianFourierFeatureTransform


def test_basic_tensor():
    x = torch.randn((1, 2, 256, 256))

    x = GaussianFourierFeatureTransform(2, 50, 10)(x)

    assert x.shape == (1, 100, 256, 256)


def test_nonsquare_tensor():
    x = torch.randn((1, 2, 256, 257))

    x = GaussianFourierFeatureTransform(2, 50, 10)(x)

    assert x.shape == (1, 100, 256, 257)


def test_one_width_height():
    x = torch.randn((1, 2, 1, 1))

    x = GaussianFourierFeatureTransform(2, 50, 10)(x)

    assert x.shape == (1, 100, 1, 1)


def test_wrong_num_dims():
    x = torch.randn((1, 2, 1))

    with pytest.raises(AssertionError) as excinfo:
        _ = GaussianFourierFeatureTransform(3, 50, 10)(x)

    assert "Expected 4D input (got 3D input)" in str(excinfo.value)


def test_mismatched_input_channels():
    x = torch.randn((1, 2, 1, 1))

    with pytest.raises(AssertionError) as excinfo:
        _ = GaussianFourierFeatureTransform(3, 50, 10)(x)

    assert "Expected input to have 3 channels (got 2 channels)" in str(excinfo.value)
