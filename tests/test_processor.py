from PIL import Image
import numpy as np
import pytest
from eros.processor import resize_image, normalize_image

@pytest.fixture
def create_image():
    """Creates a dummy image for testing."""
    return Image.new("RGB", (100, 200))

def test_resize_image(create_image: Image.Image):
    """Tests that resize_image correctly resizes an image."""
    resized = resize_image(create_image, 50)
    assert resized.size == (25, 50)

def test_normalize_image(create_image: Image.Image):
    """Tests that normalize_image correctly normalizes an image."""
    normalized = normalize_image(create_image)
    assert normalized.shape == (3, 200, 100)
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
