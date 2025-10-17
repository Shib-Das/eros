import pytest
from PIL import Image
import numpy as np
from pathlib import Path

from eros.processor import resize_image, normalize_image, preprocess_image
from eros.error import ImageError

@pytest.fixture
def landscape_image():
    return Image.new('RGB', (800, 600))

@pytest.fixture
def portrait_image():
    return Image.new('RGB', (600, 800))

@pytest.fixture
def square_image():
    return Image.new('RGB', (600, 600))

@pytest.fixture
def grayscale_image():
    return Image.new('L', (100, 100))

def test_resize_image_landscape(landscape_image):
    resized = resize_image(landscape_image, 400)
    assert resized.size == (400, 300)

def test_resize_image_portrait(portrait_image):
    resized = resize_image(portrait_image, 400)
    assert resized.size == (300, 400)

def test_resize_image_square(square_image):
    resized = resize_image(square_image, 300)
    assert resized.size == (300, 300)

def test_normalize_image_rgb(square_image):
    normalized = normalize_image(square_image)
    assert isinstance(normalized, np.ndarray)
    assert normalized.shape == (3, 600, 600)
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
    assert normalized.dtype == np.float32

def test_normalize_image_grayscale(grayscale_image):
    normalized = normalize_image(grayscale_image)
    assert isinstance(normalized, np.ndarray)
    assert normalized.shape == (3, 100, 100) # Should be converted to RGB
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
    assert normalized.dtype == np.float32

def test_preprocess_image_valid():
    # Assuming tests/image.jpg exists from the setup step
    image_path = Path("tests/image.jpg")
    preprocessed = preprocess_image(str(image_path), 224)
    assert isinstance(preprocessed, np.ndarray)
    assert preprocessed.shape[0] == 3 # 3 channels
    assert preprocessed.dtype == np.float32

def test_preprocess_image_invalid_path():
    with pytest.raises(ImageError):
        preprocess_image("non_existent_image.jpg", 224)
