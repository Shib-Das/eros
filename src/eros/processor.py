from PIL import Image
import numpy as np
from .error import ImageError

def resize_image(
    image: Image.Image,
    size: int,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> Image.Image:
    """Resizes an image to the given size, maintaining aspect ratio."""
    width, height = image.size
    if width > height:
        new_width = size
        new_height = int(height * (size / width))
    else:
        new_height = size
        new_width = int(width * (size / height))

    return image.resize((new_width, new_height), resample=resample)

def normalize_image(image: Image.Image) -> np.ndarray:
    """Normalizes an image to the range [0, 1] and transposes it to [C, H, W]."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    img_array = np.array(image, dtype=np.float32) / 255.0
    return np.transpose(img_array, (2, 0, 1))

def preprocess_image(
    image_path: str,
    size: int,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> np.ndarray:
    """Preprocesses an image for the ONNX model."""
    try:
        with Image.open(image_path) as img:
            resized_img = resize_image(img, size, resample)
            return normalize_image(resized_img)
    except IOError as e:
        raise ImageError(f"Error processing image: {e}") from e
