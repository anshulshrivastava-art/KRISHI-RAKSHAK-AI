"""
Unit tests for image preprocessing and validation.
"""

import io
import pytest
import numpy as np
from PIL import Image

from src.preprocessing import (
    preprocess_image,
    load_and_validate_image,
    ImageValidationError,
    TARGET_SIZE
)


def test_valid_rgb_image_preprocessing():
    # Create valid synthetic RGB image
    img = Image.new("RGB", (300, 300), color=(100, 150, 50))
    tensor = preprocess_image(img)
    
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, TARGET_SIZE[0], TARGET_SIZE[1], 3)
    assert tensor.dtype == np.float32
    # Check normalized range [-1, 1]
    assert np.all(tensor >= -1.0)
    assert np.all(tensor <= 1.0)


def test_png_with_alpha_channel_converted_to_rgb():
    # Create RGBA image
    img_rgba = Image.new("RGBA", (250, 250), color=(100, 200, 50, 255))
    buf = io.BytesIO()
    img_rgba.save(buf, format="PNG")
    buf.seek(0)
    
    loaded = load_and_validate_image(buf.getvalue())
    assert loaded.mode == "RGB"
    
    tensor = preprocess_image(buf.getvalue())
    assert tensor.shape == (1, 224, 224, 3)


def test_corrupt_image_raises_validation_error():
    corrupt_bytes = b"NOT_A_VALID_JPEG_HEADER_123456789"
    with pytest.raises(ImageValidationError):
        load_and_validate_image(corrupt_bytes)


def test_empty_bytes_raises_validation_error():
    with pytest.raises(ImageValidationError):
        load_and_validate_image(b"")


def test_tiny_resolution_raises_validation_error():
    # 10x10 is below the 32x32 minimum dimension
    tiny_img = Image.new("RGB", (10, 10), color=(50, 50, 50))
    with pytest.raises(ImageValidationError):
        load_and_validate_image(tiny_img)
