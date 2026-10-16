"""
Krishi Rakshak AI - Preprocessing Module
Team: Codex Verse
Description: Robust image validation, resizing, normalization, and tensor formatting
             for MobileNetV2 crop leaf disease inference.
"""

import io
from pathlib import Path
from typing import Tuple, Union, Optional
import numpy as np
from PIL import Image, UnidentifiedImageError

# MobileNetV2 expected image dimensions
TARGET_SIZE: Tuple[int, int] = (224, 224)
MAX_FILE_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB
MIN_DIMENSION: int = 32  # Minimum acceptable width/height in pixels
ALLOWED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}


class ImageValidationError(Exception):
    """Custom exception raised when an input image fails integrity or size checks."""
    pass


def validate_image_bytes(image_bytes: bytes) -> None:
    """
    Validate raw image bytes for size constraints and corrupt header data.
    """
    if not image_bytes or len(image_bytes) == 0:
        raise ImageValidationError("Empty image payload received.")
    
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        mb = len(image_bytes) / (1024 * 1024)
        raise ImageValidationError(f"Image file is too large ({mb:.2f} MB). Maximum allowed size is 15 MB.")


def load_and_validate_image(
    image_source: Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray]
) -> Image.Image:
    """
    Loads an image from various input types and validates its format, size, and integrity.
    
    Returns:
        PIL.Image in RGB mode.
    """
    img: Optional[Image.Image] = None

    try:
        if isinstance(image_source, (str, Path)):
            path = Path(image_source)
            if not path.exists():
                raise ImageValidationError(f"Image file does not exist: {path}")
            if path.stat().st_size > MAX_FILE_SIZE_BYTES:
                raise ImageValidationError("Image file exceeds maximum 15 MB size limit.")
            img = Image.open(path)
            
        elif isinstance(image_source, bytes):
            validate_image_bytes(image_source)
            img = Image.open(io.BytesIO(image_source))
            
        elif isinstance(image_source, io.BytesIO):
            img = Image.open(image_source)
            
        elif isinstance(image_source, Image.Image):
            img = image_source
            
        elif isinstance(image_source, np.ndarray):
            if image_source.size == 0:
                raise ImageValidationError("Received empty numpy array as image.")
            # Convert OpenCV BGR to RGB if 3-channel
            if len(image_source.shape) == 3 and image_source.shape[2] == 3:
                # Assume standard numpy RGB or uint8
                img = Image.fromarray(image_source.astype("uint8"))
            elif len(image_source.shape) == 2:
                img = Image.fromarray(image_source.astype("uint8"), mode="L")
            else:
                img = Image.fromarray(image_source.astype("uint8"))
        else:
            raise ImageValidationError(f"Unsupported image input type: {type(image_source)}")

        # Verify image integrity (catches truncated/corrupted files)
        img.verify()
        
        # PIL verify closes file pointer / invalidates image, so reload cleanly
        if isinstance(image_source, (str, Path)):
            img = Image.open(image_source)
        elif isinstance(image_source, bytes):
            img = Image.open(io.BytesIO(image_source))
        elif isinstance(image_source, io.BytesIO):
            image_source.seek(0)
            img = Image.open(image_source)
        elif isinstance(image_source, Image.Image):
            # Already an in-memory image
            img = image_source.copy()

        # Check dimensions
        width, height = img.size
        if width < MIN_DIMENSION or height < MIN_DIMENSION:
            raise ImageValidationError(
                f"Image resolution ({width}x{height}) is too small. "
                f"Minimum required is {MIN_DIMENSION}x{MIN_DIMENSION} pixels."
            )

        # Convert to RGB mode (removes Alpha channel, converts Grayscale or CMYK)
        if img.mode != "RGB":
            img = img.convert("RGB")

        return img

    except (UnidentifiedImageError, OSError) as e:
        raise ImageValidationError(f"Corrupt or unreadable image file: {str(e)}")


def preprocess_image(
    image_source: Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = TARGET_SIZE
) -> np.ndarray:
    """
    Standard preprocessing pipeline for MobileNetV2:
    1. Loads and validates image.
    2. Resizes with high-quality Lanczos/Bilinear filtering to target_size (224, 224).
    3. Converts to float32 numpy array.
    4. Normalizes pixel values to [-1.0, 1.0] via `(x / 127.5) - 1.0`.
    5. Adds batch dimension, returning shape `(1, 224, 224, 3)`.
    """
    pil_img = load_and_validate_image(image_source)
    
    # High-quality resize to 224x224
    resized_img = pil_img.resize(target_size, Image.Resampling.BILINEAR)
    
    # Convert to float32 array
    img_array = np.array(resized_img, dtype=np.float32)
    
    # MobileNetV2 normalization: scale [0, 255] -> [-1, 1]
    normalized_array = (img_array / 127.5) - 1.0
    
    # Add batch dimension -> (1, 224, 224, 3)
    batch_tensor = np.expand_dims(normalized_array, axis=0)
    
    return batch_tensor
