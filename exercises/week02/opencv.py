from pathlib import Path

import cv2
import numpy as np


def load_image(path: Path) -> np.ndarray:
    """Load an image file and return it in RGB format."""
    image =cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image                                                                            


def resize_image(
    image: np.ndarray,
    height: int,
    width: int, 
) -> np.ndarray:
    """Resize an RGB image."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must have shape (H, W, 3).")
    if height <= 0 or width <= 0:
        raise ValueError("Height and width must be positive integers.")
    resized_image = cv2.resize(image, (width, height))
    return resized_image


def save_image(path: Path, image: np.ndarray) -> None:
    """Save an RGB image to a file."""
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(str(path), image)
    if not success:
        raise OSError(f"Failed to save image: {path}")
    

def process_image(
    input_path: Path,
    output_path: Path,
    height: int,
    width: int,
) -> None:
    """Load, resize, and save an image."""
    image = load_image(input_path)
    resized_image = resize_image(image, height, width)
    save_image(output_path, resized_image)