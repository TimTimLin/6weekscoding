import numpy as np

def normalize_images(images: np.ndarray) -> np.ndarray:
    """
    Normalize the pixel values of images to the range [0, 1].

    Parameters:
        images (np.ndarray): Input images as a NumPy array.

    Returns:
        np.ndarray: Normalized images with pixel values in the range [0, 1].
    """
    if not isinstance(images, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    if images.dtype != np.uint8:
        raise TypeError("Input images must have dtype uint8.")

    if images.ndim != 4:
        raise ValueError(
            "Input must have shape (N, H, W, C)."
        )

    if images.shape[-1] not in (1, 3):
        raise ValueError(
            "The last dimension must be 1 or 3."
        )

    return images.astype(np.float32) / 255.0    
def center_images(images: np.ndarray) -> np.ndarray:
    """
    Subtract each channel mean from an NHWC image batch.
    """
    
    if not isinstance(images, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    if images.ndim != 4:
        raise ValueError(
            "Input must have shape (N, H, W, C)."
        )
    # Compute the mean for each channel(r, g, b) across all images, height, and width 
    channel_mean = np.mean(images, axis=(0,1, 2), keepdims=True)
    # channel_mean[0]=np.mean(images[:,:,:,0]) #mean of red channel
    # channel_mean[1]=np.mean(images[:,:,:,1]) #mean of green channel
    # channel_mean[2]=np.mean(images[:,:,:,2]) #mean of blue channel
    return images - channel_mean