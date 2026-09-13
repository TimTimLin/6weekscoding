import numpy as np
import torch
import pytest
from final_project_5.src.transforms import resize_image_and_mask, image_and_mask_to_tensor , SegmentationTransform
from PIL import Image
def test_resize_image_and_mask():
    image_array = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )

    mask_array = np.array(
        [
            [1, 2],
            [3, 1],
        ],
        dtype=np.uint8,
    )
    # 將 numpy array 轉換為 PIL Image 來測試 resize_image_and_mask 函數
    image = Image.fromarray(image_array)
    raw_mask = Image.fromarray(mask_array)
    resized_image, resized_mask = resize_image_and_mask(image, raw_mask, (4, 6))
    expected_resized_mask = np.array([[1, 1, 1, 2, 2, 2],
                                      [1, 1, 1, 2, 2, 2],
                                        [3, 3, 3, 1, 1, 1],
                                        [3, 3, 3, 1, 1, 1]], dtype=np.uint8)
    assert resized_image.size == (6, 4)
    assert resized_mask.size == (6, 4)

    assert resized_image.mode == "RGB"
    assert resized_mask.mode == "L"

    assert np.asarray(resized_image).shape == (4, 6, 3)
    assert np.asarray(resized_mask).shape == (4, 6)
    assert np.asarray(resized_mask).dtype == np.uint8

    assert np.array_equal(
        np.asarray(resized_mask),
        expected_resized_mask,
    )
    
def test_image_and_mask_to_tensor():
    image_array = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    mask_array = np.array(
        [
            [1, 2],
            [3, 1],
        ],
        dtype=np.uint8,
    )
    image = Image.fromarray(image_array)
    raw_mask = Image.fromarray(mask_array)  
    image_tensor, mask_tensor = image_and_mask_to_tensor(image, raw_mask)
    expected_image = torch.tensor(
        [
            [[1.0, 0.0],
            [0.0, 1.0]],

            [[0.0, 1.0],
            [0.0, 1.0]],

            [[0.0, 0.0],
            [1.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    expected_mask = torch.tensor(
        [
            [0, 1],
            [2, 0],
        ],
        dtype=torch.int64,
    )
    assert type(image_tensor) == torch.Tensor
    assert type(mask_tensor) == torch.Tensor
    assert image_tensor.shape == (3, 2, 2)
    assert mask_tensor.shape == (2, 2)
    assert image_tensor.dtype == torch.float32
    assert mask_tensor.dtype == torch.int64
    torch.testing.assert_close(image_tensor, expected_image, rtol=1e-5, atol=1e-8)  
    torch.testing.assert_close(mask_tensor, expected_mask, rtol=1e-5, atol=1e-8)  
    
def test_segmentation_transform():
    image_array = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    mask_array = np.array(
        [
            [1, 2],
            [3, 1],
        ],
        dtype=np.uint8,
    )
    image = Image.fromarray(image_array)
    raw_mask = Image.fromarray(mask_array)
    transform = SegmentationTransform(size=(2, 2), horizontal_flip_prob=1.0)  #必翻轉   
    transformed_image, transformed_mask = transform(image, raw_mask)
    expected_mask = torch.tensor(
        [
            [1, 0],
            [0, 2],
        ],
        dtype=torch.int64,
    )
    expected_image = torch.tensor(
        [
            [[0.0, 1.0],
            [1.0, 0.0]],

            [[1.0, 0.0],
            [1.0, 0.0]],

            [[0.0, 0.0],
            [1.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    
    assert transformed_image.dtype == torch.float32
    assert transformed_mask.dtype == torch.int64
    assert transformed_image.shape == (3, 2, 2)
    assert transformed_mask.shape == (2, 2)
    torch.testing.assert_close(transformed_image, expected_image, rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(transformed_mask, expected_mask, rtol=1e-5, atol=1e-8)
def test_segmentation_transform_no_flip():
      image_array = np.array(
          [
              [[255, 0, 0], [0, 255, 0]],
              [[0, 0, 255], [255, 255, 255]],
          ],
          dtype=np.uint8,
      )
      mask_array = np.array(
          [
              [1, 2],
              [3, 1],
          ],
          dtype=np.uint8,
      )
      image = Image.fromarray(image_array)
      raw_mask = Image.fromarray(mask_array)
      transform = SegmentationTransform(size=(2, 2), horizontal_flip_prob=0.0)  #必不翻轉
      transformed_image, transformed_mask = transform(image, raw_mask)
      expected_image = torch.tensor(
          [
              [[1.0, 0.0],
              [0.0, 1.0]],

              [[0.0, 1.0],
              [0.0, 1.0]],

              [[0.0, 0.0],
              [1.0, 1.0]],
          ],
          dtype=torch.float32,
      )
      expected_mask = torch.tensor(
          [
              [0, 1],
              [2, 0],
          ],
          dtype=torch.int64,
      )
      torch.testing.assert_close(transformed_image, expected_image)
      torch.testing.assert_close(transformed_mask, expected_mask)