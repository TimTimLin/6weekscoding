from email.mime import image

import numpy as np
import pytest

from image_utils import normalize_images
from image_utils import center_images
#pytest 會自動尋找名稱以 test_ 開頭的 function，然後執行它。所以你不需要自己寫：
def test_rgb_image_is_normalized():
    # Arrange：準備輸入
    image = np.array(
        [[[[0, 128, 255]]]],
        dtype=np.uint8,
    )

    # Act：呼叫函式
    result = normalize_images(image)

    # Assert:如果是 True，測試通過；如果是 False，測試失敗。
    assert result.dtype == np.float32
    assert result.shape == image.shape
    assert np.allclose(  #是否足夠接近
        result[0, 0, 0], #第一張圖片的第一列的第一欄的像素正規化後的RGB
        [0.0, 128 / 255, 1.0],
    )
def test_input_is_not_modified():
    image = np.array(
        [[[[0, 128, 255]]]],
        dtype=np.uint8,
    )
    original = image.copy()
    # Act：呼叫函式
    normalize_images(image)
    # Assert:original 與 image 是否相等，若相等，測試通過；若不相等，測試失敗。
    np.testing.assert_array_equal(image, original)
def test_reject_non_4d_input():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    # 用 pytest.raises 來檢查若3維參數執行正規化是否拋出 ValueError 若拋出 ValueError，測試通過；若沒有拋出 ValueError，測試失敗。
    with pytest.raises(ValueError):
        normalize_images(image)


def test_reject_invalid_channels():
    # 用 pytest.raises 來檢查若最後一維不是1或3的參數執行正規化是否拋出 ValueError 若拋出 ValueError，測試通過；若沒有拋出 ValueError，測試失敗。
    image = np.zeros((2, 4, 4, 4), dtype=np.uint8)

    with pytest.raises(ValueError):
        normalize_images(image)

def test_grayscale_image():
    # 建立一個形狀為 (2, 4, 4, 1) 的灰階圖像，像素值全為 0
    image = np.zeros((2, 4, 4, 1), dtype=np.uint8)

    result = normalize_images(image)

    assert result.shape == image.shape
    assert result.dtype == np.float32
    assert np.all(result == 0.0)
def test_center_images():
    # 建立一個形狀為 (1, 3, 3, 3) 的測試圖像，像素值為 0 到 26
    # image[0, 0, 0] = [0, 1, 2] r=0 g=1 b=2
    image = np.array(
        [
            [
                [[ 0,  1,  2], [ 3,  4,  5], [ 6,  7,  8]],
                [[ 9, 10, 11], [12, 13, 14], [15, 16, 17]],
                [[18, 19, 20], [21, 22, 23], [24, 25, 26]]
            ]
        ],
        dtype=np.float32,
    )
    original_image = image.copy()
    # R channel : [[ 0,  3,  6],
    #               [ 9, 12, 15],
    #                [18, 21, 24]]
    # avg = (0+3+6+9+12+15+18+21+24)/9 = 108/9 = 12
    # centered R channel : [[-12, -9, -6],
    #                      [-3, 0, 3],
    #                      [6, 9, 12]]
    expected_centered_image = np.array(
        [
            [
                [[-12, -12, -12], [-9, -9, -9], [-6, -6, -6]],
                [[-3, -3, -3], [0, 0, 0], [3, 3, 3]],
                [[6, 6, 6], [9, 9, 9], [12, 12, 12]],
            ]
        ],
        dtype=np.float32,
    )
    result = center_images(image)
    assert result.shape == image.shape 
    assert result.dtype == np.float32
    assert np.allclose(result, expected_centered_image)
    assert np.allclose(result.mean(axis=(0, 1, 2)), [0.0, 0.0, 0.0]) #做完中心化後，每個通道的平均值應該為0
    np.testing.assert_array_equal(image, original_image)
def test_center_images_rejects_non_array():
    #非陣列的圖像應該會拋出 TypeError
    with pytest.raises(TypeError):
        center_images([[1, 2, 3]])


def test_center_images_rejects_non_4d():
    #非4維的圖像應該會拋出 ValueError
    image = np.zeros((3, 3, 3), dtype=np.float32)

    with pytest.raises(ValueError):
        center_images(image)