import numpy as np


def blend_images(image_a, image_b, mask_a, mask_b):
    #將image_a和image_b轉換為float32，方便後續計算
    image_a = np.asarray(image_a, dtype=np.float32)
    image_b = np.asarray(image_b, dtype=np.float32)
    #將mask_a和mask_b轉換為布林值，方便後續計算
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    #A的有效區域&B的無效區域
    only_a = mask_a & ~mask_b
    #B的有效區域&A的無效區域
    only_b = mask_b & ~mask_a
    #A和B的有效區域
    overlap = mask_a & mask_b
    #建立一個空的blended圖像，大小與image_a也就是和canvas相同，dtype為float32
    blended = np.zeros_like(image_a, dtype=np.float32)
    #把image_a的有效區域放到blended中，image_b的有效區域放到blended中，A和B的有效區域取平均值放到blended中
    blended[only_a] = image_a[only_a]
    blended[only_b] = image_b[only_b]

    blended[overlap] = (
        0.5 * image_a[overlap]
        + 0.5 * image_b[overlap]
    )
    #將blended的值限制在0~255之間，並四捨五入，最後轉換為uint8
    blended = np.clip(np.rint(blended), 0, 255)
    return blended.astype(np.uint8)

