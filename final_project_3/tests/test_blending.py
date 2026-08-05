import numpy as np

from final_project_3.src.blending import blend_images


def test_blend_images_regions():
    image_a = np.zeros((3, 3, 3), dtype=np.uint8)
    image_b = np.zeros((3, 3, 3), dtype=np.uint8)

    image_a[:] = [100, 0, 0]  # BGR blue
    image_b[:] = [0, 200, 0]  # BGR green

    mask_a = np.array([
        [True, True, False],
        [True, False, False],
        [False, False, False],
    ])

    mask_b = np.array([
        [False, True, True],
        [False, True, True],
        [False, False, False],
    ])

    blended = blend_images(
        image_a,
        image_b,
        mask_a,
        mask_b,
    )

    assert blended.shape == (3, 3, 3)
    assert blended.dtype == np.uint8

    # only-A
    assert np.array_equal(blended[0, 0], [100, 0, 0])

    # overlap: 0.5 * [100, 0, 0] + 0.5 * [0, 200, 0]
    assert np.array_equal(blended[0, 1], [50, 100, 0])

    # only-B
    assert np.array_equal(blended[0, 2], [0, 200, 0])

    # invalid region
    assert np.array_equal(blended[2, 2], [0, 0, 0])