import numpy as np
from opencv import load_image, save_image, process_image

def test_process_image_pipeline(tmp_path):
    # Arrange: 準備輸入
    input_path = tmp_path / "input_image.png"
    output_path = tmp_path / "output_image.png"
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    image[:, :, 0] = 255  # Red channel
    # Act: 呼叫函式
    save_image(input_path, image)
    process_image(input_path, output_path, height=8, width=12)
    
    # Assert: 驗證結果
    assert output_path.exists()
    output_image = load_image(output_path)
    assert output_image.dtype == np.uint8
    assert output_image.shape == (8, 12, 3)
    assert np.all(output_image[:, :, 0] == 255)  # Red channel should still be 255
    assert np.all(output_image[:, :, 1] == 0)    # Green channel should still be 0
    assert np.all(output_image[:, :, 2] == 0)    # Blue channel should still be 0
