import numpy as np
import torch
import pytest
from final_project_5.src.data import remap_trimap, split_trainval_indices

def test_remap_trimap():
    # 建立一個模擬的 raw_mask，包含 foreground (1), background (2), not classified (3)
    raw_mask = np.array([[1, 2, 3],
                         [3, 1, 2]], dtype=np.uint8)

    # 將 numpy array 轉換為 PIL Image
    from PIL import Image
    raw_mask_image = Image.fromarray(raw_mask)

    # 呼叫 remap_trimap 函數
    remapped_tensor = remap_trimap(raw_mask_image)

    # 預期的 remapped 結果
    expected_remapped = np.array([[0, 1, 2],
                                   [2 ,0 ,1]], dtype=np.int64)

    # 將預期結果轉換為 torch tensor
    expected_tensor = torch.from_numpy(expected_remapped)

    assert isinstance(remapped_tensor, torch.Tensor), "Output should be a torch tensor"
    assert remapped_tensor.dtype == torch.int64, "Output tensor should have dtype torch.int64"
    assert remapped_tensor.shape == (2, 3), "Output tensor should have the same shape as the input mask"
    assert torch.equal(remapped_tensor, expected_tensor), "The remapped tensor does not match the expected output"
    
    
def test_remap_trimap_with_invalid_values():
    # 建立一個包含無效值的 raw_mask
    raw_mask = np.array([[1, 4]],  # 4 is an invalid value
                         dtype=np.uint8)

    from PIL import Image
    raw_mask_image = Image.fromarray(raw_mask)

    with pytest.raises(ValueError):
        # 預期 remap_trimap 函數會引發 ValueError，因為輸入包含無效值
        remap_trimap(raw_mask_image)    
        
def test_split_trainval_indices():
    num_samples = 10
    val_fraction = 0.2
    seed = 42

    train_indices, val_indices = split_trainval_indices(num_samples, val_fraction, seed)

    train_indices2, val_indices2 = split_trainval_indices(num_samples, val_fraction, seed)
    
    assert len(train_indices) == 8, "Train indices length is incorrect"
    assert len(val_indices) == 2, "Validation indices length is incorrect"
    assert set(train_indices).isdisjoint(set(val_indices)), "Train and validation indices should be disjoint"
    assert set(train_indices).union(set(val_indices)) == set(range(10)), "Train and validation indices should cover all samples"
    assert train_indices == train_indices2, "Train indices should be the same for the same seed"
    assert val_indices == val_indices2, "Validation indices should be the same for the same seed"