
import torch

from final_project_4.src.boxes import box_area 

def flatten_predictions(predictions: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    # 建立class_logits、box_regression和centerness_logits的空列表
    all_class_logits= []
    all_box_regression = []
    all_centerness_logits = []
    # 對每個特徵層flatten預測張量
    for level in ("p3", "p4", "p5"):
        
        class_level = predictions[level]["class_logits"]
        box_level = predictions[level]["box_regression"]
        centerness_level = predictions[level]["centerness_logits"]
        
        B, C, H, W = class_level.shape
        
        # 將每個層級的預測張量從[B, C, H, W]轉換為[B, H*W, C]，並將box_level和centerness_level也進行相同的轉換
        class_level = class_level.permute(0, 2, 3, 1).reshape(B, H*W, C)
        box_level = box_level.permute(0, 2, 3, 1).reshape(B, H*W, 4)
        centerness_level = centerness_level.permute(0, 2, 3, 1).reshape(B, H*W)
        # 將每個層級的flatten預測張量添加到對應的列表中
        all_class_logits.append(class_level)
        all_box_regression.append(box_level)
        all_centerness_logits.append(centerness_level)
        # 將所有層級的flatten預測張量沿著第二個維度（H*W）進行串接，並返回一個包含串接後的class_logits、box_regression和centerness_logits的字典
    return {
        "class_logits": torch.cat(all_class_logits, dim=1),
        "box_regression": torch.cat(all_box_regression, dim=1),
        "centerness_logits": torch.cat(all_centerness_logits, dim=1)
    }
def generate_locations(predictions: dict[str, dict[str, torch.Tensor]], stride: dict[str, int])-> tuple[torch.Tensor, torch.Tensor]:
    # 建立空列表來存儲所有層級的locations和對應的stride
    all_locations = []
    all_strides = []
    # 對每個特徵層生成locations和對應的stride
    for level in ("p3", "p4", "p5"):
        class_logits = predictions[level]["class_logits"]        
        #獲取每個層級的預測張量的高度和寬度
        B, C, H, W = class_logits.shape
        #每層級的stride
        level_stride = stride[level]
        device = class_logits.device
        dtype = class_logits.dtype
        #生成每個層級對應到原圖的網格坐標
        xs = (torch.arange(W, device=device, dtype=dtype) + 0.5) * level_stride
        ys = (torch.arange(H, device=device, dtype=dtype) + 0.5) * level_stride
        #使用meshgrid生成每個層級的網格坐標
        #ex:(H, W) = (2, 3) xs = [0.5, 1.5, 2.5] ys = [0.5, 1.5]
        #grid_x = [[0.5, 1.5, 2.5],
        #          [0.5, 1.5, 2.5]]
        #grid_y = [[0.5, 0.5, 0.5],
        #          [1.5, 1.5, 1.5]]
        #(grid_x, grid_y) = [[(0.5, 0.5), (1.5, 0.5), (2.5, 0.5)],
        #                 [(0.5, 1.5), (1.5, 1.5), (2.5, 1.5)]]
        grid_y, grid_x = torch.meshgrid(
            ys,
            xs,
            indexing="ij",
        )
        # 將每個層級的網格坐標展平，並將其堆疊成一個形狀為(H*W, 2)的張量，其中第一列是x坐標，第二列是y坐標
        #ex: level_locations = [[0.5, 0.5],
        #                      [1.5, 0.5],
        #                      [2.5, 0.5],
        #                      [0.5, 1.5],
        #                      [1.5, 1.5],
        #                      [2.5, 1.5]]
        level_locations = torch.stack(
            [
                grid_x.reshape(-1),
                grid_y.reshape(-1),
            ],
            dim=1,
        )
        # ex: p3_level_strides = [8, 8 .. 8],共有H*W個8
        #     p4_level_strides = [16, 16 .. 16],共有H*W個16
        #     p5_level_strides = [32, 32 .. 32],共有H*W個32
        level_strides = torch.full(
            (H*W,), fill_value=level_stride, device=device, dtype=dtype)
        
        all_locations.append(level_locations)
        all_strides.append(level_strides)
    #return所有層級的locations和對應的stride，將它們沿著第一個維度（H*W）進行串接
    return torch.cat(all_locations, dim=0), torch.cat(all_strides, dim=0)

def compute_ltrb_targets(locations: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    # locations: [L, 2]，表示每個位置的(x, y)坐標
    # boxes: [N, 4]，表示每個邊界框的(x_min, y_min, x_max, y_max)坐標
    x = locations[:, 0] #ex x=[4, 20, 14, 8, 16, 32...] 共有L個位置
    y = locations[:, 1]
    xmin = boxes[:, 0] #ex xmin=[0, 10, 10, 0, 8, 24...] 共有N個邊界框
    ymin = boxes[:, 1]
    xmax = boxes[:, 2]
    ymax = boxes[:, 3]
    # 計算每個位置到對應每個邊界框的左、上、右、下距離
    #              box 0          box 1
    # location 0   [l,t,r,b]      [l,t,r,b]
    # location 1   [l,t,r,b]      [l,t,r,b]
    #              xmin
    #              0     10
    #           ┌─────┬─────┐
    # x = 4     │4-0  │4-10 │
    #           ├─────┼─────┤
    # x = 20    │20-0 │20-10│
    #           └─────┴─────┘
    l = x[:,None] - xmin[None,:]  # 轉成[L, 1] - [1, N]，得到[L, N]，表示每個位置到每個邊界框的左距離
    t = y[:,None] - ymin[None,:]  # 上距離
    r = xmax[None,:] - x[:,None]  # 右距離
    b = ymax[None,:] - y[:,None]  # 下距離
    return torch.stack([l, t, r, b], dim=-1) # 返回每個(location, box)對應的(l, t, r, b)形狀為[L, N, 4]的張量

def compute_inside_box_mask(ltrb_targets: torch.Tensor) -> torch.Tensor:
    # 檢查每個位置ltrb是否>0 在對應的邊界框內，若在邊界框內則返回True，否則返回False
    # return 格式為[L, N]，表示每個位置對應每個邊界框是否在邊界框內
    return (ltrb_targets > 0).all(dim=-1)


def match_locations_to_boxes(inside_mask: torch.Tensor,boxes: torch.Tensor,) -> tuple[torch.Tensor, torch.Tensor]:
    #inside_mask
    #         box0   box1
    # loc0     T      T
    # loc1     T      F
    # loc2     F      F
    
    boxes_area = box_area(boxes)  # [N]，計算每個邊界框的面積
    L = inside_mask.shape[0]
    # 如果沒有邊界框，則返回所有位置的匹配索引為-1，表示沒有匹配到任何邊界框，並且positive_mask為全False
    if boxes.shape[0] == 0:
        matched_gt_indices = torch.full((L,),-1, dtype=torch.int64, device=inside_mask.device)
        positive_mask = torch.zeros((L,),dtype=torch.bool,device=inside_mask.device)
        # N == 0 時 candidate_areas 會是 [L,0]，
        # 無法對 dim=1 做 min，因此直接回傳全部 background。
        return matched_gt_indices, positive_mask
    
    # boxes_area: [N]
    # boxes_area[None, :]: [1, N]
    # torch.where 透過 broadcasting 將結果產生為 [L, N]
    # inside_mask           broadcast to [L, N]
    #     ↓                        ↓  
    # [T, T]                    [100, 36]
    # [T, T]                    [100, 36]
    # [T, T]                    [100, 36]
    # if inside_mask[i, j] is True, then candidate_areas[i, j] = boxes_area[j] 
    # if inside_mask[i, j] is False, then candidate_areas[i, j] = inf
    #     candidate_areas
    #         box0   box1
    # loc0    100     36
    # loc1    100    inf
    # loc2    inf    inf
    # # 建立 shape [1,N]、dtype/device 與 boxes_area 相同的張量，並將其填充為無限大
    inf_areas = torch.full_like(boxes_area[None, :], float("inf"))
    # 把不在邊界框內的候選位置的面積設為無限大，這樣在後續選擇最小面積時就不會選到這些位置
    candidate_areas = torch.where(inside_mask, boxes_area[None, :], inf_areas)
    # 找到每個位置對應的最小面積和對應的邊界框索引
    min_areas, matched_gt_indices = candidate_areas.min(dim=1)
    # 確認哪些位置有匹配到box，若最小面積為有限值，則表示該位置有匹配到box設為True，否則表示該位置沒有匹配到box設為False
    positive_mask = torch.isfinite(min_areas)
    # 將沒有匹配到box的位置的索引設為-1，表示該位置沒有匹配到任何box
    matched_gt_indices = matched_gt_indices.masked_fill(~positive_mask, -1)
    # matched_gt_indices[i] == -1 表示 background
    # positive_mask[i] == True 表示該 location 成功匹配到一個 box
    return matched_gt_indices, positive_mask




def compute_regression_range_mask(ltrb_targets: torch.Tensor, location_strides: torch.Tensor, regression_ranges: dict[int, tuple[float, float]]) -> torch.Tensor:
    # ltrb_targets: [L, N, 4]，表示每個位置對應每個邊界框的左、上、右、下距離
    # location_strides: [L]，表示每個位置對應的stride
    # regression_ranges:
    # {
    #     8: (0, 64),
    #     16: (64, 128),
    #     32: (128, inf),
    # }
    # 計算每個位置到對應每個邊界框的最大距離
    max_distances = ltrb_targets.max(dim=-1).values  # [L, N]
    # 建立一個與max_distances形狀相同的布林張量，初始值為False，表示每個位置是否在對應的回歸範圍內
    range_mask = torch.zeros_like(max_distances, dtype=torch.bool)  # [L, N]
    # p3: stride=8, regression_range=(0, 64]
    # p4: stride=16, regression_range=(64, 128]
    # p5: stride=32, regression_range=(128, inf]
    for stride, (lower, upper) in regression_ranges.items():
        # level_mask: [L]，表示每個位置是否對應到當前的stride 
        # distance_mask: [L, N]，表示每個location對應每個boxes框的最大距離是否在當前的回歸範圍內
        # range_mask: [L, N]，表示每個位置對應每個邊界框是否在當前的回歸範圍內
        level_mask = (location_strides == stride) 
        distance_mask = ((max_distances > lower) & (max_distances <= upper))
        #  ex: level_mask = [T, T, F]，表示哪些location屬於當前的stride(p3, p4, p5)
        #      level_mask[:, None] = [[T], 
        #                             [T],
        #                             [F]]將其擴展為[L, 1]的形狀和distance_mask broadcasting
        #      distance_mask = [[T, F, T, F]]
        #      and起來
        #             box
        #          0    1    2    3
        #     ┌────┬────┬────┬────┐
        # loc 0  │ T  │ F  │ T  │ F  │
        # loc 1  │ T  │ F  │ T  │ F  │
        # loc 2  │ F  │ F  │ F  │ F  │
        #     └────┴────┴────┴────┘
        range_mask |= (level_mask[:, None] & distance_mask)
        # p3 p4 p5都跑過只要有任何一個stride符合條件且在回歸範圍內，就會被設為True，否則就是False
        # range_mask只在乎「這個 location 能不能用？」 「每個 location 有哪些 box 可以選？」
    return range_mask

def compute_centerness_targets(box_targets: torch.Tensor,positive_mask: torch.Tensor) -> torch.Tensor:
    #建立L個位置的centerness targets，初始值為0，表示每個位置的centerness targets
    centerness_targets = torch.zeros((box_targets.size(0),), dtype=box_targets.dtype, device=box_targets.device)
    # box_targets: [L, 4]，表示每個位置對應每個邊界框的左、上、右、下距離
    # positive_mask: [L]，表示每個位置是否匹配到一個邊界框
    # 只計算匹配到邊界框的位置的centerness targets共有P個位置
    pos_boxes = box_targets[positive_mask]
    left_right = pos_boxes[..., [0, 2]]   # [P, 2]，取出左距離和右距離
    top_bottom = pos_boxes[..., [1, 3]]   # [P, 2]，取出上距離和下距離
    # 計算centerness targets，公式為 sqrt((min(l,r)/max(l,r)) * (min(t,b)/max(t,b)))
    pos_centerness= torch.sqrt(
        (left_right.min(dim=-1).values / left_right.max(dim=-1).values) *
        (top_bottom.min(dim=-1).values / top_bottom.max(dim=-1).values)
    )
    centerness_targets[positive_mask] = pos_centerness
    
    return centerness_targets


#處理一張圖片的targets，將每個位置對應的正確答案邊界框和正確標籤分配給每個位置
def assign_targets_to_locations(
    locations: torch.Tensor,  #[L, 2]
    locations_strides: torch.Tensor, #[L]
    boxes: torch.Tensor,  #[N, 4]
    labels: torch.Tensor,   #[N]
    regression_ranges: dict[int, tuple[float, float]], 
) -> dict[str, torch.Tensor]:
    # 計算每個位置對應每個邊界框的左、上、右、下距離 
    ltrb_targets = compute_ltrb_targets(locations, boxes) #[L, N, 4]
    # 計算每個位置是否在對應的邊界框內 bool
    inside_mask = compute_inside_box_mask(ltrb_targets) #[L, N]
    # 計算每個位置對應每個邊界框是否在回歸範圍內 bool
    range_mask = compute_regression_range_mask(ltrb_targets, locations_strides, regression_ranges) #[L, N]
    # 只有同時滿足在邊界框內和在回歸範圍內的位置才會被視為候選位置 bool 
    candidate_mask = inside_mask & range_mask #[L, N]
    # 將候選位置匹配到對應的邊界框，返回matched_indices表示每個位置匹配到的邊界框索引，positive_mask表示每個位置是否匹配到一個邊界框
    matched_gt_indices, positive_mask = match_locations_to_boxes(candidate_mask, boxes) #[L], [L]
    # 建立每個位置的目標標籤，初始值為-1，表示每個位置的目標標籤
    target_labels = torch.full((locations.size(0),), fill_value=-1, dtype=labels.dtype, device=labels.device) # [L]
    # 建立每個位置的目標邊界框，初始值為0，表示每個位置的目標邊界框 
    target_boxes = torch.zeros((locations.size(0), 4), dtype=ltrb_targets.dtype, device=ltrb_targets.device) # [L, 4]
    # 只對匹配到邊界框的位置進行目標標籤和目標邊界框的賦值
    positive_indices = torch.where(positive_mask)[0]  #ex: positive_mask = [T, T, F, T, F] -> positive_indices = [0, 1, 3]
    matched_indices = matched_gt_indices[positive_indices] #ex : matched_gt_indices = [2, 5, 1, 2, 0] -> matched_indices = [2, 5, 2]
    # 從mached_indices=[2, 5, 2]去找其對應的labels=[7, 11, 3]，放回target_labels的positive_indices=[0, 1, 3]位置
    target_labels[positive_indices] = labels[matched_indices] 
    # 從mached_indices=[2, 5, 2]去找其對應的ltrb_targets[0,2],ltrb_targets[1,5],ltrb_targets[3,2]，放回target_boxes的positive_indices=[0, 1, 3]位置
    target_boxes[positive_indices] = ltrb_targets[positive_indices, matched_indices]
    # 計算每個位置的centerness targets
    centerness_targets = compute_centerness_targets(target_boxes, positive_mask)
    return {
        "labels": target_labels,                    # [L]
        "box_targets": target_boxes,                # [L,4]
        "centerness_targets": centerness_targets,  # [L]
        "positive_mask": positive_mask,             # [L]
        "matched_gt_indices": matched_gt_indices,  # [L]
    }   