import cv2
import numpy as np
import torch
from tqdm import tqdm

# --- 核心辅助函数保持不变 ---
def load_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Cannot load image: {img_path}')
    return img

def get_binary(img, threshold=127):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary

# --- 修改后的 Metric 逻辑 ---

def Metric_IoU(bin_gen, bin_gt):
    # 要求：bin_gen 和 bin_gt 必须尺寸一致
    inter = np.logical_and(bin_gen, bin_gt).sum()
    union = np.logical_or(bin_gen, bin_gt).sum()
    return 1.0 - (inter / union) if union > 0 else 1.0

def Metric_Purity(img_bgr, bin_gt):
    mask = bin_gt > 0
    if not np.any(mask): return 0.0
    std = np.std(img_bgr[mask], axis=0).mean()
    return min(std / 128.0, 1.0)

def Metric_Dist_Normalized(cnt_gen, cnt_gt, shape_gen, shape_gt):
    """
    计算归一化质心距离。基于物体在各自画布中的百分比坐标。
    """
    M_gen = cv2.moments(cnt_gen)
    M_gt = cv2.moments(cnt_gt)
    
    h_gen, w_gen = shape_gen[:2]
    h_gt, w_gt = shape_gt[:2]

    if M_gen["m00"] == 0 or M_gt["m00"] == 0: 
        return 1.0
    
    # 归一化坐标 (0.0 - 1.0)
    nx_gen, ny_gen = (M_gen["m10"]/M_gen["m00"]) / w_gen, (M_gen["m01"]/M_gen["m00"]) / h_gen
    nx_gt, ny_gt = (M_gt["m10"]/M_gt["m00"]) / w_gt, (M_gt["m01"]/M_gt["m00"]) / h_gt
    
    # 在归一化单位正方形空间计算欧氏距离，最大值为 sqrt(2)
    dist = np.sqrt((nx_gen - nx_gt)**2 + (ny_gen - ny_gt)**2)
    return min(dist / np.sqrt(2), 1.0)

def Metric_Size_Normalized(cnt_gen, cnt_gt, shape_gen, shape_gt):
    """
    比较物体占各自画布面积的比例。
    """
    area_gen_ratio = cv2.contourArea(cnt_gen) / (shape_gen[0] * shape_gen[1])
    area_gt_ratio = cv2.contourArea(cnt_gt) / (shape_gt[0] * shape_gt[1])
    
    if area_gt_ratio == 0: 
        return 1.0 if area_gen_ratio > 0 else 0.0
    return min(abs(area_gen_ratio / area_gt_ratio - 1.0), 1.0)

# --- 核心封装函数 ---

def Shape_metrics_from_img_bgr(img_gen, img_gt, threshold=127, return_mean=True):
    # 1. 提取生成图的二值化及最大轮廓
    bin_gen_raw = get_binary(img_gen, threshold)
    contours_gen, _ = cv2.findContours(bin_gen_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 2. 提取 GT 的二值化及最大轮廓
    bin_gt_raw = get_binary(img_gt, threshold)
    contours_gt, _ = cv2.findContours(bin_gt_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 预设默认值（如果没有检测到物体）
    if not contours_gen or not contours_gt:
        res = {k: 1.0 for k in ['d_iou', 'd_dist', 'd_size', 'd_shape']}
        res['d_purity'] = 0.0
    else:
        cnt_gen = max(contours_gen, key=cv2.contourArea)
        cnt_gt = max(contours_gt, key=cv2.contourArea)

        # A. 形状匹配 (cv2.matchShapes 本身具有尺度不变性)
        d_shape = min(cv2.matchShapes(cnt_gen, cnt_gt, cv2.CONTOURS_MATCH_I1, 0.0), 1.0)

        # B. 归一化距离
        d_dist = Metric_Dist_Normalized(cnt_gen, cnt_gt, img_gen.shape, img_gt.shape)

        # C. 归一化面积比
        d_size = Metric_Size_Normalized(cnt_gen, cnt_gt, img_gen.shape, img_gt.shape)

        # D. IoU (需要将 gen 缩放到 gt 的尺寸进行像素级对齐)
        h_gt, w_gt = img_gt.shape[:2]
        # 创建实心 Mask
        mask_gen = np.zeros(img_gen.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_gen, [cnt_gen], -1, 255, -1)
        mask_gt = np.zeros(img_gt.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_gt, [cnt_gt], -1, 255, -1)
        
        # 仅对 Mask 进行最近邻插值缩放，保证坐标对齐
        mask_gen_resized = cv2.resize(mask_gen, (w_gt, h_gt), interpolation=cv2.INTER_NEAREST)
        d_iou = Metric_IoU(mask_gen_resized, mask_gt)

        # E. Purity (在原始生成图上计算)
        mask_gen_final = np.zeros(img_gen.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_gen_final, [cnt_gen], -1, 255, -1)
        d_purity = Metric_Purity(img_gen, mask_gen_final)

        res = {
            'd_iou': d_iou,
            'd_dist': d_dist,
            'd_size': d_size,
            'd_shape': d_shape,
            'd_purity': d_purity
        }

    if return_mean:
        res['mean'] = sum(res.values()) / len(res)
    return res

def Shape_metrics_from_img_path(path_gen, path_gt, **kwargs):
    img_gen = load_image(path_gen)
    img_gt = load_image(path_gt)
    return Shape_metrics_from_img_bgr(img_gen, img_gt, **kwargs)

def change_list2dict(dicts):
    new_dict = dict()
    for key in dicts[0].keys():
        new_dict[key] = [d[key] for d in dicts]
    return new_dict

def dict_mean(dicts):
    new_dict = dict()
    for key in dicts.keys():
        new_dict[key] = sum(dicts[key])/len(dicts[key])
    return new_dict

def dict2tensor(dicts):
    new_dict = dict()
    for key in dicts.keys():
        new_dict[key] = torch.tensor(dicts[key])
    return new_dict

def Shape_metrics_from_img_list(list_gen, list_gt, return_each_sample=False, **kwargs):
    if len(list_gen) != len(list_gt): raise ValueError("List length mismatch")
    res = []
    for p_gen, p_gt in tqdm(zip(sorted(list_gen), sorted(list_gt)), total=len(list_gen)):
        res.append(Shape_metrics_from_img_path(p_gen, p_gt, **kwargs))
    res = change_list2dict(res)
    return res if return_each_sample else dict_mean(res)

def Shape_metrics_from_tensor(tensor1, tensor2, return_tensor=True, return_each_sample=False, **kwargs):
    if tensor1.shape != tensor2.shape: raise ValueError("Shape mismatch")
    B = tensor1.shape[0]
    res = []
    for idx in tqdm(range(B)):
        t1 = tensor2npBGR(tensor1[idx])
        t2 = tensor2npBGR(tensor2[idx])
        res.append(Shape_metrics_from_img_bgr(t1, t2, **kwargs))
    res = change_list2dict(res)
    if return_each_sample:
        return dict2tensor(res) if return_tensor else res
    else:
        return dict2tensor(dict_mean(res)) if return_tensor else dict_mean(res)


if __name__ == '__main__':
    # img1 = 'VIOLIN_v2\data\Variation_3\id_1.png'
    # # img2 = '/data1/lhy/pure_color/my_code/pure_red.png'
    # img2 = 'VIOLIN_v2\data\Variation_3\id_2.png'


    # img_list1 = [img1]*10
    # img_list2 = [img2]*10

    # img_tensor1 = torch.randn(2,3,56,56)
    # img_tensor2 = torch.randn(2,3,56,56)

    # res = Shape_metrics_from_img_path(img1, img2)
    # print(res)

    gt = 'metrics_v2/test_cases/gt.png'
    cases = ['case_dist.png', 'case_size.png', 'case_shape.png', 'case_purity.png']

    for c in cases:
        path = f'metrics_v2/test_cases/{c}'
        res = Shape_metrics_from_img_path(path, gt)
        print(f"Results for {c}:")
        for k, v in res.items():
            print(f"  {k}: {v:.4f}")
        print("-" * 20)