"""
CGL负样本挖掘模块
使用模板匹配在真实目标邻域内挖掘高亮杂波作为负样本
"""
import cv2
import numpy as np
import torch


def compute_cgl_negative_samples(img_tensor, labels, neighborhood_scale=3.0, top_k=3):
    """
    计算CGL负样本
    
    Args:
        img_tensor: 图像张量，格式为 (3, H, W)，RGB格式
        labels: 标签张量，格式为 (N, 6)，每行为 [image_id, class, x_center, y_center, width, height] (归一化坐标)
        neighborhood_scale: 邻域缩放因子，默认3.0
        top_k: 选取的负样本数量，默认3
    
    Returns:
        cgl_neg_samples: 负样本列表，每个负样本为 [x_center, y_center, width, height] (归一化坐标)
    """
    if labels is None or len(labels) == 0:
        return []
    
    # 转换为numpy数组
    img = img_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    H, W = img.shape[:2]
    
    cgl_neg_samples = []
    
    # 遍历所有真实目标
    for label in labels:
        class_id = int(label[1])
        x_center = float(label[2])
        y_center = float(label[3])
        width = float(label[4])
        height = float(label[5])
        
        # 转换为像素坐标
        cx = int(x_center * W)
        cy = int(y_center * H)
        w = int(width * W)
        h = int(height * H)
        
        # 确保尺寸有效
        if w <= 0 or h <= 0:
            continue
        
        # 定义邻域边界（3倍尺寸）
        nx1 = max(0, int(cx - neighborhood_scale * w / 2))
        ny1 = max(0, int(cy - neighborhood_scale * h / 2))
        nx2 = min(W, int(cx + neighborhood_scale * w / 2))
        ny2 = min(H, int(cy + neighborhood_scale * h / 2))
        
        # 提取邻域和模板
        neighborhood = img[ny1:ny2, nx1:nx2]
        template = img[cy - h//2:cy + h//2, cx - w//2:cx + w//2]
        
        # 确保模板有效
        if template.shape[0] == 0 or template.shape[1] == 0:
            continue
        
        # 模板匹配
        try:
            result = cv2.matchTemplate(neighborhood, template, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            # 模板匹配失败，跳过
            continue
        
        # 扣除所有真实目标区域
        for other_label in labels:
            other_cx = int(other_label[2] * W)
            other_cy = int(other_label[3] * H)
            other_w = int(other_label[4] * W)
            other_h = int(other_label[5] * H)
            
            if other_w <= 0 or other_h <= 0:
                continue
            
            # 转换到邻域坐标
            ox1_n = max(0, other_cx - other_w // 2 - nx1)
            oy1_n = max(0, other_cy - other_h // 2 - ny1)
            ox2_n = min(result.shape[1], other_cx + other_w // 2 - nx1)
            oy2_n = min(result.shape[0], other_cy + other_h // 2 - ny1)
            
            # 扣除该区域
            if oy1_n < oy2_n and ox1_n < ox2_n:
                result[oy1_n:oy2_n, ox1_n:ox2_n] = 0
        
        # 获取Top-3或随机3个位置
        if result.max() > 0:
            # 获取匹配值最高的top_k个位置
            flat_result = result.flatten()
            top_k_indices = np.argpartition(flat_result, -top_k)[-top_k:]
            top_k_indices = top_k_indices[np.argsort(-flat_result[top_k_indices])]
        else:
            # 随机选3个位置
            top_k_indices = np.random.randint(0, result.shape[0] * result.shape[1], top_k)
        
        # 转换回图像坐标
        for idx in top_k_indices:
            ny, nx = np.unravel_index(idx, result.shape)
            
            # 使用真实目标的宽高
            neg_x1 = nx1 + nx - w // 2
            neg_y1 = ny1 + ny - h // 2
            neg_x2 = neg_x1 + w
            neg_y2 = neg_y1 + h
            
            # 确保负样本在图像范围内
            neg_x1 = max(0, neg_x1)
            neg_y1 = max(0, neg_y1)
            neg_x2 = min(W, neg_x2)
            neg_y2 = min(H, neg_y2)
            
            # 转换为归一化坐标
            neg_x_center = (neg_x1 + neg_x2) / 2 / W
            neg_y_center = (neg_y1 + neg_y2) / 2 / H
            neg_width = (neg_x2 - neg_x1) / W
            neg_height = (neg_y2 - neg_y1) / H
            
            # 确保有效
            if neg_width > 0 and neg_height > 0:
                cgl_neg_samples.append([neg_x_center, neg_y_center, neg_width, neg_height])
    
    return cgl_neg_samples


def batch_compute_cgl_negative_samples(imgs, labels_list, neighborhood_scale=3.0, top_k=3):
    """
    批量计算CGL负样本
    
    Args:
        imgs: 图像张量列表，每个格式为 (3, H, W)
        labels_list: 标签列表，每个为 (N, 6) 格式
        neighborhood_scale: 邻域缩放因子
        top_k: 选取的负样本数量
    
    Returns:
        cgl_neg_samples_list: 负样本列表的列表
    """
    cgl_neg_samples_list = []
    
    for img, labels in zip(imgs, labels_list):
        neg_samples = compute_cgl_negative_samples(img, labels, neighborhood_scale, top_k)
        cgl_neg_samples_list.append(neg_samples)
    
    return cgl_neg_samples_list