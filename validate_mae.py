import sys
import math
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from config.parser import parse_args

from tqdm import tqdm
from model import fetch_model
from utils.utils import resize_data, load_ckpt

from dataloader.flow.chairs import FlyingChairs
from dataloader.flow.things import FlyingThings3D
from dataloader.flow.sintel import MpiSintel
from dataloader.flow.kitti import KITTI
from dataloader.flow.spring import Spring
from dataloader.flow.hd1k import HD1K
from dataloader.stereo.tartanair import TartanAir
from model.backbone.depthanythingv2 import ViTMAEFeature, DepthAnythingFeature
from utils.utils import coords_grid, Padder, bilinear_sampler
import matplotlib.pyplot as plt
# 设置后端为 Agg，确保在没有显示器的服务器上也能正常保存图片
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
# ================= PCA 辅助函数 (纯 PyTorch 实现) =================
def compute_pca_projection(x):
    """计算 PCA 投影矩阵和均值"""
    c, h, w = x.shape
    x_flat = x.view(c, -1).permute(1, 0) # [N, C]
    mean = x_flat.mean(dim=0)
    x_centered = x_flat - mean
    cov = torch.mm(x_centered.t(), x_centered) / (x_centered.size(0) - 1)
    _, _, V = torch.svd(cov)
    projection_matrix = V[:, :3] # 取前3个主成分
    return projection_matrix, mean

def apply_pca(x, projection_matrix, mean):
    """应用 PCA 并归一化到 0-1 用于显示"""
    c, h, w = x.shape
    x_flat = x.view(c, -1).permute(1, 0)
    x_centered = x_flat - mean
    x_pca = torch.mm(x_centered, projection_matrix)
    # Min-Max 归一化
    x_min = x_pca.min(dim=0, keepdim=True)[0]
    x_max = x_pca.max(dim=0, keepdim=True)[0]
    x_pca = (x_pca - x_min) / (x_max - x_min + 1e-8)
    return x_pca.view(h, w, 3)


# val_dataset = Spring(split='val')
# val_loader = data.DataLoader(val_dataset, batch_size=1, pin_memory=False, shuffle=True, num_workers=16, drop_last=False)
val_dataset = MpiSintel(split='training', dstype='clean')
val_loader = data.DataLoader(val_dataset, batch_size=1, pin_memory=False, shuffle=True, num_workers=16, drop_last=False)
# pbar = tqdm(total=len(val_loader))
# print(f"load data success {len(val_loader)}")
a = 0
for i_batch, data_blob in enumerate(val_loader):
    image1, image2, flow_gt, valid = [x.cuda(non_blocking=True) for x in data_blob]
    mae = ViTMAEFeature().cuda()
    f1 = mae(image1)['out']
    f2 = mae(image2)['out']
    H, W = image1.shape[-2:]
    coords2 = (coords_grid(1, H, W, device=image1.device) + flow_gt).detach()
    image2_warpped = bilinear_sampler(image2, coords2.permute(0, 2, 3, 1))
    f2_warpped = mae(image2_warpped)['out']
    f2_warpped_direct = bilinear_sampler(f2, coords2.permute(0, 2, 3, 1))

    if i_batch == 0:
        print("Processing visualization for batch 0...")
        
        # --- 1. 准备数据 ---
        samples_tensor = [f1[0], f2[0], f2_warpped[0], f2_warpped_direct[0]]
        base_titles = ['F1 (Ref)', 'F2 (Src)', 'F2 (Img Warped)', 'F2 (Feat Warped)']
        
        # --- 2. 统一计算 PCA 并转为 Numpy (只做一次) ---
        # 在 F1 上计算基准
        proj_mat, mean = compute_pca_projection(samples_tensor[0].detach())
        
        samples_np = []
        for tensor in samples_tensor:
            # 应用基准 -> detach -> cpu -> numpy
            pca_tensor = apply_pca(tensor, proj_mat, mean)
            samples_np.append(pca_tensor.detach().cpu().numpy())
            
        # 获取特征图尺寸
        H_feat, W_feat, _ = samples_np[0].shape

        # ==========================================
        # 任务一：保存标准全景图 (Standard View)
        # ==========================================
        fig1 = plt.figure(figsize=(16, 4))
        for i, img_np in enumerate(samples_np):
            plt.subplot(1, 4, i + 1)
            plt.imshow(img_np)
            plt.title(base_titles[i], fontsize=14)
            plt.axis('off')
            
        save_path_std = f"feature_vis_standard_batch_{i_batch}.jpg"
        plt.tight_layout()
        # 标准图 DPI 不需要太高，150 足够
        fig1.savefig(save_path_std, dpi=150, bbox_inches='tight')
        plt.close(fig1) # 重要：关闭特定的 figure
        print(f"Saved standard visualization to {save_path_std}")

        # ==========================================
        # 任务二：保存局部放大图 (Zoomed Crop View)
        # ==========================================
        # 【配置】定义裁剪区域 (ROI - Region of Interest)
        # 这里默认取图像中心区域，大约占长宽的 40%
        # 你可以手动修改这些坐标值来对准你想看的物体边缘
        y_start = int(H_feat * 0.3)
        y_end   = int(H_feat * 0.7)
        x_start = int(W_feat * 0.3)
        x_end   = int(W_feat * 0.7)
        
        print(f"Creating zoomed view for crop region: y[{y_start}:{y_end}], x[{x_start}:{x_end}]")

        # 创建新的画布，尺寸可以稍大一点以展示细节
        fig2 = plt.figure(figsize=(20, 5)) 
        
        for i, img_np in enumerate(samples_np):
            # 【关键步骤】对 numpy 数组进行切片裁剪
            # img_np shape is [H, W, 3]
            img_crop = img_np[y_start:y_end, x_start:x_end, :]
            
            plt.subplot(1, 4, i + 1)
            plt.imshow(img_crop)
            # 标题加上 Zoomed 标识
            plt.title(f"{base_titles[i]} (Zoomed)", fontsize=14, fontweight='bold')
            plt.axis('off')

        save_path_zoom = f"feature_vis_zoomed_batch_{i_batch}.jpg"
        plt.tight_layout()
        # 【关键】放大图使用更高的 DPI (例如 300) 来保留清晰度
        fig2.savefig(save_path_zoom, dpi=300, bbox_inches='tight')
        plt.close(fig2) # 关闭 figure
        print(f"Saved zoomed visualization to {save_path_zoom}")
        # ==========================================
    break

