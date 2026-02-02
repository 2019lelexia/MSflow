import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.utils import bilinear_sampler, coords_grid
# from compute_sparse_correlation import compute_sparse_corr, compute_sparse_corr_torch, compute_sparse_corr_mink

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass

class CorrBlockMine:
    def __init__(self, fmap1_list, fmap2_list, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # 多尺度相关性：每层都用对应分辨率的 fmap1 和 fmap2
        assert len(fmap1_list) > num_levels, 'len(fmap_list) < num_levels'
        for i in range(num_levels):
            fmap1 = fmap1_list[-i-2]  # 第 i 层的 f1
            fmap2 = fmap2_list[-i-2]  # 第 i 层的 f2
            corr = CorrBlockMine.corr(fmap1, fmap2)  # [B, H1, W1, 1, H2, W2]
            b, h1, w1, dim, h2, w2 = corr.shape
            corr = corr.reshape(b * h1 * w1, dim, h2, w2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        """
        coords: [B, 2, H, W], 表示最高分辨率下的坐标
        """
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)  # [B, H, W, 2]
        b, H, W, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]

            coords_lvl = coords / (2**i)
            coords_lvl = F.interpolate(
                coords_lvl.permute(0, 3, 1, 2),  # [B, 2, H, W]
                scale_factor=1/(2**i), 
                mode='bilinear',
                align_corners=True
            ).permute(0, 2, 3, 1)  # [B, H/2^i, W/2^i, 2]

            bh, h1, w1, _ = coords_lvl.shape
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1)  # [2r+1, 2r+1, 2]

            centroid_lvl = coords_lvl.reshape(bh * h1 * w1, 1, 1, 2)
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2).to(coords.device)
            coords_lvl = centroid_lvl + delta_lvl

            # bilinear sample
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(b, h1, w1, -1)
            if (h1, w1) != (H, W):
                corr = F.interpolate(corr.permute(0,3,1,2), size=(H, W), mode='bilinear', align_corners=True)
                corr = corr.permute(0,2,3,1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        b, dim, h1, w1 = fmap1.shape
        _, _, h2, w2 = fmap2.shape

        fmap1 = fmap1.view(b, dim, h1 * w1)
        fmap2 = fmap2.view(b, dim, h2 * w2)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(b, h1, w1, 1, h2, w2)
        return corr / torch.sqrt(torch.tensor(dim, dtype=torch.float32, device=fmap1.device))


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


class CorrBlockSingleScale(nn.Module):
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        super().__init__()
        self.radius = radius

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)
        batch, h1, w1, dim, h2, w2 = corr.shape
        self.corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        corr = self.corr
        dx = torch.linspace(-r, r, 2 * r + 1)
        dy = torch.linspace(-r, r, 2 * r + 1)
        delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1).to(coords.device)

        centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2)
        delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
        coords_lvl = centroid_lvl + delta_lvl

        corr = bilinear_sampler(corr, coords_lvl)
        out = corr.view(batch, h1, w1, -1)
        out = out.permute(0, 3, 1, 2).contiguous().float()
        return out

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


class LocalCorrBlock(nn.Module):
    def __init__(self, num_levels=4, radius=4):
        super().__init__()
        self.num_levels = num_levels
        self.radius = radius
        
        r = self.radius
        dx = torch.linspace(-r, r, 2 * r + 1)
        dy = torch.linspace(-r, r, 2 * r + 1)
        delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1)
        self.K = (2 * r + 1)**2
        self.register_buffer('delta', delta.view(1, self.K, 1, 1, 2))

    def build_feature_pyramid(self, fmap):
        pyramid = [fmap]
        for i in range(self.num_levels - 1):
            fmap = F.avg_pool2d(fmap, 2, stride=2)
            pyramid.append(fmap)
        return pyramid

    def forward(self, fmap0, fmap1_warped):
        fmap1_warped_pyramid = self.build_feature_pyramid(fmap1_warped)
        B, C, H, W = fmap0.shape
        fmap0_g = fmap0.unsqueeze(2)
        coords0 = coords_grid(B, H, W, fmap0.device)
        out_pyramid = []
        for i in range(self.num_levels):
            fmap1_lvl = fmap1_warped_pyramid[i]
            coords0_lvl = (coords0 / (2**i)).permute(0, 2, 3, 1).unsqueeze(1)
            sample_coords = coords0_lvl + self.delta.to(coords0.device)
            coords_for_sampler = sample_coords.view(B, self.K * H, W, 2)
            fmap1_sampled = bilinear_sampler(fmap1_lvl, coords_for_sampler)
            fmap1_sampled = fmap1_sampled.view(B, C, self.K, H, W)
            corr_lvl = torch.sum(fmap0_g * fmap1_sampled, dim=1)            
            out_pyramid.append(corr_lvl)
        out = torch.cat(out_pyramid, dim=1)
        out = out / torch.sqrt(torch.tensor(C).float())
        return out.contiguous().float()