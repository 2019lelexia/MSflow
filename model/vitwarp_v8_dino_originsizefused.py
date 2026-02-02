import numpy as np
import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.depthanythingv2 import DepthAnythingFeature, DinoV3Feature
from model.backbone.vit import VisionTransformer, MODEL_CONFIGS

from utils.utils import coords_grid, Padder, bilinear_sampler
from .corr import CorrBlock

import timm
from flash_attn import flash_attn_func

class resconv(nn.Module):
    def __init__(self, inp, oup, k=3, s=1):
        super(resconv, self).__init__()
        self.conv = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(inp, oup, kernel_size=k, stride=s, padding=k//2, bias=True),
            nn.GELU(),
            nn.Conv2d(oup, oup, kernel_size=3, stride=1, padding=1, bias=True),
        )
        if inp != oup or s != 1:
            self.skip_conv = nn.Conv2d(inp, oup, kernel_size=1, stride=s, padding=0, bias=True)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.skip_conv(x)

class ResNet18DeconvOriginSize(nn.Module):
    def __init__(self, inp, oup):
        super(ResNet18DeconvOriginSize, self).__init__()
        self.feature_dims = [64, 128, 256, 512]
        self.ds1 = resconv(inp, 64, k=7, s=2)
        a = timm.create_model("resnet18.a3_in1k", pretrained=False, features_only=False)
        a.load_state_dict(torch.load('resnet18.bin'), strict=False)
        self.conv1 = a.layer1
        self.conv2 = a.layer2
        self.conv3 = a.layer3
        self.conv4 = a.layer4
        self.up_4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, bias=True)
        self.proj_3 = resconv(256, 256, k=3, s=1)
        self.up_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, bias=True)
        self.proj_2 = resconv(128, 128, k=3, s=1)
        self.up_2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, bias=True)
        self.proj_1 = resconv(64, oup, k=3, s=1)

        self.conv_0 = resconv(inp, oup // 2, k=3, s=1) 
        self.up_1 = nn.ConvTranspose2d(oup, oup // 2, kernel_size=2, stride=2, padding=0, bias=True)
        self.proj_0 = resconv(oup // 2, oup // 2, k=3, s=1)

    def forward(self, x):
        out_1 = self.ds1(x)
        out_1 = self.conv1(out_1)
        out_2 = self.conv2(out_1)
        out_3 = self.conv3(out_2)
        out_4 = self.conv4(out_3)
        out_3 = self.proj_3(out_3 + self.up_4(out_4))
        out_2 = self.proj_2(out_2 + self.up_3(out_3))
        out_1 = self.proj_1(out_1 + self.up_2(out_2))

        feat_0 = self.conv_0(x)
        out_0 = self.proj_0(feat_0 + self.up_1(out_1))

        return [out_1, out_2, out_3, out_4, out_0]

class HighResRefineBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        # Input composition: 
        # Flow (2) + Image1 (3) + Warped_Image2 (3) + Error_Map (3) + Feat1_1x (in_channels)
        input_dim = 2 + 3 + 3 + 3 + in_channels
        
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # 使用空洞卷积扩大感受野，捕捉细微运动的周边上下文
        self.res_block = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2), # Dilation
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 输出 flow 的残差
        self.out_conv = nn.Conv2d(hidden_dim, 2, 1)

    def forward(self, flow, img1, img2, feat1):
        """
        flow: Upsampled flow at 1x resolution (B, 2, H, W)
        img1, img2: Original RGB images (B, 3, H, W)
        feat1: 1x resolution features from fnet (B, C, H, W)
        """
        N, _, H, W = img1.shape
        coords = coords_grid(N, H, W, device=img1.device)
        coords_warped = coords + flow
        
        u_norm = 2 * coords_warped[:, 0, :, :] / (W - 1) - 1
        v_norm = 2 * coords_warped[:, 1, :, :] / (H - 1) - 1
        grid = torch.stack([u_norm, v_norm], dim=-1)
        
        warped_img2 = F.grid_sample(img2, grid, align_corners=True, padding_mode="border")
        error_map = img1 - warped_img2
        x = torch.cat([flow, img1, warped_img2, error_map, feat1], dim=1)
        x = self.conv1(x)
        x = self.res_block(x) + x
        delta_flow = self.out_conv(x)
        
        return flow + delta_flow

class ViTWarpV8(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.da_feature = self.freeze_(DepthAnythingFeature(encoder=args.dav2_backbone))
        self.dino_feature = DinoV3Feature(freeze_backbone=True)
        self.pretrain_dim = 128
        self.network_dim = MODEL_CONFIGS[args.network_backbone]['features']
        self.num_levels = 4
        self.num_radius = 4
        self.refine_net = VisionTransformer(args.network_backbone, self.network_dim, patch_size=8)
        self.fnet = ResNet18DeconvOriginSize(self.pretrain_dim//2 + 3, 64)
        # self.fnet = ResNet18Deconv(3, 64)
        self.fmap_conv = nn.Conv2d(self.pretrain_dim//2 + 64, self.network_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fmap_conv_originsize = nn.Conv2d(self.pretrain_dim//2 + 32, self.network_dim//2, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fmap_conv = nn.Conv2d(64, self.network_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.hidden_conv = nn.Conv2d(self.network_dim*2, self.network_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.warp_linear = nn.Conv2d(3*self.network_dim+2, self.network_dim, 1, 1, 0, bias=True)
        self.refine_transform = nn.Conv2d(self.network_dim//2*3, self.network_dim, 1, 1, 0, bias=True)
        self.upsample_weight = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(self.network_dim + self.network_dim, 2*self.network_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*self.network_dim, 4*9, 1, padding=0, bias=True)
        )
        self.flow_head = nn.Sequential(
            # flow(2)
            nn.Conv2d(self.network_dim + self.network_dim//2, 2*self.network_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*self.network_dim, 2, 1, padding=0, bias=True)
        )
        self.repair = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(self.network_dim + self.network_dim * 3 + 2, 2 * self.network_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * self.network_dim, 6, 1, padding=0, bias=True)
        )

    def freeze_(self, model):
        model = model.eval()
        for p in model.parameters():
            p.requires_grad = False
        for p in model.buffers():
            p.requires_grad = False
        return model

    def upsample_data(self, flow, info, mask):
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 2, 2, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(2 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)
        
        return up_flow.reshape(N, 2, 2*H, 2*W), up_info.reshape(N, C, 2*H, 2*W)
    
    def upsample_data_only_flow(self, flow, mask):
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 2, 2, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(2 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        
        return up_flow.reshape(N, 2, 2*H, 2*W)

    def normalize_image(self, img):
        '''
        @img: (B,C,H,W) in range 0-255, RGB order
        '''
        tf = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
        return tf(img/255.0).contiguous()

    def forward(self, image1, image2, iters=None, flow_gt=None):
        """ Estimate optical flow between pair of frames """
        if iters is None:
            iters = self.args.iters
        image1 = self.normalize_image(image1)
        image2 = self.normalize_image(image2)
        padder = Padder(image1.shape, factor=16)
        image1 = padder.pad(image1)
        image2 = padder.pad(image2)
        flow_predictions = []
        info_predictions = []
        semantic_losses = []
        N, _, H, W = image1.shape
        # initial feature
        da_feature1 = self.dino_feature(image1)
        da_feature2 = self.dino_feature(image2)
        fmap1_feats = self.fnet(torch.cat([da_feature1['out'], image1], dim=1))
        fmap2_feats = self.fnet(torch.cat([da_feature2['out'], image2], dim=1))
        da_feature1_2x = F.interpolate(da_feature1['out'], scale_factor=0.5, mode='bilinear', align_corners=True)
        da_feature2_2x = F.interpolate(da_feature2['out'], scale_factor=0.5, mode='bilinear', align_corners=True)
        fmap1_2x = self.fmap_conv(torch.cat([fmap1_feats[0], da_feature1_2x], dim=1))
        fmap2_2x = self.fmap_conv(torch.cat([fmap2_feats[0], da_feature2_2x], dim=1))
        fmap1_1x = self.fmap_conv_originsize(torch.cat([fmap1_feats[-1], da_feature1['out']], dim=1))
        fmap2_1x = self.fmap_conv_originsize(torch.cat([fmap2_feats[-1], da_feature2['out']], dim=1))
        net = self.hidden_conv(torch.cat([fmap1_2x, fmap2_2x], dim=1))
        flow_2x = torch.zeros(N, 2, H//2, W//2).to(image1.device)
        
        # da_feature1_2x_norm = F.normalize(da_feature1_2x, dim=1, p=2)

        for itr in range(iters):
            flow_2x = flow_2x.detach()
            coords2 = (coords_grid(N, H//2, W//2, device=image1.device) + flow_2x).detach()
            warp_2x = bilinear_sampler(fmap2_2x, coords2.permute(0, 2, 3, 1))

            # warp_dino_2x = bilinear_sampler(da_feature2_2x, coords2.permute(0, 2, 3, 1))
            
            # if flow_gt is not None:
            #     warp_dino_2x_norm = F.normalize(warp_dino_2x, dim=1, p=2)
            #     similarity = torch.sum(da_feature1_2x_norm * warp_dino_2x_norm, dim=1, keepdim=True)
            #     with torch.no_grad():
            #         ones_mask = torch.ones(N, 1, H//2, W//2).to(image1.device)
            #         valid_mask = bilinear_sampler(ones_mask, coords2.permute(0, 2, 3, 1))
            #         valid_mask = (valid_mask > 0.99).float()
            #     if valid_mask.sum() != 0:
            #         iter_sem_loss = (1.0 - similarity) * valid_mask
            #         iter_sem_loss = iter_sem_loss.sum() / (valid_mask.sum())
            #         semantic_losses.append(iter_sem_loss)
            
            refine_inp = self.warp_linear(torch.cat([fmap1_2x, warp_2x, net, flow_2x], dim=1))
            refine_outs = self.refine_net(refine_inp)
            net = self.refine_transform(torch.cat([refine_outs['out'], net], dim=1))
            flow_update = self.flow_head(torch.cat([refine_outs['out'], net], dim=1))
            weight_update = .25 * self.upsample_weight(torch.cat([fmap1_2x, net], dim=1))
            flow_2x = flow_2x + flow_update
            # upsample predictions
            flow_up = self.upsample_data_only_flow(flow_2x, weight_update)
            coords2 = (coords_grid(N, H, W, device=image1.device) + flow_up).detach()
            warp_1x = bilinear_sampler(fmap2_1x, coords2.permute(0, 2, 3, 1))
            warp_dino_1x = bilinear_sampler(da_feature2['out'], coords2.permute(0, 2, 3, 1))
            net_1x = F.interpolate(net, scale_factor=2, mode='bilinear', align_corners=True)
            flow_refine = self.repair(torch.cat([fmap1_1x, warp_1x, da_feature1['out'], warp_dino_1x, net_1x, flow_up], dim=1))
            flow_1x = flow_up + flow_refine[:, :2]
            info_1x = flow_refine[:, 2:]

            flow_predictions.append(flow_1x)
            info_predictions.append(info_1x)

        for i in range(len(info_predictions)):
            flow_predictions[i] = padder.unpad(flow_predictions[i])
            info_predictions[i] = padder.unpad(info_predictions[i])
        
        if flow_gt is not None:
            nf_predictions = []
            for i in range(len(info_predictions)):                 
                raw_b = info_predictions[i][:, 2:]
                log_b = torch.zeros_like(raw_b)
                weight = info_predictions[i][:, :2]
                log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=self.args.var_max)
                log_b[:, 1] = torch.clamp(raw_b[:, 1], min=self.args.var_min, max=0)
                term2 = ((flow_gt - flow_predictions[i]).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
                term1 = weight - math.log(2) - log_b
                nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
                nf_predictions.append(nf_loss)
            output = {'flow': flow_predictions, 'info': info_predictions, 'nf': nf_predictions, 'sl': semantic_losses}
        else:
            output = {'flow': flow_predictions, 'info': info_predictions}    
        
        return output