import numpy as np
import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.depthanythingv2 import DepthAnythingDepth
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

class ResNet18Deconv(nn.Module):
    def __init__(self, inp, oup):
        super(ResNet18Deconv, self).__init__()
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

    def forward(self, x):
        out_1 = self.ds1(x)
        out_1 = self.conv1(out_1)
        out_2 = self.conv2(out_1)
        out_3 = self.conv3(out_2)
        out_4 = self.conv4(out_3)
        out_3 = self.proj_3(out_3 + self.up_4(out_4))
        out_2 = self.proj_2(out_2 + self.up_3(out_3))
        out_1 = self.proj_1(out_1 + self.up_2(out_2))
        return [out_1, out_2, out_3, out_4]

import torch

def forward_warp_with_reduce(fmap1, flow, depth=None):
    B, C, H, W = fmap1.shape
    device = fmap1.device
    coords_y, coords_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    grid = torch.stack([coords_x, coords_y], dim=0).float().expand(B, -1, -1, -1)
    target_grid = grid + flow
    tx = target_grid[:, 0, ...].flatten(1)
    ty = target_grid[:, 1, ...].flatten(1)
    valid = (tx >= 0) & (tx < W-1) & (ty >= 0) & (ty < H-1) # [B, H*W]
    ix = tx.long()
    iy = ty.long()
    index = (iy * W + ix)
    if depth is not None:
        weight = torch.exp(-depth).flatten(1) # [B, H*W]
    else:
        weight = torch.ones((B, H * W), device=device) 
    src_features = fmap1.flatten(2) * weight.unsqueeze(1) # [B, C, H*W]
    src_weight = weight.unsqueeze(1) # [B, 1, H*W]
    target_features_list = []
    target_weights_list = []
    for b in range(B):
        b_valid = valid[b]
        b_index = index[b][b_valid]
        b_src_feat = src_features[b][:, b_valid]
        b_src_weight = src_weight[b][:, b_valid]
        b_target_features = fmap1.new_zeros(C, H * W)
        b_target_weights = fmap1.new_zeros(1, H * W)
        b_target_features.scatter_reduce_(1, b_index.unsqueeze(0).expand(C, -1), b_src_feat, reduce='sum', include_self=False)
        b_target_weights.scatter_reduce_(1, b_index.unsqueeze(0), b_src_weight, reduce='sum', include_self=False)
        target_features_list.append(b_target_features)
        target_weights_list.append(b_target_weights)  
    target_features = torch.stack(target_features_list, dim=0)
    target_weights = torch.stack(target_weights_list, dim=0)
    mask = (target_weights > 0)
    final_fmap = torch.where(mask, target_features / target_weights, target_features)
    return final_fmap.view(B, C, H, W), mask.view(B, 1, H, W).float()

class ViTWarpV8(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.da_feature = self.freeze_(DepthAnythingDepth(encoder=args.dav2_backbone))
        # self.pretrain_dim = self.da_feature.model_configs[args.dav2_backbone]['features']
        self.network_dim = MODEL_CONFIGS[args.network_backbone]['features']
        self.num_levels = 4
        self.num_radius = 4
        self.refine_net_pull = VisionTransformer(args.network_backbone, self.network_dim, patch_size=8)
        self.refine_net_push = VisionTransformer(args.network_backbone, self.network_dim, patch_size=8)
        # self.fnet = ResNet18Deconv(self.pretrain_dim//2 + 3, 64)
        self.fnet = ResNet18Deconv(4, 64)
        # self.fmap_conv = nn.Conv2d(self.pretrain_dim//2 + 64, self.network_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fmap_conv = nn.Conv2d(64, self.network_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.hidden_conv = nn.Conv2d(self.network_dim*2, self.network_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.hidden_update_conv = nn.Conv2d(self.network_dim//2*3, self.network_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.warp_linear_pull = nn.Conv2d(3*self.network_dim+2+1, self.network_dim, 1, 1, 0, bias=True)
        self.warp_linear_push = nn.Conv2d(3*self.network_dim+2+1, self.network_dim, 1, 1, 0, bias=True)

        self.refine_transform = nn.Conv2d(self.network_dim*2, self.network_dim, 1, 1, 0, bias=True)
        self.upsample_weight = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(self.network_dim, self.network_dim*2, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.network_dim*2, 4*9, 1, padding=0, bias=True)
        )
        self.flow_head = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(self.network_dim, self.network_dim*2, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.network_dim*2, 6, 1, padding=0, bias=True)
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
        padder = Padder(image1.shape, factor=112)
        image1 = padder.pad(image1)
        image2 = padder.pad(image2)
        flow_predictions = []
        info_predictions = [] 
        N, _, H, W = image1.shape
        # initial feature
        d1 = self.da_feature(image1)
        d2 = self.da_feature(image2)
        fmap1_feats = self.fnet(torch.cat([image1, d1['out']], dim=1))
        fmap2_feats = self.fnet(torch.cat([image2, d2['out']], dim=1))
        d1_2x = F.interpolate(d1['out'], scale_factor=0.5, mode='bilinear', align_corners=True)
        # da_feature1_2x = F.interpolate(da_feature1['out'], scale_factor=0.5, mode='bilinear', align_corners=True)
        # da_feature2_2x = F.interpolate(da_feature2['out'], scale_factor=0.5, mode='bilinear', align_corners=True)
        fmap1_2x = self.fmap_conv(fmap1_feats[0])
        fmap2_2x = self.fmap_conv(fmap2_feats[0])
        net = self.hidden_conv(torch.cat([fmap1_2x, fmap2_2x], dim=1))
        flow_2x = torch.zeros(N, 2, H//2, W//2).to(image1.device)
        
        for itr in range(iters):
            flow_2x = flow_2x.detach()
            coords2 = (coords_grid(N, H//2, W//2, device=image1.device) + flow_2x).detach()
            warp_2x = bilinear_sampler(fmap2_2x, coords2.permute(0, 2, 3, 1))
            mask_pull = (coords2[:, 0:1, :, :] >= 0) & \
                        (coords2[:, 0:1, :, :] <= (W//2 - 1)) & \
                        (coords2[:, 1:2, :, :] >= 0) & \
                        (coords2[:, 1:2, :, :] <= (H//2 - 1))
            mask_pull = mask_pull.float()
            warp_2x_push, mask_push = forward_warp_with_reduce(fmap1_2x, flow_2x, d1_2x)
            refine_inp_pull = self.warp_linear_pull(torch.cat([fmap1_2x, warp_2x, net, flow_2x, mask_pull], dim=1))
            refine_inp_push = self.warp_linear_push(torch.cat([fmap2_2x, warp_2x_push, net, flow_2x, mask_push], dim=1))
            refine_outs_pull = self.refine_net_pull(refine_inp_pull)
            refine_outs_push = self.refine_net_push(refine_inp_push)

            # refine_inp = self.warp_linear(torch.cat([fmap1_2x, fmap2_2x, warp_2x, warp_2x_push, net, flow_2x, mask_pull, mask_push], dim=1))
            # refine_outs = self.refine_net(refine_inp)
            net = self.refine_transform(torch.cat([refine_outs_pull['out'], refine_outs_push['out'], net], dim=1))
            flow_update = self.flow_head(net)
            weight_update = .25 * self.upsample_weight(net)

            flow_2x = flow_2x + flow_update[:, :2]
            info_2x = flow_update[:, 2:]
            # upsample predictions
            flow_up, info_up = self.upsample_data(flow_2x, info_2x, weight_update)
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)

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
            output = {'flow': flow_predictions, 'info': info_predictions, 'nf': nf_predictions}
        else:
            output = {'flow': flow_predictions, 'info': info_predictions}    
        
        return output
