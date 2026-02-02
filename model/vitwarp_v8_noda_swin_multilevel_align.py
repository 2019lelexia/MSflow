import numpy as np
import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.depthanythingv2 import DepthAnythingFeature
from model.backbone.vit import SwinRefineNet, MODEL_CONFIGS

from utils.utils import coords_grid, Padder, bilinear_sampler
from .corr import CorrBlock

import timm
from flash_attn import flash_attn_func
class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        zero_init: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, x_q, x_k, x_v):
        B, N, C = x_q.shape
        _, M, _ = x_k.shape
        original_dtype = x_q.dtype

        q = self.q(x_q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k(x_k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v(x_v).reshape(B, M, self.num_heads, C // self.num_heads)

        q = q.half()
        k = k.half()
        v = v.half()

        x = flash_attn_func(q, k, v)
        x = x.to(original_dtype)
        
        x = x.reshape(B, N, C)
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, d_model, num_heads=2, mlp_expand=4):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)
        self.attention = CrossAttention(dim=d_model, num_heads=num_heads)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_expand),
            nn.GELU(),
            nn.Linear(d_model * mlp_expand, d_model)
        )
        
    def forward(self, x_q, x_k, x_v):
        attn_out = self.attention(
            self.norm_q(x_q), 
            self.norm_k(x_k), 
            self.norm_v(x_v)
        )
        x = x_q + attn_out

        x = x + self.mlp(self.norm_mlp(x))
        
        return x

class HybridFusion(nn.Module):
    def __init__(self, dim, num_blocks=4, num_heads=2, mlp_expand=2, **kwargs):
        super(HybridFusion, self).__init__()
        
        self.blocks = nn.ModuleList([
            Block(dim, num_heads=num_heads, mlp_expand=mlp_expand) 
            for _ in range(num_blocks)
        ])
        
    def reshape_to_sequence(self, x):
        return x.flatten(2).permute(0, 2, 1)

    def reshape_to_spatial(self, x, H, W):
        B, L, C = x.shape
        return x.permute(0, 2, 1).reshape(B, C, H, W)

    def forward(self, x_q, x_k, x_v):
        B, C, H, W = x_q.shape
        _B, _C, _h, _w = x_k.shape

        q_seq = self.reshape_to_sequence(x_q)
        k_seq = self.reshape_to_sequence(x_k)
        v_seq = self.reshape_to_sequence(x_v)

        for block in self.blocks:
            q_seq = block(q_seq, k_seq, v_seq)
        x_out = self.reshape_to_spatial(q_seq, H, W)
        return x_out

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

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h

class ViTWarpV8(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.da_feature = self.freeze_(DepthAnythingFeature(encoder=args.dav2_backbone))
        # self.pretrain_dim = self.da_feature.model_configs[args.dav2_backbone]['features']
        self.network_dim = MODEL_CONFIGS[args.network_backbone]['features']
        self.num_levels = 4
        self.num_radius = 4
        self.refine_net = SwinRefineNet(input_dim=self.network_dim * 3)
        # self.fnet = ResNet18Deconv(self.pretrain_dim//2 + 3, 64)
        self.fnet = ResNet18Deconv(3, 64)
        # self.fmap_conv = nn.Conv2d(self.pretrain_dim//2 + 64, self.network_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fmap_conv = nn.Conv2d(64, self.network_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fmap_conv_4x = nn.Conv2d(128, self.network_dim * 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.hidden_conv = nn.Conv2d(self.network_dim*2, self.network_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.hidden_conv_4x = nn.Conv2d(self.network_dim*4, self.network_dim * 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.warp_linear = nn.Conv2d(3*self.network_dim+2, self.network_dim, 1, 1, 0, bias=True)
        self.warp_linear_4x = nn.Conv2d(3*2*self.network_dim+2, self.network_dim * 2, 1, 1, 0, bias=True)
        self.refine_transform = nn.Conv2d(self.network_dim*2, self.network_dim, 1, 1, 0, bias=True)
        self.refine_transform_4x = nn.Conv2d(self.network_dim*4, self.network_dim * 2, 1, 1, 0, bias=True)
        self.upsample_weight = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(self.network_dim*3, 2*self.network_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*self.network_dim, 4*9, 1, padding=0, bias=True)
        )
        self.flow_head = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(self.network_dim*3, 2*self.network_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*self.network_dim, 6, 1, padding=0, bias=True)
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
        padder = Padder(image1.shape, factor=32)
        image1 = padder.pad(image1)
        image2 = padder.pad(image2)
        flow_predictions = []
        info_predictions = [] 
        N, _, H, W = image1.shape
        # initial feature
        # da_feature1 = self.da_feature(image1)
        # da_feature2 = self.da_feature(image2)
        # fmap1_feats = self.fnet(torch.cat([da_feature1['out'], image1], dim=1))
        # fmap2_feats = self.fnet(torch.cat([da_feature2['out'], image2], dim=1))
        fmap1_feats = self.fnet(image1)
        fmap2_feats = self.fnet(image2)
        # da_feature1_2x = F.interpolate(da_feature1['out'], scale_factor=0.5, mode='bilinear', align_corners=True)
        # da_feature2_2x = F.interpolate(da_feature2['out'], scale_factor=0.5, mode='bilinear', align_corners=True)
        # fmap1_2x = self.fmap_conv(torch.cat([fmap1_feats[0], da_feature1_2x], dim=1))
        # fmap2_2x = self.fmap_conv(torch.cat([fmap2_feats[0], da_feature2_2x], dim=1))
        fmap1_2x = self.fmap_conv(fmap1_feats[0])
        fmap2_2x = self.fmap_conv(fmap2_feats[0])
        fmap1_4x = self.fmap_conv_4x(fmap1_feats[1])
        fmap2_4x = self.fmap_conv_4x(fmap2_feats[1])

        net = self.hidden_conv(torch.cat([fmap1_2x, fmap2_2x], dim=1))
        net_4x = self.hidden_conv_4x(torch.cat([fmap1_4x, fmap2_4x], dim=1))
        flow_2x = torch.zeros(N, 2, H//2, W//2).to(image1.device)
        for itr in range(iters):
            flow_2x = flow_2x.detach()
            coords2 = (coords_grid(N, H//2, W//2, device=image1.device) + flow_2x).detach()
            warp_2x = bilinear_sampler(fmap2_2x, coords2.permute(0, 2, 3, 1))

            flow_4x_guide = F.interpolate(flow_2x, scale_factor=0.5, mode='bilinear', align_corners=True) * 0.5
            coords2_4x = (coords_grid(N, H//4, W//4, device=image1.device) + flow_4x_guide).detach()
            warp_4x = bilinear_sampler(fmap2_4x, coords2_4x.permute(0, 2, 3, 1))


            # corr = correlation(fmap1_2x, warp_2x, False, radius=4, flow=None)
            # refine_inp = self.warp_linear(torch.cat([fmap1_2x, warp_2x, net, corr, flow_2x], dim=1))
            
            # fmap1_2x_norm = F.normalize(fmap1_2x, p=2, dim=1)
            # warp_2x_norm = F.normalize(warp_2x, p=2, dim=1)
            # similarity_map = torch.sum(fmap1_2x_norm * warp_2x_norm, dim=1, keepdim=True)
            # different_map = fmap1_2x - warp_2x
            # refine_inp = self.warp_linear(torch.cat([fmap1_2x, warp_2x, different_map, similarity_map, net, flow_2x], dim=1))
            
            # refine_inp = self.warp_linear(torch.cat([fmap1_2x, warp_2x, net, flow_2x], dim=1))
            refine_inp = self.warp_linear(torch.cat([fmap1_2x, warp_2x, net, flow_2x], dim=1))
            refine_inp_4x = self.warp_linear_4x(torch.cat([fmap1_4x, warp_4x, net_4x, flow_4x_guide], dim=1))
            refine_inp_4x_2x = F.interpolate(refine_inp_4x, scale_factor=2, mode='bilinear', align_corners=True)
            refine_outs = self.refine_net(torch.cat([refine_inp, refine_inp_4x_2x], dim=1))['out']
            refine_outs_2x1, refine_outs_2x2 = torch.split(refine_outs, [self.network_dim, 2 * self.network_dim], dim=1)
            net = self.refine_transform(torch.cat([refine_outs_2x1, net], dim=1))
            refine_outs_2x_4x = F.interpolate(refine_outs_2x2, scale_factor=0.5, mode='bilinear', align_corners=True)
            net_4x = self.refine_transform_4x(torch.cat([refine_outs_2x_4x, net_4x], dim=1))
            net_4x_2x = F.interpolate(net_4x, scale_factor=2, mode='bilinear', align_corners=True)
            flow_update = self.flow_head(torch.cat([net, net_4x_2x], dim=1))
            weight_update = .25 * self.upsample_weight(torch.cat([net, net_4x_2x], dim=1))
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
            align_loss_maps = []
            mono_loss_maps = []
            prev_flow = torch.zeros_like(flow_gt)
            prev_epe = torch.norm(flow_gt - prev_flow, p=2, dim=1)

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

                curr_flow = flow_predictions[i]
                pred_delta = curr_flow - prev_flow
                target_delta = flow_gt - prev_flow
                cos_sim = F.cosine_similarity(pred_delta, target_delta, dim=1, eps=1e-8)
                target_mag = torch.norm(target_delta, p=2, dim=1)
                valid_mask = (target_mag > 0.5).float()
                a_loss = (1.0 - cos_sim) * valid_mask
                align_loss_maps.append(a_loss)

                tolerance = 0.1
                curr_epe = torch.norm(flow_gt - curr_flow, p=2, dim=1)
                m_loss = F.relu(curr_epe - prev_epe - tolerance)
                mono_loss_maps.append(m_loss)

                prev_flow = curr_flow.detach()
                prev_epe = curr_epe.detach()

            output = {'flow': flow_predictions, 'info': info_predictions, 'nf': nf_predictions, 'align': align_loss_maps, 'mono': mono_loss_maps}
        else:
            output = {'flow': flow_predictions, 'info': info_predictions}    
        
        return output