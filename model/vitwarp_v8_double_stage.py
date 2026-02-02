import numpy as np
import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.depthanythingv2 import DepthAnythingFeature
from model.backbone.vit import VisionTransformer, MODEL_CONFIGS, vits_dino, vitb_dino, vits_dino_16, vitb_dino_16

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

class ResNet18(nn.Module):
    def __init__(self, inp, oup):
        super(ResNet18, self).__init__()
        self.feature_dims = [64, 128, 256, 512]
        self.ds1 = resconv(inp, 64, k=7, s=2)
        a = timm.create_model("resnet18.a3_in1k", pretrained=False, features_only=False)
        a.load_state_dict(torch.load('resnet18.bin'), strict=False)
        self.conv1 = a.layer1
        self.conv2 = a.layer2
        self.conv3 = a.layer3

    def forward(self, x):
        out_1 = self.ds1(x)
        out_1 = self.conv1(out_1)
        out_2 = self.conv2(out_1)
        out_3 = self.conv3(out_2)
        return [out_1, out_2, out_3]

class ResizingDepthAnythingWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        
        self.core_model = self.freeze_(encoder)
    
    def freeze_(self, model):
        model = model.eval()
        for p in model.parameters():
            p.requires_grad = False
        for p in model.buffers():
            p.requires_grad = False
        return model

    def forward(self, x):
        original_h, original_w = x.shape[-2:]
        target_h = int(np.ceil(original_h / 14) * 14)
        target_w = int(np.ceil(original_w / 14) * 14)
        if target_h != original_h or target_w != original_w:
            x_input = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=True)
        else:
            x_input = x
        feats = self.core_model(x_input) 
        if isinstance(feats, dict):
            out_feat = feats['out']
        else:
            out_feat = feats
        if out_feat.shape[-2:] != (original_h, original_w):
            out_feat = F.interpolate(out_feat, size=(original_h, original_w), mode='bilinear', align_corners=True)
        return {'out': out_feat}

class ViTWarpV8(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        ## step 1
        # self.da_feature = ResizingDepthAnythingWrapper(DepthAnythingFeature(encoder=args.dav2_backbone))
        # self.pretrain_dim = self.da_feature.core_model.model_configs[args.dav2_backbone]['features']
        # self.pretrain_dim = self.da_feature.model_configs[args.dav2_backbone]['features']
        self.network_dim = MODEL_CONFIGS[args.network_backbone]['features']
        self.refine_net_step1 = vitb_dino_16(self.network_dim * 2, patch_size=16)
        # self.refine_net = VisionTransformer(args.network_backbone, self.network_dim, patch_size=8)
        # self.refine_net_4x = VisionTransformer('vitb', self.network_dim * 2, patch_size=8)
        # self.refine_net_1x = VisionTransformer('vitt', self.network_dim // 2, patch_size=8)
        # self.fnet = ResNet18Deconv(self.pretrain_dim//2 + 3, 64)
        self.fnet = ResNet18Deconv(3, 64)
        # self.fmap_conv = nn.Conv2d(self.pretrain_dim//2 + 64, self.network_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fmap_conv = nn.Conv2d(64, self.network_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fmap_conv_4x = nn.Conv2d(128, self.network_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fmap_conv_8x = nn.Conv2d(256, self.network_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.compress = nn.Conv2d(3 * self.network_dim, self.network_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.flow_head_step1 = nn.Sequential(
            nn.Conv2d(self.network_dim, 2*self.network_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*self.network_dim, 6, 1, padding=0, bias=True)
        )
        self.upsample_weight_step1 = nn.Sequential(
            nn.Conv2d(self.network_dim, 2*self.network_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*self.network_dim, 4*9, 1, padding=0, bias=True)
        )

        ## step 2
        self.fnet_warpped = ResNet18(3, 64)
        self.fmap_conv_warpped = nn.Conv2d(64, self.network_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fmap_conv_4x_warpped = nn.Conv2d(128, self.network_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fmap_conv_8x_warpped = nn.Conv2d(256, self.network_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.compress_warpped = nn.Conv2d(3 * self.network_dim, self.network_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.warp_linear = nn.Conv2d(4 * self.network_dim + 2, 2 * self.network_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.refine_net_step2 = vits_dino(2 * self.network_dim, 8)

        self.flow_head_step2 = nn.Sequential(
            nn.Conv2d(self.network_dim//2, self.network_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.network_dim, 6, 1, padding=0, bias=True)
        )
        self.upsample_weight_step2 = nn.Sequential(
            nn.Conv2d(self.network_dim//2, self.network_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.network_dim, 4*9, 1, padding=0, bias=True)
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
        fmap1_feats = self.fnet(image1)
        fmap2_feats = self.fnet(image2)
        # da_feature1_2x = F.interpolate(da_feature1['out'], scale_factor=0.5, mode='bilinear', align_corners=True)
        # da_feature2_2x = F.interpolate(da_feature2['out'], scale_factor=0.5, mode='bilinear', align_corners=True)
        fmap1_2x = self.fmap_conv(fmap1_feats[0])
        fmap2_2x = self.fmap_conv(fmap2_feats[0])
        fmap1_4x = self.fmap_conv_4x(fmap1_feats[1])
        fmap2_4x = self.fmap_conv_4x(fmap2_feats[1])
        fmap1_8x = self.fmap_conv_8x(fmap1_feats[2])
        fmap2_8x = self.fmap_conv_8x(fmap2_feats[2])
        fmap1_8x_2x = F.interpolate(fmap1_8x, scale_factor=4, mode='bilinear', align_corners=True)
        fmap2_8x_2x = F.interpolate(fmap2_8x, scale_factor=4, mode='bilinear', align_corners=True)
        fmap1_4x_2x = F.interpolate(fmap1_4x, scale_factor=2, mode='bilinear', align_corners=True)
        fmap2_4x_2x = F.interpolate(fmap2_4x, scale_factor=2, mode='bilinear', align_corners=True)
        fmap1_2x_fused = self.compress(torch.cat([fmap1_2x, fmap1_4x_2x, fmap1_8x_2x], dim=1))
        fmap2_2x_fused = self.compress(torch.cat([fmap2_2x, fmap2_4x_2x, fmap2_8x_2x], dim=1))
        refine_out_step1_2x = self.refine_net_step1(torch.cat([fmap1_2x_fused, fmap2_2x_fused], dim=1))
        flow_update_step1_2x = self.flow_head_step1(refine_out_step1_2x['out'])
        weight_update_step1_2x = .25 * self.upsample_weight_step1(refine_out_step1_2x['out'])
        flow_step1_2x = flow_update_step1_2x[:, :2]
        info_step1_2x = flow_update_step1_2x[:, 2:]
        flow_step1_1x, info_step1_1x = self.upsample_data(flow_step1_2x, info_step1_2x, weight_update_step1_2x)
        flow_predictions.append(flow_step1_1x)
        info_predictions.append(info_step1_1x)
        
        flow_2x = torch.zeros(N, 2, H//2, W//2).to(image1.device)
        flow_1x = flow_step1_1x.detach()
        flow_2x = flow_2x + flow_step1_2x.detach()
        for itr in range(iters):
            flow_1x = flow_1x.detach()
            flow_2x = flow_2x.detach()
            coords2_1x = (coords_grid(N, H, W, device=image1.device) + flow_1x).detach()
            image2_warp_1x = bilinear_sampler(image2, coords2_1x.permute(0, 2, 3, 1))
            fmap2_warpped = self.fnet_warpped(image2_warp_1x)
            fmap2_warpped_2x = self.fmap_conv_warpped(fmap2_warpped[0])
            fmap2_warpped_4x = self.fmap_conv_4x_warpped(fmap2_warpped[1])
            fmap2_warpped_8x = self.fmap_conv_8x_warpped(fmap2_warpped[2])
            fmap2_warpped_4x_2x = F.interpolate(fmap2_warpped_4x, scale_factor=2, mode='bilinear', align_corners=True)
            fmap2_warpped_8x_2x = F.interpolate(fmap2_warpped_8x, scale_factor=4, mode='bilinear', align_corners=True)
            fmap2_warpped_2x_fused = self.compress_warpped(torch.cat([fmap2_warpped_2x, fmap2_warpped_4x_2x, fmap2_warpped_8x_2x], dim=1))
            coords2_2x = (coords_grid(N, H // 2, W // 2, device=image1.device) + flow_2x).detach()
            warp_2x = bilinear_sampler(fmap2_2x_fused, coords2_2x.permute(0, 2, 3, 1))
            refine_inp_step2_2x = self.warp_linear(torch.cat([fmap1_2x_fused, fmap2_2x_fused, fmap2_warpped_2x_fused, warp_2x, flow_2x], dim=1))
            refine_out_step2_2x = self.refine_net_step2(refine_inp_step2_2x)
            flow_update_step2_2x = self.flow_head_step2(refine_out_step2_2x['out'])
            weight_update_step2_2x = .25 * self.upsample_weight_step2(refine_out_step2_2x['out'])
            flow_2x = flow_2x + flow_update_step2_2x[:, :2]
            info_2x = flow_update_step2_2x[:, 2:]
            flow_step2_1x, info_step2_1x = self.upsample_data(flow_2x, info_2x, weight_update_step2_2x)
            flow_1x = flow_step2_1x.detach()
        
            flow_predictions.append(flow_step2_1x)
            info_predictions.append(info_step2_1x)

        for i in range(len(info_predictions)):
            flow_predictions[i] = padder.unpad(flow_predictions[i])
            info_predictions[i] = padder.unpad(info_predictions[i])
        
        if flow_gt is not None:
            nf_predictions = []
            # mono_loss_maps = []
            # prev_flow = flow_predictions[0].detach()
            # prev_epe = torch.norm(flow_gt - prev_flow, p=2, dim=1)
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

                # if i > 0:
                #     curr_flow = flow_predictions[i]
                #     curr_epe = torch.norm(flow_gt - curr_flow, p=2, dim=1)
                #     m_loss = F.relu(curr_epe - prev_epe)
                #     mono_loss_maps.append(m_loss)
                #     prev_epe = curr_epe.detach()
                # else:
                #     mono_loss_maps.append(0.0 * nf_loss)
            
            output = {'flow': flow_predictions, 'info': info_predictions, 'nf': nf_predictions}
        else:
            output = {'flow': flow_predictions, 'info': info_predictions}    
        
        return output