import torch
import torch.nn as nn
import timm
import numpy as np
import torchvision
import torch.nn.functional as F
import math
import sys
import os
from thirdparty.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2

class DepthAnythingFeature(nn.Module):
    def __init__(self, encoder='vits', pretrained=True):
        super().__init__()
        self.model_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
        }
        self.encoder = encoder
        depth_anything = DepthAnythingV2(**self.model_configs[encoder])
        if pretrained:
            depth_anything.load_state_dict(torch.load(f'depth-anything-ckpts/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        self.depth_anything = depth_anything


    def forward(self, x):
        """
        @x: (B,C,H,W)
        """
        h, w = x.shape[-2:]
        features = self.depth_anything.pretrained.get_intermediate_layers(x, self.depth_anything.intermediate_layer_idx[self.encoder], return_class_token=True)
        patch_size = self.depth_anything.pretrained.patch_size
        patch_h, patch_w = h // patch_size, w // patch_size

        out, path_1, path_2, path_3, path_4 = self.depth_anything.depth_head.forward(features, patch_h, patch_w, return_intermediate=True)

        return {'out': out, 'path_1':path_1, 'path_2':path_2, 'path_3':path_3, 'path_4':path_4, 'features':features}  # path_1 is 1/2; path_2 is 1/4

def normalize_image(img):
    '''
    @img: (B,C,H,W) in range 0-255, RGB order
    '''
    tf = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
    return tf(img/255.0).contiguous()

def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)

    return scratch

class ResidualConvUnit(nn.Module):
    def __init__(self, features, activation, bn):
        super().__init__()
        self.bn = bn
        self.groups = 1
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        return self.skip_add.add(out, x)

class FeatureFusionBlock(nn.Module):
    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        super(FeatureFusionBlock, self).__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = 1
        self.expand = expand
        out_features = features
        if self.expand:
            out_features = features // 2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        output = xs[0]
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        output = self.resConfUnit2(output)
        
        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}
            
        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)
        return output

def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )

class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels=768,
        features=128,
        use_bn=False,
        out_channels=[96, 192, 384, 768],
        num_classes=1,
        patch_size=16
    ):
        super(DPTHead, self).__init__()
        
        self.patch_size = patch_size
        
        # 1. 自动处理 out_channels
        # 如果用户没传，且输入是 ViT (单一同维度)，则自动填充 list
        if out_channels is None:
            if isinstance(in_channels, int):
                out_channels = [in_channels] * 4
            else:
                out_channels = in_channels
        
        # 2. Projects: 调整通道数
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels if isinstance(in_channels, int) else in_channels[i],
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for i, out_channel in enumerate(out_channels)
        ])
        
        # 3. Resize Layers (Reassemble)
        # 核心逻辑：输入全是 1/16 尺度，需要映射到 [1/4, 1/8, 1/16, 1/32]
        self.resize_layers = nn.ModuleList([
            # Layer 0: 1/16 -> 1/4 (放大 4 倍)
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            
            # Layer 1: 1/16 -> 1/8 (放大 2 倍)
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            
            # Layer 2: 1/16 -> 1/16 (不变)
            nn.Identity(),
            
            # Layer 3: 1/16 -> 1/32 (缩小 2 倍)
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1)
        ])
        
        # 4. RefineNet 融合模块
        self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32
        
        # 5. Output Heads
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, num_classes, kernel_size=1, stride=1, padding=0), 
            nn.Identity(),
        )
    
    def forward(self, out_features, patch_h=None, patch_w=None, return_intermediate=True):
        """
        Args:
            out_features: list of 4 tensors [(B, C, H/16, W/16)...]
            patch_h, patch_w: (可选) 原图以 patch_size 切分后的网格数量 (H_img // 16, W_img // 16)
        """
        # Step 1: Reassemble
        out = []
        for i, x in enumerate(out_features):
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        # Step 2: Feature Fusion (Deep to Shallow)
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        
        # Step 3: Upsample to Original Resolution
        if patch_h is not None and patch_w is not None:
             target_size = (int(patch_h * self.patch_size), int(patch_w * self.patch_size))
             out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=True)
        else:
             out = F.interpolate(out, scale_factor=4, mode="bilinear", align_corners=True)

        if return_intermediate:
            return out, path_1, path_2, path_3, path_4
        else:
            out = self.scratch.output_conv2(out)
            return out
    
class DinoV3Feature(nn.Module):
    def __init__(self, freeze_backbone=True, pretrained=True, patch_size=16):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_dinov3.lvd1689m', pretrained=False)
        self.backbone.load_state_dict(torch.load('dinov3.bin'), strict=False)
        self.patch_size = patch_size
        self.dpthead = DPTHead(in_channels=768, features=128, out_channels=[96, 192, 384, 768], patch_size=self.patch_size)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.buffers():
                param.requires_grad = False
            print("Info: Backbone parameters frozen.")

    def forward(self, x):
        h, w = x.shape[-2:]
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        features = self.backbone.forward_intermediates(x, indices=[2, 5, 8, 11], output_fmt='NCHW', intermediates_only=True)
        out, path_1, path_2, path_3, path_4 = self.dpthead(features, patch_h, patch_w, return_intermediate=True)
        return {'out': out, 'path_1':path_1, 'path_2':path_2, 'path_3':path_3, 'path_4':path_4}

class ViTMAEFeature(nn.Module):
    def __init__(self, freeze_backbone=True, pretrained=True, patch_size=16):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224.mae', pretrained=False)
        self.backbone.load_state_dict(torch.load('vitb_mae.bin'), strict=False)
        self.patch_size = patch_size
        self.dpthead = DPTHead(in_channels=768, features=128, out_channels=[96, 192, 384, 768], patch_size=self.patch_size)
        self.pre_size = (224, 224)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.buffers():
                param.requires_grad = False
            print("Info: Backbone parameters frozen.")

    def forward(self, x):
        h, w = x.shape[-2:]
        if (h, w) != self.pre_size:
            self.backbone.set_input_size(img_size=(h, w))
            self.pre_size = (h, w)
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        features = self.backbone.forward_intermediates(x, indices=[2, 5, 8, 11], output_fmt='NCHW', intermediates_only=True)
        out, path_1, path_2, path_3, path_4 = self.dpthead(features, patch_h, patch_w, return_intermediate=True)
        return {'out': out, 'path_1':path_1, 'path_2':path_2, 'path_3':path_3, 'path_4':path_4}

class ViTMAEFeature_abspos(nn.Module):
    def __init__(self, freeze_backbone=True, pretrained=True, patch_size=16):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224.mae', pretrained=False)
        self.backbone.load_state_dict(torch.load('vitb_mae.bin'), strict=False)
        self.backbone.patch_embed.strict_img_size = False
        self.patch_size = patch_size
        raw_pos_embed = self.backbone.pos_embed
        self.cls_pos_embed = nn.Parameter(raw_pos_embed[:, 0:1, :])
        self.orig_pos_embed = nn.Parameter(raw_pos_embed[:, 1:, :])
        self.orig_grid_size = int(math.sqrt(self.orig_pos_embed.shape[1]))

        self.dpthead = DPTHead(in_channels=768, features=128, out_channels=[96, 192, 384, 768], patch_size=self.patch_size)
        self.pre_size = (224, 224)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.buffers():
                param.requires_grad = False
            print("Info: Backbone parameters frozen.")
    
    def interpolate_pos_encoding(self, x, h, w):
        previous_dtype = x.dtype
        npatch = x.shape[1]
        N = self.orig_pos_embed.shape[1]
        if npatch == N and w == h:
            return self.orig_pos_embed.to(previous_dtype)
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        sqrt_N = self.orig_grid_size
        data = self.orig_pos_embed.float().transpose(1, 2).view(1, dim, sqrt_N, sqrt_N)
        out = nn.functional.interpolate(
            data,
            size=(h0, w0),
            mode="bicubic",
            antialias=False
        )
        assert int(w0) == out.shape[-1]
        assert int(h0) == out.shape[-2]
        out = out.permute(0, 2, 3, 1).view(1, -1, dim)
        return out.to(previous_dtype)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_patch = self.backbone.patch_embed(x)
        pos_embed_patch = self.interpolate_pos_encoding(x_patch, H, W)
        x_patch = x_patch + pos_embed_patch
        if self.backbone.cls_token is not None:
            cls_token = self.backbone.cls_token.expand(B, -1, -1)
            cls_token = cls_token + self.cls_pos_embed
            x = torch.cat((cls_token, x_patch), dim=1)
        else:
            x = x_patch
        x = self.backbone.pos_drop(x)
        features = []
        target_indices = {2, 5, 8, 11}
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if i in target_indices:
                if self.backbone.cls_token is not None:
                    out = x[:, 1:, :] 
                else:
                    out = x
                out = out.transpose(1, 2).reshape(B, -1, patch_h, patch_w)
                features.append(out)
        out, path_1, path_2, path_3, path_4 = self.dpthead(features, patch_h, patch_w, return_intermediate=True)
        return {'out': out, 'path_1':path_1, 'path_2':path_2, 'path_3':path_3, 'path_4':path_4}

class ViTMAEFeature_nopos(nn.Module):
    def __init__(self, freeze_backbone=True, pretrained=True, patch_size=16):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224.mae', pretrained=False)
        self.backbone.load_state_dict(torch.load('vitb_mae.bin'), strict=False)
        self.backbone.patch_embed.strict_img_size = False
        self.patch_size = patch_size
        # raw_pos_embed = self.backbone.pos_embed
        # self.cls_pos_embed = nn.Parameter(raw_pos_embed[:, 0:1, :])
        # self.orig_pos_embed = nn.Parameter(raw_pos_embed[:, 1:, :])
        # self.orig_grid_size = int(math.sqrt(self.orig_pos_embed.shape[1]))

        self.dpthead = DPTHead(in_channels=768, features=128, out_channels=[96, 192, 384, 768], patch_size=self.patch_size)
        self.pre_size = (224, 224)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.buffers():
                param.requires_grad = False
            print("Info: Backbone parameters frozen.")
    
    def interpolate_pos_encoding(self, x, h, w):
        previous_dtype = x.dtype
        npatch = x.shape[1]
        N = self.orig_pos_embed.shape[1]
        if npatch == N and w == h:
            return self.orig_pos_embed.to(previous_dtype)
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        sqrt_N = self.orig_grid_size
        data = self.orig_pos_embed.float().transpose(1, 2).view(1, dim, sqrt_N, sqrt_N)
        out = nn.functional.interpolate(
            data,
            size=(h0, w0),
            mode="bicubic",
            antialias=False
        )
        assert int(w0) == out.shape[-1]
        assert int(h0) == out.shape[-2]
        out = out.permute(0, 2, 3, 1).view(1, -1, dim)
        return out.to(previous_dtype)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_patch = self.backbone.patch_embed(x)
        # pos_embed_patch = self.interpolate_pos_encoding(x_patch, H, W)
        # x_patch = x_patch + pos_embed_patch
        # if self.backbone.cls_token is not None:
        #     cls_token = self.backbone.cls_token.expand(B, -1, -1)
        #     cls_token = cls_token + self.cls_pos_embed
        #     x = torch.cat((cls_token, x_patch), dim=1)
        # else:
        #     x = x_patch
        # x = self.backbone.pos_drop(x)
        x = x_patch
        features = []
        target_indices = {2, 5, 8, 11}
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if i in target_indices:
                out = x
                out = out.transpose(1, 2).reshape(B, -1, patch_h, patch_w)
                features.append(out)
        out, path_1, path_2, path_3, path_4 = self.dpthead(features, patch_h, patch_w, return_intermediate=True)
        return {'out': out, 'path_1':path_1, 'path_2':path_2, 'path_3':path_3, 'path_4':path_4}
    
if __name__ == '__main__':
    m = ViTMAEFeature_abspos().cuda()
    a = torch.randn(2, 3, 480, 640).cuda()
    b = m(a)
    print(b['out'].shape)
    print(b['path_1'].shape)
    print(b['path_2'].shape)