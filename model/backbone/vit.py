import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math

import sys
from model.backbone.patch_embed import PatchEmbed
from thirdparty.DepthAnythingV2.depth_anything_v2.dpt import DPTHead
from timm.layers import PatchEmbed as timmPatchEmbed
from timm.layers import resample_abs_pos_embed
from timm.models.twins import PatchEmbed as twinsPatchEmbed

MODEL_CONFIGS = {
    'vitl': {'encoder': 'vit_large_patch16_224', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitb': {'encoder': 'vit_base_patch16_224', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vits': {'encoder': 'vit_small_patch16_224', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitt': {'encoder': 'vit_tiny_patch16_224', 'features': 32, 'out_channels': [24, 48, 96, 192]}
}
class VisionTransformer(nn.Module):
    def __init__(self, model_name, input_dim, patch_size=16):
        super(VisionTransformer, self).__init__()
        model = timm.create_model(
            MODEL_CONFIGS[model_name]['encoder'],
            pretrained=False,
            num_classes=0,  # remove classifier nn.Linear
        )
        if model_name == 'vits':
            model.load_state_dict(torch.load('vits.bin'), strict=False)
        elif model_name == 'vitl':
            model.load_state_dict(torch.load('vitl.bin'), strict=False)
        elif model_name == 'vitb':
            model.load_state_dict(torch.load('vitb.bin'), strict=False)
        elif model_name == 'vitt':
            model.load_state_dict(torch.load('vitt.bin'), strict=False)
        self.intermediate_layer_idx = {
            'vitt': [2, 5, 8, 11],
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        self.idx = self.intermediate_layer_idx[model_name]
        self.blks = model.blocks
        self.embed_dim = model.embed_dim
        self.input_dim = input_dim
        self.img_size = (224, 224)
        self.patch_size = patch_size
        self.output_dim = MODEL_CONFIGS[model_name]['features']
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, self.embed_dim))
        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, in_chans=input_dim, embed_dim=self.embed_dim)
        self.dpt_head = DPTHead(self.embed_dim, MODEL_CONFIGS[model_name]['features'], out_channels=MODEL_CONFIGS[model_name]['out_channels'])

    def interpolate_pos_encoding(self, x, h, w):
        previous_dtype = x.dtype
        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sy, sx),
            mode="bicubic",
            antialias=False
        )
        assert int(w0) == pos_embed.shape[-1]
        assert int(h0) == pos_embed.shape[-2]
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return pos_embed.to(previous_dtype)

    def forward(self, x):
        B, nc, h, w = x.shape
        x = self.patch_embed(x)
        x = x + self.interpolate_pos_encoding(x, h, w)
        outputs = []
        for i in range(len(self.blks)):
            x = self.blks[i](x)
            if i in self.idx:
                outputs.append([x])

        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        out, path_1, path_2, path_3, path_4 = self.dpt_head.forward(outputs, patch_h, patch_w, return_intermediate=True)
        out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
        return {'out': out, 'path_1':path_1, 'path_2':path_2, 'path_3':path_3, 'path_4':path_4}  # path_1 is 1/2; path_2 is 1/4

        

class SwinRefineNet(nn.Module):
    def __init__(self, input_dim=64):
        super().__init__()
        self.backbone = timm.create_model('swinv2_tiny_window8_256.ms_in1k', pretrained=False, strict_img_size=False)
        self.backbone.load_state_dict(torch.load('swint.bin'), strict=False)
        orig_embed = self.backbone.patch_embed
        embed_dim = self.backbone.embed_dim
        self.backbone.patch_embed = timmPatchEmbed(
            img_size=None,
            patch_size=orig_embed.patch_size,
            in_chans=input_dim,
            embed_dim=embed_dim,
            norm_layer=type(orig_embed.norm),  
            flatten=True,
            output_fmt=orig_embed.output_fmt
        )
        del self.backbone.head
        del self.backbone.norm
        
        self.feat_dims = [stage.blocks[0].dim for stage in self.backbone.layers]
        total_dim = sum(self.feat_dims)

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(total_dim, input_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim * 2, input_dim, kernel_size=3, padding=1)
        )
        
        self.window_size = (8, 8)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        self.backbone.set_input_size(
            img_size=(H, W), 
            window_size=self.window_size
        )

        features = self.backbone.forward_intermediates(
            x,
            indices=[0, 1, 2, 3],
            output_fmt='NCHW',
            intermediates_only=True
        )
        
        c2, c3, c4, c5 = features

        size = c2.shape[-2:]
        c3 = F.interpolate(c3, size=size, mode='bilinear', align_corners=False)
        c4 = F.interpolate(c4, size=size, mode='bilinear', align_corners=False)
        c5 = F.interpolate(c5, size=size, mode='bilinear', align_corners=False)

        fused = torch.cat([c2, c3, c4, c5], dim=1)
        out = self.fuse_conv(fused)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        return {'out': out, 'path_1': c2, 'path_2': c3, 'path_3': c4, 'path_4': c5}

class TwinsRefineNet(nn.Module):
    def __init__(self, input_dim=64):
        super().__init__()
        self.backbone = timm.create_model('twins_svt_small.in1k', pretrained=False, in_chans=3)
        self.backbone.load_state_dict(torch.load('twinss.bin'), strict=False)
        self.backbone.patch_embeds[0] = twinsPatchEmbed(img_size=224, patch_size=4, in_chans=64, embed_dim=64)
        del self.backbone.head
        del self.backbone.head_drop
        del self.backbone.norm

        self.feat_dims = self.backbone.embed_dims
        total_dim = sum(self.feat_dims)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(total_dim, input_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim * 2, input_dim, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape

        features = self.backbone.forward_intermediates(
            x,
            indices=[0, 1, 2, 3],
            output_fmt='NCHW',
            intermediates_only=True
        )
        c2, c3, c4, c5 = features

        size = c2.shape[-2:]
        c3 = F.interpolate(c3, size=size, mode='bilinear', align_corners=False)
        c4 = F.interpolate(c4, size=size, mode='bilinear', align_corners=False)
        c5 = F.interpolate(c5, size=size, mode='bilinear', align_corners=False)

        fused = torch.cat([c2, c3, c4, c5], dim=1)
        out = self.fuse_conv(fused)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        return {'out': out, 'path_1': c2, 'path_2': c3, 'path_3': c4, 'path_4': c5}


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

class DPTHead1(nn.Module):
    def __init__(
        self, 
        in_channels=768,
        features=128,
        use_bn=False,
        out_channels=[96, 192, 384, 768],
        num_classes=1,
        patch_size=16
    ):
        super(DPTHead1, self).__init__()
        
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
    
class DinoV3ViTSmall(nn.Module):
    def __init__(self, freeze_backbone=False, pretrained=True, in_channel=64, patch_size=16):
        super().__init__()
        self.backbone = timm.create_model('vit_small_patch16_dinov3.lvd1689m', pretrained=False)
        if pretrained:
            self.backbone.load_state_dict(torch.load('dinov3_small.bin'), strict=False)
        self.patch_size = patch_size
        self.backbone.patch_embed = timmPatchEmbed(img_size=(256, 256), patch_size=self.patch_size, in_chans=in_channel, embed_dim=384, dynamic_img_pad=False, bias=True, strict_img_size=False, output_fmt='NHWC')
        self.dpthead = DPTHead1(in_channels=384, features=64, out_channels=[48, 96, 192, 384], patch_size=self.patch_size)
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
        return {'out': out}

class EfficientViTL1(nn.Module):
    def __init__(self, pretrained=True, input_dim=64):
        super().__init__()
        self.backbone = timm.create_model('efficientvit_l1.r224_in1k', pretrained=False)
        if pretrained:
            self.backbone.load_state_dict(torch.load('efficientvitl1.bin'), strict=False)
        self.backbone.stages[0].blocks[0].main.spatial_conv.conv = nn.Conv2d(input_dim, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.backbone.stem = nn.Identity()
        del self.backbone.head

        total_dim = 64 + 128 + 256 + 512
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(total_dim, input_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim * 2, input_dim // 2, kernel_size=3, padding=1)
        )
    def forward(self, x):
        B, C, H, W = x.shape

        features = self.backbone.forward_intermediates(
            x,
            indices=[0, 1, 2, 3],
            output_fmt='NCHW',
            intermediates_only=True
        )
        
        c2, c3, c4, c5 = features

        size = c2.shape[-2:]
        c3 = F.interpolate(c3, size=size, mode='bilinear', align_corners=False)
        c4 = F.interpolate(c4, size=size, mode='bilinear', align_corners=False)
        c5 = F.interpolate(c5, size=size, mode='bilinear', align_corners=False)

        fused = torch.cat([c2, c3, c4, c5], dim=1)
        out = self.fuse_conv(fused)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        return {'out': out, 'path_1': c2, 'path_2': c3, 'path_3': c4, 'path_4': c5}


class XCiT(nn.Module):
    """
    size should be totally divided by 8
    """
    def __init__(self, pretrained=True, input_dim=64):
        super().__init__()
        self.backbone = timm.create_model('xcit_small_12_p8_224.fb_in1k', pretrained=False)
        if pretrained:
            self.backbone.load_state_dict(torch.load('xcits.bin'), strict=True)
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = 8
        if input_dim != 3:
            self.backbone.patch_embed.proj[0][0] = nn.Conv2d(input_dim, out_channels=self.embed_dim//4, kernel_size=3, stride=2, padding=1, bias=False)
        self.dpthead = DPTHead1(in_channels=384, features=64, out_channels=[48, 96, 192, 384], patch_size=self.patch_size)

    def forward(self, x):
        h, w = x.shape[-2:]
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        features = self.backbone.forward_intermediates(
            x,
            indices=[2, 5, 8, 11],
            output_fmt='NCHW',
            intermediates_only=True
        )
        out, path_1, path_2, path_3, path_4 = self.dpthead(features, patch_h, patch_w, return_intermediate=True)
        return {'out': out}

from transformers import AutoModelForDepthEstimation
class SwinV2DPT(nn.Module):
    """
    size should be totally divided by 32
    """
    def __init__(self, input_dim=64):
        super().__init__()
        self.backbone = AutoModelForDepthEstimation.from_pretrained('dpt-swinv2-tiny-256')
        if input_dim != 3:
            self.backbone.backbone.embeddings.patch_embeddings.projection = nn.Conv2d(input_dim, 96, kernel_size=4, stride=4)
        self.patch_size = 8
        del self.backbone.head.head[-1]
        del self.backbone.head.head[-1]
        del self.backbone.head.head[-1]
        
    def forward(self, x):
        h ,w = x.shape[-2:]
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        features = self.backbone.backbone(x).feature_maps
        features = self.backbone.neck(features, patch_h, patch_w)
        out = self.backbone.head(features)
        return {'out': out}


class vits_dino(nn.Module):
    def __init__(self, input_dim=64, patch_size=8):
        super().__init__()
        self.backbone = timm.create_model('vit_small_patch8_224.dino', pretrained=False, in_chans=3)
        self.backbone.load_state_dict(torch.load('vits_dino.bin'), strict=True)
        if input_dim != 3:
            self.backbone.patch_embed.proj = nn.Conv2d(input_dim, self.backbone.embed_dim, kernel_size=patch_size, stride=patch_size)
        self.backbone.patch_embed.strict_img_size = False
        self.patch_size = patch_size
        self.embed_dim = self.backbone.embed_dim
        self.dpt_head = DPTHead1(in_channels=384, features=64, out_channels=[48, 96, 192, 384], patch_size=self.patch_size)
        self.intermediate_layer_idx = {
            'vitt': [2, 5, 8, 11],
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        self.idx = self.intermediate_layer_idx['vits']
        self.pos_embed = self.backbone.pos_embed
    
    def forward(self, x):
        h, w = x.shape[-2:]
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        x = self.backbone.patch_embed(x)
        dynamic_pos_embed = resample_abs_pos_embed(
            self.pos_embed, 
            new_size=(patch_h, patch_w), 
            num_prefix_tokens=1
        )
        dynamic_pos_embed = dynamic_pos_embed[:, 1:, :]
        x = x + dynamic_pos_embed
        outputs = []
        for i in range(len(self.backbone.blocks)):
            x = self.backbone.blocks[i](x)
            if i in self.idx:
                B, N, C = x.shape
                feat = x.transpose(1, 2).reshape(B, C, patch_h, patch_w)
                outputs.append(feat)
        # features = self.backbone.forward_intermediates(x, indices=[2, 5, 8, 11], output_fmt='NCHW', intermediates_only=True)
        # out, path_1, path_2, path_3, path_4 = self.dpt_head(outputs, patch_h, patch_w, return_intermediate=True)
        out, path_1, path_2, path_3, path_4 = self.dpt_head(outputs, patch_h, patch_w, return_intermediate=True)
        return {'out': out, 'path_1':path_1, 'path_2':path_2, 'path_3':path_3, 'path_4':path_4}

class vits_dino_16(nn.Module):
    def __init__(self, input_dim=64, patch_size=16):
        super().__init__()
        self.backbone = timm.create_model('vit_small_patch16_224.dino', pretrained=False, in_chans=3)
        self.backbone.load_state_dict(torch.load('vits_dino_16.bin'), strict=True)
        if input_dim != 3:
            self.backbone.patch_embed.proj = nn.Conv2d(input_dim, self.backbone.embed_dim, kernel_size=patch_size, stride=patch_size)
        self.backbone.patch_embed.strict_img_size = False
        self.patch_size = patch_size
        self.embed_dim = self.backbone.embed_dim
        self.dpt_head = DPTHead1(in_channels=384, features=64, out_channels=[48, 96, 192, 384], patch_size=self.patch_size)
        self.intermediate_layer_idx = {
            'vitt': [2, 5, 8, 11],
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        self.idx = self.intermediate_layer_idx['vits']
        self.pos_embed = self.backbone.pos_embed
    
    def forward(self, x):
        h, w = x.shape[-2:]
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        x = self.backbone.patch_embed(x)
        dynamic_pos_embed = resample_abs_pos_embed(
            self.pos_embed, 
            new_size=(patch_h, patch_w), 
            num_prefix_tokens=1
        )
        dynamic_pos_embed = dynamic_pos_embed[:, 1:, :]
        x = x + dynamic_pos_embed
        outputs = []
        for i in range(len(self.backbone.blocks)):
            x = self.backbone.blocks[i](x)
            if i in self.idx:
                B, N, C = x.shape
                feat = x.transpose(1, 2).reshape(B, C, patch_h, patch_w)
                outputs.append(feat)
        # features = self.backbone.forward_intermediates(x, indices=[2, 5, 8, 11], output_fmt='NCHW', intermediates_only=True)
        # out, path_1, path_2, path_3, path_4 = self.dpt_head(outputs, patch_h, patch_w, return_intermediate=True)
        out, path_1, path_2, path_3, path_4 = self.dpt_head(outputs, patch_h, patch_w, return_intermediate=True)
        return {'out': out, 'path_1':path_1, 'path_2':path_2, 'path_3':path_3, 'path_4':path_4}

class vitb_dino(nn.Module):
    def __init__(self, input_dim=64, patch_size=8):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch8_224.dino', pretrained=False, in_chans=3)
        self.backbone.load_state_dict(torch.load('vitb_dino.bin'), strict=True)
        if input_dim != 3:
            self.backbone.patch_embed.proj = nn.Conv2d(input_dim, self.backbone.embed_dim, kernel_size=patch_size, stride=patch_size)
        self.backbone.patch_embed.strict_img_size = False
        self.patch_size = patch_size
        self.embed_dim = self.backbone.embed_dim
        self.dpt_head = DPTHead1(in_channels=768, features=128, out_channels=[96, 192, 384, 768], patch_size=self.patch_size)
        self.intermediate_layer_idx = {
            'vitt': [2, 5, 8, 11],
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        self.idx = self.intermediate_layer_idx['vitb']
        self.pos_embed = self.backbone.pos_embed
    
    def forward(self, x):
        h, w = x.shape[-2:]
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        x = self.backbone.patch_embed(x)
        dynamic_pos_embed = resample_abs_pos_embed(
            self.pos_embed, 
            new_size=(patch_h, patch_w), 
            num_prefix_tokens=1
        )
        dynamic_pos_embed = dynamic_pos_embed[:, 1:, :]
        x = x + dynamic_pos_embed
        outputs = []
        for i in range(len(self.backbone.blocks)):
            x = self.backbone.blocks[i](x)
            if i in self.idx:
                B, N, C = x.shape
                feat = x.transpose(1, 2).reshape(B, C, patch_h, patch_w)
                outputs.append(feat)
        # features = self.backbone.forward_intermediates(x, indices=[2, 5, 8, 11], output_fmt='NCHW', intermediates_only=True)
        # out, path_1, path_2, path_3, path_4 = self.dpt_head(outputs, patch_h, patch_w, return_intermediate=True)
        out, path_1, path_2, path_3, path_4 = self.dpt_head(outputs, patch_h, patch_w, return_intermediate=True)
        return {'out': out, 'path_1':path_1, 'path_2':path_2, 'path_3':path_3, 'path_4':path_4}

class vitb_dino_16(nn.Module):
    def __init__(self, input_dim=64, patch_size=16):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224.dino', pretrained=False, in_chans=3)
        self.backbone.load_state_dict(torch.load('vitb_dino_16.bin'), strict=True)
        if input_dim != 3:
            self.backbone.patch_embed.proj = nn.Conv2d(input_dim, self.backbone.embed_dim, kernel_size=patch_size, stride=patch_size)
        self.backbone.patch_embed.strict_img_size = False
        self.patch_size = patch_size
        self.embed_dim = self.backbone.embed_dim
        self.dpt_head = DPTHead1(in_channels=768, features=128, out_channels=[96, 192, 384, 768], patch_size=self.patch_size)
        self.intermediate_layer_idx = {
            'vitt': [2, 5, 8, 11],
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        self.idx = self.intermediate_layer_idx['vitb']
        self.pos_embed = self.backbone.pos_embed
    
    def forward(self, x):
        h, w = x.shape[-2:]
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        x = self.backbone.patch_embed(x)
        dynamic_pos_embed = resample_abs_pos_embed(
            self.pos_embed, 
            new_size=(patch_h, patch_w), 
            num_prefix_tokens=1
        )
        dynamic_pos_embed = dynamic_pos_embed[:, 1:, :]
        x = x + dynamic_pos_embed
        outputs = []
        for i in range(len(self.backbone.blocks)):
            x = self.backbone.blocks[i](x)
            if i in self.idx:
                B, N, C = x.shape
                feat = x.transpose(1, 2).reshape(B, C, patch_h, patch_w)
                outputs.append(feat)
        # features = self.backbone.forward_intermediates(x, indices=[2, 5, 8, 11], output_fmt='NCHW', intermediates_only=True)
        # out, path_1, path_2, path_3, path_4 = self.dpt_head(outputs, patch_h, patch_w, return_intermediate=True)
        out, path_1, path_2, path_3, path_4 = self.dpt_head(outputs, patch_h, patch_w, return_intermediate=True)
        return {'out': out, 'path_1':path_1, 'path_2':path_2, 'path_3':path_3, 'path_4':path_4}

class SwinDPTHead(nn.Module):
    def __init__(
        self, 
        in_channels=[96, 192, 384, 768],
        features=256,
        use_bn=False,
        num_classes=1,
        patch_size=4
    ):
        super(SwinDPTHead, self).__init__()
        
        self.patch_size = patch_size
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_c,
                out_channels=features,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for in_c in in_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Identity()
        ])
        
        self.scratch = _make_scratch([features] * 4, features, groups=1, expand=False)
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32

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
            out_features: 来自 Swin 的 list，包含 4 个 Tensor
                          [B, 96, H/4, W/4], [B, 192, H/8, W/8], ...
        """
        out = []
        for i, x in enumerate(out_features):
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        

        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        
        if patch_h is not None and patch_w is not None:
             target_size = (int(patch_h * self.patch_size), int(patch_w * self.patch_size))
             out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=True)
        else:
             out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)
        
        if return_intermediate:
            return out, path_1, path_2, path_3, path_4
        else:
            out = self.scratch.output_conv2(out)
            return out

class SwinRefineNetDPT(nn.Module):
    def __init__(self, input_dim=64):
        super().__init__()
        self.backbone = timm.create_model('swinv2_tiny_window8_256.ms_in1k', pretrained=False, strict_img_size=False)
        self.backbone.load_state_dict(torch.load('swint.bin'), strict=False)
        orig_embed = self.backbone.patch_embed
        embed_dim = self.backbone.embed_dim
        self.backbone.patch_embed = timmPatchEmbed(
            img_size=None,
            patch_size=orig_embed.patch_size,
            in_chans=input_dim,
            embed_dim=embed_dim,
            norm_layer=type(orig_embed.norm),  
            flatten=True,
            output_fmt=orig_embed.output_fmt
        )
        del self.backbone.head
        del self.backbone.norm
        
        self.feat_dims = [stage.blocks[0].dim for stage in self.backbone.layers]
        # total_dim = sum(self.feat_dims)

        # self.fuse_conv = nn.Sequential(
        #     nn.Conv2d(total_dim, input_dim * 2, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(input_dim * 2, input_dim, kernel_size=3, padding=1)
        # )
        
        self.window_size = (8, 8)
        self.dpt_head = SwinDPTHead(features=128, patch_size=4)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        self.backbone.set_input_size(
            img_size=(H, W), 
            window_size=self.window_size
        )

        features = self.backbone.forward_intermediates(
            x,
            indices=[0, 1, 2, 3],
            output_fmt='NCHW',
            intermediates_only=True
        )
        
        c2, c3, c4, c5 = features
        out, path_1, path_2, path_3, path_4 = self.dpt_head(features)

        # size = c2.shape[-2:]
        # c3 = F.interpolate(c3, size=size, mode='bilinear', align_corners=False)
        # c4 = F.interpolate(c4, size=size, mode='bilinear', align_corners=False)
        # c5 = F.interpolate(c5, size=size, mode='bilinear', align_corners=False)

        # fused = torch.cat([c2, c3, c4, c5], dim=1)
        # out = self.fuse_conv(fused)
        # out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        return {'out': out, 'path_1': path_1, 'path_2': path_2, 'path_3': path_3, 'path_4': path_4}

class SwinRefineNetDPT_abspos(nn.Module):
    def __init__(self, input_dim=64):
        super().__init__()
        self.backbone = timm.create_model('swinv2_tiny_window8_256.ms_in1k', pretrained=False, strict_img_size=False)
        self.backbone.load_state_dict(torch.load('swint.bin'), strict=False)
        
        orig_embed = self.backbone.patch_embed
        
        self.patch_size = 4
        self.embed_dim = self.backbone.embed_dim
        self.num_patches = (256 // self.patch_size) * (256 // self.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.backbone.patch_embed = timmPatchEmbed(
            img_size=None,
            patch_size=orig_embed.patch_size,
            in_chans=input_dim,
            embed_dim=self.embed_dim,
            norm_layer=type(orig_embed.norm),  
            flatten=True,
            output_fmt=orig_embed.output_fmt
        )
        del self.backbone.head
        del self.backbone.norm
        
        self.feat_dims = [stage.blocks[0].dim for stage in self.backbone.layers]
        # total_dim = sum(self.feat_dims)

        # self.fuse_conv = nn.Sequential(
        #     nn.Conv2d(total_dim, input_dim * 2, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(input_dim * 2, input_dim, kernel_size=3, padding=1)
        # )
        
        self.window_size = (8, 8)
        self.dpt_head = SwinDPTHead(features=128, patch_size=4)
    
    def interpolate_pos_encoding(self, x, h, w):
        previous_dtype = x.dtype
        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sy, sx),
            mode="bicubic",
            antialias=False
        )
        assert int(w0) == pos_embed.shape[-1]
        assert int(h0) == pos_embed.shape[-2]
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return pos_embed.to(previous_dtype)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # self.backbone.set_input_size(
        #     img_size=(H, W), 
        #     window_size=self.window_size
        # )
        x = self.backbone.patch_embed(x)
        B, H_grid, W_grid, C_embed = x.shape
        x = x.view(B, -1, C_embed)
        pos_embed = self.interpolate_pos_encoding(x, H, W)
        x = x + pos_embed
        x = x.view(B, H_grid, W_grid, C_embed)
        
        outs = []
        for i, layer in enumerate(self.backbone.layers):
            x = layer(x)
            out_feat = x.permute(0, 3, 1, 2).contiguous()
            outs.append(out_feat)
        
        out, path_1, path_2, path_3, path_4 = self.dpt_head(outs)
        return {'out': out, 'path_1': path_1, 'path_2': path_2, 'path_3': path_3, 'path_4': path_4}

class SwinRefineNetDPT_nopos(nn.Module):
    def __init__(self, input_dim=64):
        super().__init__()
        self.backbone = timm.create_model('swinv2_tiny_window8_256.ms_in1k', pretrained=False, strict_img_size=False)
        self.backbone.load_state_dict(torch.load('swint.bin'), strict=False)
        
        orig_embed = self.backbone.patch_embed
        
        self.patch_size = 4
        self.embed_dim = self.backbone.embed_dim
        # self.num_patches = (256 // self.patch_size) * (256 // self.patch_size)
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        # nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.backbone.patch_embed = timmPatchEmbed(
            img_size=None,
            patch_size=orig_embed.patch_size,
            in_chans=input_dim,
            embed_dim=self.embed_dim,
            norm_layer=type(orig_embed.norm),  
            flatten=True,
            output_fmt=orig_embed.output_fmt
        )
        del self.backbone.head
        del self.backbone.norm
        
        self.feat_dims = [stage.blocks[0].dim for stage in self.backbone.layers]
        # total_dim = sum(self.feat_dims)

        # self.fuse_conv = nn.Sequential(
        #     nn.Conv2d(total_dim, input_dim * 2, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(input_dim * 2, input_dim, kernel_size=3, padding=1)
        # )
        
        self.window_size = (8, 8)
        self.dpt_head = SwinDPTHead(features=128, patch_size=4)
    
    def interpolate_pos_encoding(self, x, h, w):
        previous_dtype = x.dtype
        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sy, sx),
            mode="bicubic",
            antialias=False
        )
        assert int(w0) == pos_embed.shape[-1]
        assert int(h0) == pos_embed.shape[-2]
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return pos_embed.to(previous_dtype)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # self.backbone.set_input_size(
        #     img_size=(H, W), 
        #     window_size=self.window_size
        # )
        x = self.backbone.patch_embed(x)
        # B, H_grid, W_grid, C_embed = x.shape
        # x = x.view(B, -1, C_embed)
        # pos_embed = self.interpolate_pos_encoding(x, H, W)
        # x = x + pos_embed
        # x = x.view(B, H_grid, W_grid, C_embed)
        
        outs = []
        for i, layer in enumerate(self.backbone.layers):
            x = layer(x)
            out_feat = x.permute(0, 3, 1, 2).contiguous()
            outs.append(out_feat)
        
        out, path_1, path_2, path_3, path_4 = self.dpt_head(outs)
        return {'out': out, 'path_1': path_1, 'path_2': path_2, 'path_3': path_3, 'path_4': path_4}

if __name__ == '__main__':
    model = SwinRefineNetDPT_abspos().cuda()
    input1 = torch.randn(1, 64, 320, 640).cuda()
    output1 = model(input1)
    print(output1['out'].shape)

    # model = SwinRefineNetDPT().cuda()
    # input1 = torch.randn(1, 64, 320, 640).cuda()
    # output1 = model(input1)
    # print(output1['out'].shape)

    # model = vits_dino().cuda()
    # input1 = torch.randn(1, 64, 480, 640).cuda()
    # output1 = model(input1)
    # print(output1['out'].shape)

    # model = SwinV2DPT().cuda()
    # input1 = torch.randn(1, 64, 256, 256).cuda()
    # output1 = model(input1)
    # print(output1['out'].shape)

    # model = XCiT().cuda()
    # input1 = torch.randn(1, 64, 240, 320).cuda()
    # output1 = model(input1)
    # print(output1['out'].shape)

    # model = EfficientViTL1().cuda()
    # input1 = torch.randn(1, 64, 240, 320).cuda()
    # output1 = model(input1)
    # print(output1['out'].shape)
    
    # model = DinoV3ViTSmall(patch_size=8).cuda()
    # input1 = torch.randn(1, 64, 240, 368).cuda()
    # output1 = model(input1)
    # print(output1['out'].shape)


    # model = TwinsRefineNet().cuda()
    # input1 = torch.randn(1, 64, 256, 480).cuda()
    # output1 = model(input1)

    # model = SwinRefineNet().cuda()
    # input1 = torch.randn(1, 64, 256, 240).cuda()
    # output1 = model(input1)

    # model = VisionTransformer('vitt', 95)
    # input = torch.randn(1, 95, 512, 768)
    # output = model(input)
    # print(output['out'].shape)
    # print(output['path_1'].shape)
    # print(output['path_2'].shape)
    # print(output['path_3'].shape)
    # print(output['path_4'].shape)
    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA
    #     ],
    #     with_flops=True) as prof:
    #         output = model(input)
    
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cuda_time_total', row_limit=5))
    # events = prof.events()
    # forward_MACs = sum([int(evt.flops) for evt in events])
    # print("forward MACs: ", forward_MACs / 2 / 1e9, "G")