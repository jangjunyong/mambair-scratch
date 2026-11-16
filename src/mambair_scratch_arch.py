"""
MambaIR-Scratch: MambaIR + Scratch-specific modules
  - DirectionalSSM  : 1x7 / 7x1 방향 커널로 선형 패턴 명시적 학습
  - ScratchMaskAttn : 손상 영역 집중 처리 (Partial Conv 아이디어)
  - ScratchMambaBlock : RSSB 교체용 블록

배치 위치: basicsr/archs/mambair_scratch_arch.py
등록 방법: basicsr/archs/__init__.py 에 import 한 줄 추가
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba
from basicsr.utils.registry import ARCH_REGISTRY


# ─────────────────────────────────────────────
# 0. 공통 유틸
# ─────────────────────────────────────────────
class LayerNorm(nn.Module):
    """채널-우선(C,H,W) 입력용 LayerNorm"""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, C, H, W)
        x = x.permute(0, 2, 3, 1)   # → (B,H,W,C)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2) # → (B,C,H,W)


# ─────────────────────────────────────────────
# 1. DirectionalConv
#    스크래치의 선형 방향성 패턴을 명시적으로 캡처
#    1x7, 7x1 두 방향 DW conv 를 병렬로 처리 후 합산
# ─────────────────────────────────────────────
class DirectionalConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 수평 방향 (1x7)
        self.conv_h = nn.Conv2d(
            dim, dim, kernel_size=(1, 7),
            padding=(0, 3), groups=dim, bias=False
        )
        # 수직 방향 (7x1)
        self.conv_v = nn.Conv2d(
            dim, dim, kernel_size=(7, 1),
            padding=(3, 0), groups=dim, bias=False
        )
        # 방향 가중치 (학습 가능)
        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.5)
        self.beta  = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.5)

    def forward(self, x):
        return self.alpha * self.conv_h(x) + self.beta * self.conv_v(x)


# ─────────────────────────────────────────────
# 2. ScratchMaskAttn
#    Partial Conv 아이디어 기반:
#    손상 강도 맵(scratch_map)을 추정해 유효 픽셀에 가중치를 높임
#    scratch_map 은 입력에서 자동 추정 (별도 GT 불필요)
# ─────────────────────────────────────────────
class ScratchMaskAttn(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        # 손상 맵 추정기: 가벼운 CNN
        self.mask_estimator = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, 1, 1),
            nn.Sigmoid()   # 0=손상, 1=정상
        )
        # 손상 영역과 정상 영역을 각각 다른 가중치로 처리
        self.clean_proj   = nn.Conv2d(dim, dim, 1)
        self.scratch_proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        mask = self.mask_estimator(x)          # (B,1,H,W) 손상 확률
        clean_feat   = self.clean_proj(x)  * mask
        scratch_feat = self.scratch_proj(x) * (1 - mask)
        return clean_feat + scratch_feat


# ─────────────────────────────────────────────
# 3. ChannelAttention (MambaIR 원본과 동일)
# ─────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


# ─────────────────────────────────────────────
# 4. SSM 모듈 (Mamba 래퍼)
# ─────────────────────────────────────────────
class VSSModule(nn.Module):
    """2D feature → flatten → Mamba SSM → reshape"""
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # (B, C, H, W) → (B, H*W, C)
        x_flat = x.flatten(2).transpose(1, 2)
        x_out  = self.mamba(x_flat)
        # (B, H*W, C) → (B, C, H, W)
        return x_out.transpose(1, 2).reshape(B, C, H, W)


# ─────────────────────────────────────────────
# 5. ScratchMambaBlock (RSSB 교체용)
#    RSSB(원본) + DirectionalConv + ScratchMaskAttn
# ─────────────────────────────────────────────
class ScratchMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        # SSM branch
        self.ssm   = VSSModule(dim, d_state, d_conv, expand)

        # Directional Conv branch (스크래치 특화)
        self.dir_conv = DirectionalConv(dim)

        # Scratch Mask Attention
        self.sma = ScratchMaskAttn(dim)

        # Channel Attention (원본 RSSB 와 동일)
        self.ca  = ChannelAttention(dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1),
        )

        # 학습 가능한 스킵 가중치 (MambaIR 원본)
        self.gamma1 = nn.Parameter(torch.ones(dim) * 1e-4)
        self.gamma2 = nn.Parameter(torch.ones(dim) * 1e-4)

    def forward(self, x):
        B, C, H, W = x.shape

        # ── Branch 1: SSM + DirectionalConv ──
        residual = x
        x_norm = self.norm1(x)
        ssm_out = self.ssm(x_norm)
        dir_out = self.dir_conv(x_norm)
        # SSM과 방향 conv 결합
        branch1 = ssm_out + dir_out
        # Channel Attention
        branch1 = self.ca(branch1)
        # Scratch Mask Attention
        branch1 = self.sma(branch1)
        g1 = self.gamma1.view(1, C, 1, 1)
        x = residual + g1 * branch1

        # ── Branch 2: FFN ──
        residual = x
        g2 = self.gamma2.view(1, C, 1, 1)
        x = residual + g2 * self.ffn(self.norm2(x))

        return x


# ─────────────────────────────────────────────
# 6. MambaIRScratch (최종 네트워크)
#    MambaIR 전체 구조 + ScratchMambaBlock 적용
# ─────────────────────────────────────────────
import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class MambaIRScratch(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=48, num_blocks=8, d_state=16, d_conv=4, expand=2, img_range=1.):
        super(MambaIRScratch, self).__init__()
        self.img_range = img_range
        self.num_blocks = num_blocks
        
        self.conv_in = nn.Conv2d(inp_channels, dim, 3, 1, 1)
        
        # 블록들을 리스트로 관리하여 각각의 출력을 빼낼 수 있게 분리 (Dense Skip 핵심)
        self.blocks = nn.ModuleList([
            ScratchMambaBlock(dim, d_state, d_conv, expand) for _ in range(num_blocks)
        ])
        
        # 모든 블록의 출력을 하나로 합치므로 입력 채널이 (dim * num_blocks)가 됨
        self.fusion_conv = nn.Conv2d(dim * num_blocks, dim, 1, 1, 0) 
        self.conv_after_blocks = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv_out = nn.Conv2d(dim, out_channels, 3, 1, 1)

    def forward(self, x):
        x = x / self.img_range
        
        # 1. 초기 특징 추출
        feat = self.conv_in(x)
        
        # 2. Dense Skip Connections
        block_features = []
        curr_feat = feat
        for block in self.blocks:
            curr_feat = block(curr_feat)
            block_features.append(curr_feat) # 층마다 특징 저장
            
        # 3. 잊어버린 정보들을 모두 합침 (Concatenation)
        dense_feat = torch.cat(block_features, dim=1)
        fused_feat = self.fusion_conv(dense_feat)
        
        # 4. 최종 복원 (Global Residual)
        out = self.conv_after_blocks(fused_feat) + feat
        out = self.conv_out(out)
        
        out = out * self.img_range
        return out
# class MambaIRScratch(nn.Module):
#     """
#     MambaIR-Scratch: 스크래치 복원 특화 MambaIR

#     Args:
#         inp_channels  (int): 입력 채널 (default: 3)
#         out_channels  (int): 출력 채널 (default: 3)
#         dim           (int): feature dimension (default: 48)
#         num_blocks    (int): ScratchMambaBlock 반복 횟수 (default: 8)
#         d_state       (int): Mamba SSM state 차원 (default: 16)
#         d_conv        (int): Mamba conv 크기 (default: 4)
#         expand        (int): Mamba expand ratio (default: 2)
#         img_range     (float): 픽셀 정규화 범위 (default: 1.0)
#     """
#     def __init__(
#         self,
#         inp_channels=3,
#         out_channels=3,
#         dim=48,
#         num_blocks=8,
#         d_state=16,
#         d_conv=4,
#         expand=2,
#         img_range=1.0,
#     ):
#         super().__init__()
#         self.img_range = img_range
#         # 픽셀 평균 정규화용
#         mean = torch.Tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1)
#         self.register_buffer('mean', mean)

#         # ─ Shallow Feature Extraction ─
#         self.conv_in = nn.Conv2d(inp_channels, dim, 3, 1, 1)

#         # ─ Deep Feature Extraction (ScratchMambaBlocks) ─
#         self.blocks = nn.Sequential(*[
#             ScratchMambaBlock(dim, d_state, d_conv, expand)
#             for _ in range(num_blocks)
#         ])
#         self.conv_after_blocks = nn.Conv2d(dim, dim, 3, 1, 1)

#         # ─ Reconstruction ─
#         self.conv_out = nn.Conv2d(dim, out_channels, 3, 1, 1)

#     def forward(self, x):
#         # 픽셀 정규화
#         x = (x - self.mean * self.img_range) / self.img_range

#         # Shallow feature
#         feat_shallow = self.conv_in(x)

#         # Deep feature
#         feat_deep = self.blocks(feat_shallow)
#         feat_deep = self.conv_after_blocks(feat_deep)

#         # Residual + Reconstruction
#         out = self.conv_out(feat_shallow + feat_deep)

#         # 역정규화
#         out = out * self.img_range + self.mean * self.img_range
#         return out
