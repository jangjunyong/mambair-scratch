"""
Scratch-specific Loss Functions
  - CharbonnierLoss     : L1보다 outlier에 강건
  - EdgeLoss            : Sobel 기반 경계 선명도 보존
  - SSIMLoss            : 구조적 유사도 최적화
  - FFTLoss             : 주파수 도메인 손실 (미세 질감, 스크래치 복원)
  - ScratchLoss         : 위 4가지를 가중합한 최종 Loss

배치 위치: basicsr/losses/scratch_losses.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from basicsr.utils.registry import LOSS_REGISTRY


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.sqrt(diff * diff + self.eps ** 2).mean()


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_h = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_v = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer('sobel_h', sobel_h)
        self.register_buffer('sobel_v', sobel_v)

    def _edge(self, x):
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        edge_h = F.conv2d(gray, self.sobel_h, padding=1)
        edge_v = F.conv2d(gray, self.sobel_v, padding=1)
        return torch.sqrt(edge_h ** 2 + edge_v ** 2 + 1e-6)

    def forward(self, pred, target):
        return F.l1_loss(self._edge(pred), self._edge(target))


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, C1=0.01**2, C2=0.03**2):
        super().__init__()
        self.window_size = window_size
        self.C1 = C1
        self.C2 = C2
        kernel = self._gaussian_kernel(window_size)
        self.register_buffer('kernel', kernel)

    def _gaussian_kernel(self, size, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.outer(g).unsqueeze(0).unsqueeze(0)

    def _ssim_channel(self, x, y):
        pad = self.window_size // 2
        k = self.kernel
        mu_x = F.conv2d(x, k, padding=pad)
        mu_y = F.conv2d(y, k, padding=pad)
        mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x*mu_y
        sig_x  = F.conv2d(x*x, k, padding=pad) - mu_x2
        sig_y  = F.conv2d(y*y, k, padding=pad) - mu_y2
        sig_xy = F.conv2d(x*y, k, padding=pad) - mu_xy
        ssim_map = ((2*mu_xy + self.C1)*(2*sig_xy + self.C2)) / \
                   ((mu_x2 + mu_y2 + self.C1)*(sig_x + sig_y + self.C2))
        return ssim_map.mean()

    def forward(self, pred, target):
        C = pred.shape[1]
        return 1.0 - sum(
            self._ssim_channel(pred[:, c:c+1], target[:, c:c+1])
            for c in range(C)
        ) / C


class FFTLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_fft   = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        return torch.mean(torch.abs(pred_fft - target_fft))


@LOSS_REGISTRY.register()
class ScratchLoss(nn.Module):
    """
    스크래치 복원 특화 복합 Loss
    L = w_char * Charbonnier + w_edge * Edge + w_ssim * SSIM + w_fft * FFT

    YAML 예시:
      pixel_opt:
        type: ScratchLoss
        loss_weight: 1.0
        w_char: 1.0
        w_edge: 0.2
        w_ssim: 0.1
        w_fft:  0.1
        eps: !!float 1e-3
    """
    def __init__(
        self,
        loss_weight=1.0,
        w_char=1.0,
        w_edge=0.2,
        w_ssim=0.1,
        w_fft=0.1,
        eps=1e-3,
        reduction='mean',
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.w_char = w_char
        self.w_edge = w_edge
        self.w_ssim = w_ssim
        self.w_fft  = w_fft
        self.char_loss = CharbonnierLoss(eps=eps)
        self.edge_loss = EdgeLoss()
        self.ssim_loss = SSIMLoss()
        self.fft_loss  = FFTLoss()

    def forward(self, pred, target, **kwargs):
        l_char = self.char_loss(pred, target)
        l_edge = self.edge_loss(pred, target)
        l_ssim = self.ssim_loss(pred, target)
        l_fft  = self.fft_loss(pred, target)
        total  = (self.w_char * l_char
                + self.w_edge * l_edge
                + self.w_ssim * l_ssim
                + self.w_fft  * l_fft)
        return self.loss_weight * total
