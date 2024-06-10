from typing import *
import torch.nn as nn
from config import *


class ConvolutionBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1, stride: int = 1,
                 dilation: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        return self.relu(self.bn2(self.conv2(self.bn1(self.conv1(x)))))


class EncoderBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.c1 = ConvolutionBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor):
        s = self.c1(x)
        return s, self.maxpool(s)


class AttentionBlock(nn.Module):

    def __init__(self, in_channels: List[int], out_channels: int):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels[0], out_channels, kernel_size=1, padding="same"),
            nn.BatchNorm2d(out_channels)
        )

        self.skip = nn.Sequential(
            nn.Conv2d(in_channels[1], out_channels, kernel_size=1, padding="same"),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.ReLU()
        self.psi_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding="same")
        self.sigmoid = nn.Sigmoid()

    def forward(self, gate: torch.Tensor, skip: torch.Tensor):
        wg = self.gate(gate)
        ws = self.skip(skip)
        x = self.relu(wg + ws)
        return self.sigmoid(self.psi_conv(x)) * skip


class DecoderBlock(nn.Module):

    def __init__(self, in_channels: List[int], out_channels: int):
        super().__init__()

        # in_channels = List[input_channels_gate, input_channels_skip]
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.attention = AttentionBlock(in_channels, out_channels)
        # after concatenation, the number of channels get added
        self.c1 = ConvolutionBlock(in_channels[0] + in_channels[1], out_channels)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor):
        # make the spatial dimensions of gate and skip the same
        gate = self.up1(gate)
        # now, pass the gate and skip through the attention gate to generate attention coefficients => These act as
        # the new skip connections
        skip = self.attention(gate, skip)
        # concatenate along 'channels' dimension -> Shape of tensor = [N, C, H, W] => channel axis => dim = 1
        # perform convolution
        return self.c1(torch.cat([gate, skip], dim=1))


class AttentionUNet(nn.Module):

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.e1 = EncoderBlock(in_channels=in_channels, out_channels=64)
        self.e2 = EncoderBlock(in_channels=64, out_channels=128)
        self.e3 = EncoderBlock(in_channels=128, out_channels=256)

        self.b1 = ConvolutionBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1)

        self.d1 = DecoderBlock([512, 256], 256)
        self.d2 = DecoderBlock([256, 128], 128)
        self.d3 = DecoderBlock([128, 64], 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1, padding="same")

    def forward(self, x: torch.Tensor):
        s1, x = self.e1(x)
        s2, x = self.e2(x)
        s3, x = self.e3(x)

        x = self.b1(x)

        d1 = self.d1(x, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)

        return self.final_conv(d3)
