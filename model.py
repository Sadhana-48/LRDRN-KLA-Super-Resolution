import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):

    def __init__(self, channels=64, growth_channels=32, num_layers=4):
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(num_layers):
            input_channels = channels + i * growth_channels

            self.layers.append(
                nn.Conv2d(
                    input_channels,
                    growth_channels,
                    kernel_size=3,
                    padding=1
                )
            )

        self.lff = nn.Conv2d(
            channels + num_layers * growth_channels,
            channels,
            kernel_size=1
        )

    def forward(self, x):

        features = [x]

        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            out = F.relu(out, inplace=True)
            features.append(out)

        fused = self.lff(torch.cat(features, dim=1))

        return x + fused * 0.2


class LRDRN(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=64,
        num_blocks=4
    ):
        super().__init__()

        self.head = nn.Conv2d(
            in_channels,
            features,
            kernel_size=3,
            padding=1
        )

        self.rdbs = nn.ModuleList([
            ResidualDenseBlock(
                channels=features,
                growth_channels=32,
                num_layers=4
            )
            for _ in range(num_blocks)
        ])

        self.gff = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            padding=1
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(
                features,
                features * 4,
                kernel_size=3,
                padding=1
            ),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

        self.tail = nn.Conv2d(
            features,
            out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        bicubic = F.interpolate(
            x,
            scale_factor=2,
            mode="bicubic",
            align_corners=False
        )

        shallow = self.head(x)

        features = shallow

        for block in self.rdbs:
            features = block(features)

        features = self.gff(features)

        features = features + shallow

        features = self.upsample(features)

        residual = self.tail(features)

        output = bicubic + residual

        return output
