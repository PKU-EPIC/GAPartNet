from typing import List

import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(spconv.SparseModule):
    def __init__(
        self, in_channels: int, out_channels: int, norm_fn: nn.Module, indice_key=None
    ):
        super().__init__()

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            # assert False
            self.shortcut = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, \
                bias=False),
                norm_fn(out_channels),
            )

        self.conv1 = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels, out_channels, kernel_size=3,
                padding=1, bias=False, indice_key=indice_key,
            ),
            norm_fn(out_channels),
        )

        self.conv2 = spconv.SparseSequential(
            spconv.SubMConv3d(
                out_channels, out_channels, kernel_size=3,
                padding=1, bias=False, indice_key=indice_key,
            ),
            norm_fn(out_channels),
        )

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        shortcut = self.shortcut(x)

        x = self.conv1(x)
        x = x.replace_feature(F.relu(x.features))

        x = self.conv2(x)
        x = x.replace_feature(F.relu(x.features + shortcut.features))

        return x


class UBlock(nn.Module):
    def __init__(
        self,
        channels: List[int],
        block_fn: nn.Module,
        block_repeat: int,
        norm_fn: nn.Module,
        indice_key_id: int = 1,
    ):
        super().__init__()

        self.channels = channels

        encoder_blocks = [
            block_fn(
                channels[0], channels[0], norm_fn, indice_key=f"subm{indice_key_id}"
            )
            for _ in range(block_repeat)
        ]
        self.encoder_blocks = spconv.SparseSequential(*encoder_blocks)

        if len(channels) > 1:
            self.downsample = spconv.SparseSequential(
                spconv.SparseConv3d(
                    channels[0], channels[1], kernel_size=2, stride=2,
                    bias=False, indice_key=f"spconv{indice_key_id}",
                ),
                norm_fn(channels[1]),
                nn.ReLU(),
            )

            self.ublock = UBlock(
                channels[1:], block_fn, block_repeat, norm_fn, indice_key_id + 1
            )

            self.upsample = spconv.SparseSequential(
                spconv.SparseInverseConv3d(
                    channels[1], channels[0], kernel_size=2,
                    bias=False, indice_key=f"spconv{indice_key_id}",
                ),
                norm_fn(channels[0]),
                nn.ReLU(),
            )

            decoder_blocks = [
                block_fn(
                    channels[0] * 2, channels[0], norm_fn,
                    indice_key=f"subm{indice_key_id}",
                ),
            ]
            for _ in range(block_repeat -1):
                decoder_blocks.append(
                    block_fn(
                        channels[0], channels[0], norm_fn,
                        indice_key=f"subm{indice_key_id}",
                    )
                )
            self.decoder_blocks = spconv.SparseSequential(*decoder_blocks)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        x = self.encoder_blocks(x)
        shortcut = x

        if len(self.channels) > 1:
            x = self.downsample(x)
            x = self.ublock(x)
            x = self.upsample(x)

            x = x.replace_feature(torch.cat([x.features, shortcut.features],\
                 dim=-1))
            x = self.decoder_blocks(x)

        return x


class SparseUNet(nn.Module):
    def __init__(self, stem: nn.Module, ublock: UBlock):
        super().__init__()

        self.stem = stem
        self.ublock = ublock

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        x = self.ublock(x)
        return x

    @classmethod
    def build(
        cls,
        in_channels: int,
        channels: List[int],
        block_repeat: int,
        norm_fn: nn.Module,
        without_stem: bool = False,
    ):
        if not without_stem:
            stem = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels, channels[0], kernel_size=3,
                    padding=1, bias=False, indice_key="subm1",
                ),
                norm_fn(channels[0]),
                nn.ReLU(),
            )
        else:
            stem = spconv.SparseSequential(
                norm_fn(channels[0]),
                nn.ReLU(),
            )

        block = UBlock(channels, ResBlock, block_repeat, norm_fn, \
            indice_key_id=1)

        return SparseUNet(stem, block)



class UBlock_NoSkip(nn.Module):
    def __init__(
        self,
        channels: List[int],
        block_fn: nn.Module,
        block_repeat: int,
        norm_fn: nn.Module,
        indice_key_id: int = 1,
    ):
        super().__init__()

        self.channels = channels

        encoder_blocks = [
            block_fn(
                channels[0], channels[0], norm_fn, indice_key=f"subm{indice_key_id}"
            )
            for _ in range(block_repeat)
        ]
        self.encoder_blocks = spconv.SparseSequential(*encoder_blocks)

        if len(channels) > 1:
            self.downsample = spconv.SparseSequential(
                spconv.SparseConv3d(
                    channels[0], channels[1], kernel_size=2, stride=2,
                    bias=False, indice_key=f"spconv{indice_key_id}",
                ),
                norm_fn(channels[1]),
                nn.ReLU(),
            )

            self.ublock = UBlock(
                channels[1:], block_fn, block_repeat, norm_fn, indice_key_id + 1
            )

            self.upsample = spconv.SparseSequential(
                spconv.SparseInverseConv3d(
                    channels[1], channels[0], kernel_size=2,
                    bias=False, indice_key=f"spconv{indice_key_id}",
                ),
                norm_fn(channels[0]),
                nn.ReLU(),
            )

            decoder_blocks = [
                block_fn(
                    channels[0], channels[0], norm_fn,
                    indice_key=f"subm{indice_key_id}",
                ),
            ]
            for _ in range(block_repeat -1):
                decoder_blocks.append(
                    block_fn(
                        channels[0], channels[0], norm_fn,
                        indice_key=f"subm{indice_key_id}",
                    )
                )
            self.decoder_blocks = spconv.SparseSequential(*decoder_blocks)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        x = self.encoder_blocks(x)
        # shortcut = x

        if len(self.channels) > 1:
            x = self.downsample(x)
            x = self.ublock(x)
            x = self.upsample(x)

            # x = x.replace_feature(torch.cat([x.features, shortcut.features],\
            #      dim=-1))
            x = self.decoder_blocks(x)

        return x


class SparseUNet_NoSkip(nn.Module):
    def __init__(self, stem: nn.Module, ublock: UBlock_NoSkip):
        super().__init__()

        self.stem = stem
        self.ublock = ublock

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        x = self.ublock(x)
        return x

    @classmethod
    def build(
        cls,
        in_channels: int,
        channels: List[int],
        block_repeat: int,
        norm_fn: nn.Module,
        without_stem: bool = False,
    ):
        if not without_stem:
            stem = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels, channels[0], kernel_size=3,
                    padding=1, bias=False, indice_key="subm1",
                ),
                norm_fn(channels[0]),
                nn.ReLU(),
            )
        else:
            stem = spconv.SparseSequential(
                norm_fn(channels[0]),
                nn.ReLU(),
            )

        block = UBlock(channels, ResBlock, block_repeat, norm_fn, \
            indice_key_id=1)

        return SparseUNet(stem, block)
