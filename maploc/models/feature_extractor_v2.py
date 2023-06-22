import logging

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor

from .base import BaseModel

logger = logging.getLogger(__name__)


class DecoderBlock(nn.Module):
    def __init__(
        self, previous, out, ksize=3, num_convs=1, norm=nn.BatchNorm2d, padding="zeros"
    ):
        super().__init__()
        layers = []
        for i in range(num_convs):
            conv = nn.Conv2d(
                previous if i == 0 else out,
                out,
                kernel_size=ksize,
                padding=ksize // 2,
                bias=norm is None,
                padding_mode=padding,
            )
            layers.append(conv)
            if norm is not None:
                layers.append(norm(out))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, previous, skip):
        _, _, hp, wp = previous.shape
        _, _, hs, ws = skip.shape
        scale = 2 ** np.round(np.log2(np.array([hs / hp, ws / wp])))
        upsampled = nn.functional.interpolate(
            previous, scale_factor=scale.tolist(), mode="bilinear", align_corners=False
        )
        # If the shape of the input map `skip` is not a multiple of 2,
        # it will not match the shape of the upsampled map `upsampled`.
        # If the downsampling uses ceil_mode=False, we nedd to crop `skip`.
        # If it uses ceil_mode=True (not supported here), we should pad it.
        _, _, hu, wu = upsampled.shape
        _, _, hs, ws = skip.shape
        if (hu <= hs) and (wu <= ws):
            skip = skip[:, :, :hu, :wu]
        elif (hu >= hs) and (wu >= ws):
            skip = nn.functional.pad(skip, [0, wu - ws, 0, hu - hs])
        else:
            raise ValueError(
                f"Inconsistent skip vs upsampled shapes: {(hs, ws)}, {(hu, wu)}"
            )

        return self.layers(skip) + upsampled


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, **kw):
        super().__init__()
        self.first = nn.Conv2d(
            in_channels_list[-1], out_channels, 1, padding=0, bias=True
        )
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(c, out_channels, ksize=1, **kw)
                for c in in_channels_list[::-1][1:]
            ]
        )
        self.out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, layers):
        feats = None
        for idx, x in enumerate(reversed(layers.values())):
            if feats is None:
                feats = self.first(x)
            else:
                feats = self.blocks[idx - 1](feats, x)
        out = self.out(feats)
        return out


def remove_conv_stride(conv):
    conv_new = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        bias=conv.bias is not None,
        stride=1,
        padding=conv.padding,
    )
    conv_new.weight = conv.weight
    conv_new.bias = conv.bias
    return conv_new


class FeatureExtractor(BaseModel):
    default_conf = {
        "pretrained": True,
        "input_dim": 3,
        "output_dim": 128,  # # of channels in output feature maps
        "encoder": "resnet50",  # torchvision net as string
        "remove_stride_from_first_conv": False,
        "num_downsample": None,  # how many downsample block
        "decoder_norm": "nn.BatchNorm2d",  # normalization ind decoder blocks
        "do_average_pooling": False,
        "checkpointed": False,  # whether to use gradient checkpointing
    }
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def build_encoder(self, conf):
        assert isinstance(conf.encoder, str)
        if conf.pretrained:
            assert conf.input_dim == 3
        Encoder = getattr(torchvision.models, conf.encoder)

        kw = {}
        if conf.encoder.startswith("resnet"):
            layers = ["relu", "layer1", "layer2", "layer3", "layer4"]
            kw["replace_stride_with_dilation"] = [False, False, False]
        elif conf.encoder == "vgg13":
            layers = [
                "features.3",
                "features.8",
                "features.13",
                "features.18",
                "features.23",
            ]
        elif conf.encoder == "vgg16":
            layers = [
                "features.3",
                "features.8",
                "features.15",
                "features.22",
                "features.29",
            ]
        else:
            raise NotImplementedError(conf.encoder)

        if conf.num_downsample is not None:
            layers = layers[: conf.num_downsample]
        encoder = Encoder(weights="DEFAULT" if conf.pretrained else None, **kw)
        encoder = create_feature_extractor(encoder, return_nodes=layers)
        if conf.encoder.startswith("resnet") and conf.remove_stride_from_first_conv:
            encoder.conv1 = remove_conv_stride(encoder.conv1)

        if conf.do_average_pooling:
            raise NotImplementedError
        if conf.checkpointed:
            raise NotImplementedError

        return encoder, layers

    def _init(self, conf):
        # Preprocessing
        self.register_buffer("mean_", torch.tensor(self.mean), persistent=False)
        self.register_buffer("std_", torch.tensor(self.std), persistent=False)

        # Encoder
        self.encoder, self.layers = self.build_encoder(conf)
        s = 128
        inp = torch.zeros(1, 3, s, s)
        features = list(self.encoder(inp).values())
        self.skip_dims = [x.shape[1] for x in features]
        self.layer_strides = [s / f.shape[-1] for f in features]
        self.scales = [self.layer_strides[0]]

        # Decoder
        norm = eval(conf.decoder_norm) if conf.decoder_norm else None  # noqa
        self.decoder = FPN(self.skip_dims, out_channels=conf.output_dim, norm=norm)

        logger.debug(
            "Built feature extractor with layers {name:dim:stride}:\n"
            f"{list(zip(self.layers, self.skip_dims, self.layer_strides))}\n"
            f"and output scales {self.scales}."
        )

    def _forward(self, data):
        image = data["image"]
        image = (image - self.mean_[:, None, None]) / self.std_[:, None, None]

        skip_features = self.encoder(image)
        output = self.decoder(skip_features)
        pred = {"feature_maps": [output], "skip_features": skip_features}
        return pred
