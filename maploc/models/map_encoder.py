# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn

from .base import BaseModel
from .feature_extractor import FeatureExtractor


class MapEncoder(BaseModel):
    default_conf = {
        "embedding_dim": "???",
        "output_dim": None,
        "num_classes": "???",
        "backbone": "???",
        "unary_prior": False,
    }

    def _init(self, conf):
        self.embeddings = torch.nn.ModuleDict(
            {
                k: torch.nn.Embedding(n + 1, conf.embedding_dim)
                for k, n in conf.num_classes.items()
            }
        )
        input_dim = len(conf.num_classes) * conf.embedding_dim
        output_dim = conf.output_dim
        if output_dim is None:
            output_dim = conf.backbone.output_dim
        if conf.unary_prior:
            output_dim += 1
        if conf.backbone is None:
            self.encoder = nn.Conv2d(input_dim, output_dim, 1)
        elif conf.backbone == "simple":
            self.encoder = nn.Sequential(
                nn.Conv2d(input_dim, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, output_dim, 3, padding=1),
            )
        else:
            self.encoder = FeatureExtractor(
                {
                    **conf.backbone,
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                }
            )

    def _forward(self, data):
        embeddings = [
            self.embeddings[k](data["map"][:, i])
            for i, k in enumerate(("areas", "ways", "nodes"))
        ]
        embeddings = torch.cat(embeddings, dim=-1).permute(0, 3, 1, 2)
        if isinstance(self.encoder, BaseModel):
            features = self.encoder({"image": embeddings})["feature_maps"]
        else:
            features = [self.encoder(embeddings)]
        pred = {}
        if self.conf.unary_prior:
            pred["log_prior"] = [f[:, -1] for f in features]
            features = [f[:, :-1] for f in features]
        pred["map_features"] = features
        return pred
