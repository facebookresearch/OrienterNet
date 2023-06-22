# Copyright (c) Meta Platforms, Inc. and affiliates.

from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from torchmetrics import MeanMetric, MetricCollection

from . import logger
from .models import get_model


class AverageKeyMeter(MeanMetric):
    def __init__(self, key, *args, **kwargs):
        self.key = key
        super().__init__(*args, **kwargs)

    def update(self, dict):
        value = dict[self.key]
        value = value[torch.isfinite(value)]
        return super().update(value)


class GenericModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        name = cfg.model.get("name")
        name = "orienternet" if name in ("localizer_bev_depth", None) else name
        self.model = get_model(name)(cfg.model)
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.metrics_val = MetricCollection(self.model.metrics(), prefix="val/")
        self.losses_val = None  # we do not know the loss keys in advance

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch):
        pred = self(batch)
        losses = self.model.loss(pred, batch)
        self.log_dict(
            {f"loss/{k}/train": v.mean() for k, v in losses.items()},
            prog_bar=True,
            rank_zero_only=True,
        )
        return losses["total"].mean()

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        losses = self.model.loss(pred, batch)
        if self.losses_val is None:
            self.losses_val = MetricCollection(
                {k: AverageKeyMeter(k).to(self.device) for k in losses},
                prefix="loss/",
                postfix="/val",
            )
        self.metrics_val(pred, batch)
        self.log_dict(self.metrics_val, sync_dist=True)
        self.losses_val.update(losses)
        self.log_dict(self.losses_val, sync_dist=True)

    def validation_epoch_start(self, batch):
        self.losses_val = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.training.lr)
        ret = {"optimizer": optimizer}
        cfg_scheduler = self.cfg.training.get("lr_scheduler")
        if cfg_scheduler is not None:
            scheduler = getattr(torch.optim.lr_scheduler, cfg_scheduler.name)(
                optimizer=optimizer, **cfg_scheduler.get("args", {})
            )
            ret["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "loss/total/val",
                "strict": True,
                "name": "learning_rate",
            }
        return ret

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        hparams_file=None,
        strict=True,
        cfg=None,
        find_best=False,
    ):
        assert hparams_file is None, "hparams are not supported."

        checkpoint = torch.load(
            checkpoint_path, map_location=map_location or (lambda storage, loc: storage)
        )
        if find_best:
            best_score, best_name = None, None
            modes = {"min": torch.lt, "max": torch.gt}
            for key, state in checkpoint["callbacks"].items():
                if not key.startswith("ModelCheckpoint"):
                    continue
                mode = eval(key.replace("ModelCheckpoint", ""))["mode"]
                if best_score is None or modes[mode](
                    state["best_model_score"], best_score
                ):
                    best_score = state["best_model_score"]
                    best_name = Path(state["best_model_path"]).name
            logger.info("Loading best checkpoint %s", best_name)
            if best_name != checkpoint_path:
                return cls.load_from_checkpoint(
                    Path(checkpoint_path).parent / best_name,
                    map_location,
                    hparams_file,
                    strict,
                    cfg,
                    find_best=False,
                )

        logger.info(
            "Using checkpoint %s from epoch %d and step %d.",
            checkpoint_path.name,
            checkpoint["epoch"],
            checkpoint["global_step"],
        )
        cfg_ckpt = checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY]
        if list(cfg_ckpt.keys()) == ["cfg"]:  # backward compatibility
            cfg_ckpt = cfg_ckpt["cfg"]
        cfg_ckpt = OmegaConf.create(cfg_ckpt)

        if cfg is None:
            cfg = {}
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        with open_dict(cfg_ckpt):
            cfg = OmegaConf.merge(cfg_ckpt, cfg)

        return pl.core.saving._load_state(cls, checkpoint, strict=strict, cfg=cfg)
