# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import os.path as osp

import pytorch_lightning as pl
import stl.lightning.io.filesystem as stlfs
import torch
from omegaconf import DictConfig, OmegaConf, open_dict  # @manual
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
        self.model = get_model(cfg.model.get("name", "localizer_basic"))(cfg.model)
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

        checkpoint = pl.utilities.cloud_io.load(
            checkpoint_path, map_location=map_location or (lambda storage, loc: storage)
        )
        if find_best:
            best_score, best_path = None, None
            modes = {"min": torch.lt, "max": torch.gt}
            for key, state in checkpoint["callbacks"].items():
                if not key.startswith("ModelCheckpoint"):
                    continue
                mode = eval(key.replace("ModelCheckpoint", ""))["mode"]
                if best_score is None or modes[mode](
                    state["best_model_score"], best_score
                ):
                    best_score = state["best_model_score"]
                    best_path = state["best_model_path"]
            logger.info("Loading best checkpoint %s", best_path)
            if best_path != checkpoint_path:
                return cls.load_from_checkpoint(
                    best_path,
                    map_location,
                    hparams_file,
                    strict,
                    cfg,
                    find_best=False,
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

        return cls._load_model_state(checkpoint, strict=strict, cfg=cfg)

    @classmethod
    def load_for_evaluation(cls, experiment, cfg=None, device=None, find_best=True):
        path = f"manifold://psarlin/tree/maploc/experiments/{experiment}/last-step.ckpt"
        fs = stlfs.get_filesystem(path)
        if not fs.exists(path):
            path = path.replace("last-step.ckpt", "last.ckpt")
        model = GenericModule.load_from_checkpoint(
            path, strict=True, find_best=find_best, cfg=cfg
        )
        model = model.eval()
        if torch.cuda.is_available():
            model = model.cuda(device)
        return model


def find_best_checkpoint(dir_):
    """A hacky way to find the best checkpoint for an experiment. Mostly for backward
    compatibility for experiments ran before the checkpointing fix.
    """
    fs = stlfs.get_filesystem(dir_)
    ckpts = fs.glob(osp.join(dir_, "*.ckpt"))
    best_path = None
    best_value = None
    steps = []
    for path in ckpts:
        checkpoint = pl.utilities.cloud_io.load(
            path, map_location=(lambda storage, loc: storage)
        )
        info = next(iter(checkpoint["callbacks"].values()))
        value = info["current_score"]
        step = checkpoint["global_step"]
        steps.append(step)
        if best_value is None or value <= best_value:
            best_value = value
            best_path = path
    assert best_path is not None
    logger.info("Found best checkpoint with value %f", best_value)
    return best_path
