# Copyright (c) Meta Platforms, Inc. and affiliates.

import os.path as osp
from typing import Optional
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only

from . import logger, pl_logger, EXPERIMENTS_PATH
from .data import modules as data_modules
from .module import GenericModule


class CleanProgressBar(pl.callbacks.TQDMProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)  # don't show the version number
        items.pop("loss", None)
        return items


class SeedingCallback(pl.callbacks.Callback):
    def on_epoch_start_(self, trainer, module):
        seed = module.cfg.experiment.seed
        is_overfit = module.cfg.training.trainer.get("overfit_batches", 0) > 0
        if trainer.training and not is_overfit:
            seed = seed + trainer.current_epoch

        # Temporarily disable the logging (does not seem to work?)
        pl_logger.disabled = True
        try:
            pl.seed_everything(seed, workers=True)
        finally:
            pl_logger.disabled = False

    def on_train_epoch_start(self, *args, **kwargs):
        self.on_epoch_start_(*args, **kwargs)

    def on_validation_epoch_start(self, *args, **kwargs):
        self.on_epoch_start_(*args, **kwargs)

    def on_test_epoch_start(self, *args, **kwargs):
        self.on_epoch_start_(*args, **kwargs)


class ConsoleLogger(pl.callbacks.Callback):
    @rank_zero_only
    def on_train_epoch_start(self, trainer, module):
        logger.info(
            "New training epoch %d for experiment '%s'.",
            module.current_epoch,
            module.cfg.experiment.name,
        )

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, module):
        results = {
            **dict(module.metrics_val.items()),
            **dict(module.losses_val.items()),
        }
        results = [f"{k} {v.compute():.3E}" for k, v in results.items()]
        logger.info(f'[Validation] {{{", ".join(results)}}}')


def find_last_checkpoint_path(experiment_dir):
    cls = pl.callbacks.ModelCheckpoint
    path = osp.join(experiment_dir, cls.CHECKPOINT_NAME_LAST + cls.FILE_EXTENSION)
    if osp.exists(path):
        return path
    else:
        return None


def prepare_experiment_dir(experiment_dir, cfg, rank):
    config_path = osp.join(experiment_dir, "config.yaml")
    last_checkpoint_path = find_last_checkpoint_path(experiment_dir)
    if last_checkpoint_path is not None:
        if rank == 0:
            logger.info(
                "Resuming the training from checkpoint %s", last_checkpoint_path
            )
        if osp.exists(config_path):
            with open(config_path, "r") as fp:
                cfg_prev = OmegaConf.create(fp.read())
            compare_keys = ["experiment", "data", "model", "training"]
            if OmegaConf.masked_copy(cfg, compare_keys) != OmegaConf.masked_copy(
                cfg_prev, compare_keys
            ):
                raise ValueError(
                    "Attempting to resume training with a different config: "
                    f"{OmegaConf.masked_copy(cfg, compare_keys)} vs "
                    f"{OmegaConf.masked_copy(cfg_prev, compare_keys)}"
                )
    if rank == 0:
        Path(experiment_dir).mkdir(exist_ok=True, parents=True)
        with open(config_path, "w") as fp:
            OmegaConf.save(cfg, fp)
    return last_checkpoint_path


def train(cfg: DictConfig, job_id: Optional[int] = None):
    torch.set_float32_matmul_precision("medium")
    OmegaConf.resolve(cfg)
    rank = rank_zero_only.rank

    if rank == 0:
        logger.info("Starting training with config:\n%s", OmegaConf.to_yaml(cfg))
    if cfg.experiment.gpus in (None, 0):
        logger.warning("Will train on CPU...")
        cfg.experiment.gpus = 0
    elif not torch.cuda.is_available():
        raise ValueError("Requested GPU but no NVIDIA drivers found.")
    pl.seed_everything(cfg.experiment.seed, workers=True)

    init_checkpoint_path = cfg.training.get("finetune_from_checkpoint")
    if init_checkpoint_path is not None:
        logger.info("Initializing the model from checkpoint %s.", init_checkpoint_path)
        model = GenericModule.load_from_checkpoint(
            init_checkpoint_path, strict=True, find_best=False, cfg=cfg
        )
    else:
        model = GenericModule(cfg)
    if rank == 0:
        logger.info("Network:\n%s", model.model)

    experiment_dir = osp.join(EXPERIMENTS_PATH, cfg.experiment.name)
    last_checkpoint_path = prepare_experiment_dir(experiment_dir, cfg, rank)
    checkpointing_epoch = pl.callbacks.ModelCheckpoint(
        dirpath=experiment_dir,
        filename="checkpoint-{epoch:02d}",
        save_last=True,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        verbose=True,
        **cfg.training.checkpointing,
    )
    checkpointing_step = pl.callbacks.ModelCheckpoint(
        dirpath=experiment_dir,
        filename="checkpoint-{step}",
        save_last=True,
        every_n_train_steps=20000,
        verbose=True,
        **cfg.training.checkpointing,
    )
    checkpointing_step.CHECKPOINT_NAME_LAST = "last-step"

    strategy = None
    if cfg.experiment.gpus > 1:
        strategy = pl.strategies.DDPStrategy(find_unused_parameters=False)
        for split in ["train", "val"]:
            cfg.data["loading"][split].batch_size = (
                cfg.data["loading"][split].batch_size // cfg.experiment.gpus
            )
            cfg.data["loading"][split].num_workers = int(
                (cfg.data["loading"][split].num_workers + cfg.experiment.gpus - 1)
                / cfg.experiment.gpus
            )
    data = data_modules[cfg.data.get("name", "mapillary")](cfg.data)

    tb_args = {"name": cfg.experiment.name, "version": ""}
    tb = pl.loggers.TensorBoardLogger(EXPERIMENTS_PATH, **tb_args)

    callbacks = [
        checkpointing_epoch,
        checkpointing_step,
        pl.callbacks.LearningRateMonitor(),
        SeedingCallback(),
        CleanProgressBar(),
        ConsoleLogger(),
    ]
    if cfg.experiment.gpus > 0:
        callbacks.append(pl.callbacks.DeviceStatsMonitor())

    trainer = pl.Trainer(
        default_root_dir=experiment_dir,
        detect_anomaly=False,
        enable_model_summary=False,
        sync_batchnorm=True,
        enable_checkpointing=True,
        logger=tb,
        callbacks=callbacks,
        strategy=strategy,
        check_val_every_n_epoch=1,
        accelerator="gpu",
        num_nodes=1,
        **cfg.training.trainer,
    )
    trainer.fit(model, data, ckpt_path=last_checkpoint_path)


@hydra.main(
    config_path=osp.join(osp.dirname(__file__), "conf"), config_name="orienternet"
)
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
