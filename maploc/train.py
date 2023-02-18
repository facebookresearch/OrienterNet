# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import os
from typing import Optional

import fsspec  # noqa: F401, fsspec is monkey patched by stlfs
import pytorch_lightning as pl
import stl.lightning.io.filesystem as stlfs
import torch
from omegaconf import DictConfig, OmegaConf  # @manual
from pytorch_lightning.utilities import rank_zero_only
from stl.lightning.io.filesystem import clean_path
from stl.lightning.loggers import ManifoldTensorBoardLogger
from stl.lightning.utilities.checkpoint import find_last_checkpoint_path
from tensorboard.fb.xdb import tb_log_creation_event

from . import logger, pl_logger
from .data import modules as data_modules
from .module import GenericModule


class CleanProgressBar(pl.callbacks.TQDMProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)  # don't show the version number
        items.pop("loss", None)
        return items


class SeedingCallback(pl.callbacks.Callback):
    def on_epoch_start(self, trainer, module):
        seed = module.cfg.experiment.seed
        is_overfit = module.cfg.training.trainer.get("overfit_batches", 0) > 0
        if trainer.training and not is_overfit:
            seed = seed + trainer.current_epoch

        # Temporarily disable the logging
        # TODO: does not seem to work
        pl_logger.disabled = True
        try:
            pl.seed_everything(seed, workers=True)
        finally:
            pl_logger.disabled = False


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


def prepare_experiment_dir(experiment_dir, cfg, rank):
    fs = stlfs.get_filesystem(experiment_dir)
    config_path = os.path.join(experiment_dir, "config.yaml")
    last_checkpoint_path = find_last_checkpoint_path(experiment_dir)
    if last_checkpoint_path is not None:
        if rank == 0:
            logger.info(
                "Resuming the training from checkpoint %s", last_checkpoint_path
            )
        if fs.exists(config_path):
            with fs.open(config_path, "r") as fp:
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
        fs.mkdir(experiment_dir)
        with fs.open(config_path, "w") as fp:
            OmegaConf.save(cfg, fp)
    return last_checkpoint_path


def train(cfg: DictConfig, job_id: Optional[int] = None):
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
    is_flow = job_id is not None  # FBLearner condition

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
    data = data_modules[cfg.data.get("name", "mapillary")](cfg.data)

    experiment_dir = os.path.join(cfg.experiment.root, cfg.experiment.name)
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

    tb_args = {"name": cfg.experiment.name, "version": ""}
    if cfg.experiment.root.startswith("manifold://"):
        bucket, *path = cfg.experiment.root[len("manifold://") :].split("/")
        tb = ManifoldTensorBoardLogger(
            manifold_bucket=bucket,
            manifold_path="/".join(path),
            **tb_args,
        )
        if is_flow:
            if rank == 0:
                logger.info("Running in Flow mode.")
            tb_log_creation_event(clean_path(tb.log_dir), job_id)
    else:
        tb = pl.loggers.TensorBoardLogger(cfg.experiment.root, **tb_args)

    callbacks = [
        checkpointing_epoch,
        checkpointing_step,
        pl.callbacks.LearningRateMonitor(),
        SeedingCallback(),
        CleanProgressBar(),
        ConsoleLogger(),
    ]
    if cfg.experiment.gpus > 0 and not is_flow:
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
        **cfg.training.trainer,
    )
    trainer.fit(model, data, ckpt_path=last_checkpoint_path)
