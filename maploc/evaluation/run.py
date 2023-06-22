# Copyright (c) Meta Platforms, Inc. and affiliates.

import functools
from itertools import islice
from typing import Callable, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torchmetrics import MetricCollection
from pytorch_lightning import seed_everything
from tqdm import tqdm

from .. import logger, EXPERIMENTS_PATH
from ..data.torch import collate, unbatch_to_device
from ..models.voting import argmax_xyr, fuse_gps
from ..models.metrics import AngleError, LateralLongitudinalError, Location2DError
from ..models.sequential import GPSAligner, RigidAligner
from ..module import GenericModule
from ..utils.io import download_file, DATA_URL
from .viz import plot_example_single, plot_example_sequential
from .utils import write_dump


pretrained_models = dict(
    OrienterNet_MGL=("orienternet_mgl.ckpt", dict(num_rotations=256)),
)


def resolve_checkpoint_path(experiment_or_path: str) -> Path:
    path = Path(experiment_or_path)
    if not path.exists():
        # provided name of experiment
        path = Path(EXPERIMENTS_PATH, *experiment_or_path.split("/"))
        if not path.exists():
            if experiment_or_path in set(p for p, _ in pretrained_models.values()):
                download_file(f"{DATA_URL}/{experiment_or_path}", path)
            else:
                raise FileNotFoundError(path)
    if path.is_file():
        return path
    # provided only the experiment name
    maybe_path = path / "last-step.ckpt"
    if not maybe_path.exists():
        maybe_path = path / "step.ckpt"
    if not maybe_path.exists():
        raise FileNotFoundError(f"Could not find any checkpoint in {path}.")
    return maybe_path


@torch.no_grad()
def evaluate_single_image(
    dataloader: torch.utils.data.DataLoader,
    model: GenericModule,
    num: Optional[int] = None,
    callback: Optional[Callable] = None,
    progress: bool = True,
    mask_index: Optional[Tuple[int]] = None,
    has_gps: bool = False,
):
    ppm = model.model.conf.pixel_per_meter
    metrics = MetricCollection(model.model.metrics())
    metrics["directional_error"] = LateralLongitudinalError(ppm)
    if has_gps:
        metrics["xy_gps_error"] = Location2DError("uv_gps", ppm)
        metrics["xy_fused_error"] = Location2DError("uv_fused", ppm)
        metrics["yaw_fused_error"] = AngleError("yaw_fused")
    metrics = metrics.to(model.device)

    for i, batch_ in enumerate(
        islice(tqdm(dataloader, total=num, disable=not progress), num)
    ):
        batch = model.transfer_batch_to_device(batch_, model.device, i)
        # Ablation: mask semantic classes
        if mask_index is not None:
            mask = batch["map"][0, mask_index[0]] == (mask_index[1] + 1)
            batch["map"][0, mask_index[0]][mask] = 0
        pred = model(batch)

        if has_gps:
            (uv_gps,) = pred["uv_gps"] = batch["uv_gps"]
            pred["log_probs_fused"] = fuse_gps(
                pred["log_probs"], uv_gps, ppm, sigma=batch["accuracy_gps"]
            )
            uvt_fused = argmax_xyr(pred["log_probs_fused"])
            pred["uv_fused"] = uvt_fused[..., :2]
            pred["yaw_fused"] = uvt_fused[..., -1]
            del uv_gps, uvt_fused

        results = metrics(pred, batch)
        if callback is not None:
            callback(
                i, model, unbatch_to_device(pred), unbatch_to_device(batch_), results
            )
        del batch_, batch, pred, results

    return metrics.cpu()


@torch.no_grad()
def evaluate_sequential(
    dataset: torch.utils.data.Dataset,
    chunk2idx: Dict,
    model: GenericModule,
    num: Optional[int] = None,
    shuffle: bool = False,
    callback: Optional[Callable] = None,
    progress: bool = True,
    num_rotations: int = 512,
    mask_index: Optional[Tuple[int]] = None,
    has_gps: bool = True,
):
    chunk_keys = list(chunk2idx)
    if shuffle:
        chunk_keys = [chunk_keys[i] for i in torch.randperm(len(chunk_keys))]
    if num is not None:
        chunk_keys = chunk_keys[:num]
    lengths = [len(chunk2idx[k]) for k in chunk_keys]
    logger.info(
        "Min/max/med lengths: %d/%d/%d, total number of images: %d",
        min(lengths),
        np.median(lengths),
        max(lengths),
        sum(lengths),
    )
    viz = callback is not None

    metrics = MetricCollection(model.model.metrics())
    ppm = model.model.conf.pixel_per_meter
    metrics["directional_error"] = LateralLongitudinalError(ppm)
    metrics["xy_seq_error"] = Location2DError("uv_seq", ppm)
    metrics["yaw_seq_error"] = AngleError("yaw_seq")
    metrics["directional_seq_error"] = LateralLongitudinalError(ppm, key="uv_seq")
    if has_gps:
        metrics["xy_gps_error"] = Location2DError("uv_gps", ppm)
        metrics["xy_gps_seq_error"] = Location2DError("uv_gps_seq", ppm)
        metrics["yaw_gps_seq_error"] = AngleError("yaw_gps_seq")
    metrics = metrics.to(model.device)

    keys_save = ["uvr_max", "uv_max", "yaw_max", "uv_expectation"]
    if has_gps:
        keys_save.append("uv_gps")
    if viz:
        keys_save.append("log_probs")

    for chunk_index, key in enumerate(tqdm(chunk_keys, disable=not progress)):
        indices = chunk2idx[key]
        aligner = RigidAligner(track_priors=viz, num_rotations=num_rotations)
        if has_gps:
            aligner_gps = GPSAligner(track_priors=viz, num_rotations=num_rotations)
        batches = []
        preds = []
        for i in indices:
            data = dataset[i]
            data = model.transfer_batch_to_device(data, model.device, 0)
            pred = model(collate([data]))

            canvas = data["canvas"]
            data["xy_geo"] = xy = canvas.to_xy(data["uv"].double())
            data["yaw"] = yaw = data["roll_pitch_yaw"][-1].double()
            aligner.update(pred["log_probs"][0], canvas, xy, yaw)

            if has_gps:
                (uv_gps) = pred["uv_gps"] = data["uv_gps"][None]
                xy_gps = canvas.to_xy(uv_gps.double())
                aligner_gps.update(xy_gps, data["accuracy_gps"], canvas, xy, yaw)

            if not viz:
                data.pop("image")
                data.pop("map")
            batches.append(data)
            preds.append({k: pred[k][0] for k in keys_save})
            del pred

        xy_gt = torch.stack([b["xy_geo"] for b in batches])
        yaw_gt = torch.stack([b["yaw"] for b in batches])
        aligner.compute()
        xy_seq, yaw_seq = aligner.transform(xy_gt, yaw_gt)
        if has_gps:
            aligner_gps.compute()
            xy_gps_seq, yaw_gps_seq = aligner_gps.transform(xy_gt, yaw_gt)
        results = []
        for i in range(len(indices)):
            preds[i]["uv_seq"] = batches[i]["canvas"].to_uv(xy_seq[i]).float()
            preds[i]["yaw_seq"] = yaw_seq[i].float()
            if has_gps:
                preds[i]["uv_gps_seq"] = (
                    batches[i]["canvas"].to_uv(xy_gps_seq[i]).float()
                )
                preds[i]["yaw_gps_seq"] = yaw_gps_seq[i].float()
            results.append(metrics(preds[i], batches[i]))
        if viz:
            callback(chunk_index, model, batches, preds, results, aligner)
        del aligner, preds, batches, results
    return metrics.cpu()


def evaluate(
    experiment: str,
    cfg: DictConfig,
    dataset,
    split: str,
    sequential: bool = False,
    output_dir: Optional[Path] = None,
    callback: Optional[Callable] = None,
    num_workers: int = 1,
    viz_kwargs=None,
    **kwargs,
):
    if experiment in pretrained_models:
        experiment, cfg_override = pretrained_models[experiment]
        cfg = OmegaConf.merge(OmegaConf.create(dict(model=cfg_override)), cfg)

    logger.info("Evaluating model %s with config %s", experiment, cfg)
    checkpoint_path = resolve_checkpoint_path(experiment)
    model = GenericModule.load_from_checkpoint(
        checkpoint_path, cfg=cfg, find_best=not experiment.endswith(".ckpt")
    )
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    dataset.prepare_data()
    dataset.setup()

    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)
        if callback is None:
            if sequential:
                callback = plot_example_sequential
            else:
                callback = plot_example_single
            callback = functools.partial(
                callback, out_dir=output_dir, **(viz_kwargs or {})
            )
    kwargs = {**kwargs, "callback": callback}

    seed_everything(dataset.cfg.seed)
    if sequential:
        dset, chunk2idx = dataset.sequence_dataset(split, **cfg.chunking)
        metrics = evaluate_sequential(dset, chunk2idx, model, **kwargs)
    else:
        loader = dataset.dataloader(split, shuffle=True, num_workers=num_workers)
        metrics = evaluate_single_image(loader, model, **kwargs)

    results = metrics.compute()
    logger.info("All results: %s", results)
    if output_dir is not None:
        write_dump(output_dir, experiment, cfg, results, metrics)
        logger.info("Outputs have been written to %s.", output_dir)
    return metrics
