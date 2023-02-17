import argparse
from itertools import islice
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torchmetrics import MetricCollection
from tqdm import tqdm

from . import logger
from .data import KittiDataModule
from .data.torch import collate
from .evaluation import experiments
from .models.hough_voting import argmax_xy, argmax_xyr, log_softmax_spatial
from .models.metrics import (
    angle_error,
    AngleError,
    LateralLongitudinalError,
    Location2DError,
)
from .models.sequential import RigidAligner
from .module import GenericModule

default_cfg_single = OmegaConf.create({"seed": 0})
# For the sequential evaluation, we need to center the map around the GT location,
# since random offsets would accumulate and leave only the GT location with a valid mask.
# This should not have much impact on the results.
default_cfg_sequential = OmegaConf.create(
    {
        "seed": 0,
        "data": {
            "mask_radius": KittiDataModule.default_cfg["max_init_error"],
            "max_init_error": 0,
        },
        "chunking": {
            "max_length": 100,  # about 10s?
        },
    }
)


def mask_with_prior(log_probs, map_mask, yaw_init=None, rot_range=None):
    if log_probs.ndim > 3:  # BxHxWxR
        num_rotations = log_probs.shape[-1]
        angles = torch.arange(0, 360, 360 / num_rotations, device=log_probs.device)
        rot_mask = angle_error(angles, yaw_init.unsqueeze(-1)) < rot_range
        map_mask = map_mask[..., None] & rot_mask[..., None, None, :]
    masked = torch.masked_fill(log_probs, ~map_mask, -np.inf)
    return log_softmax_spatial(masked)


@torch.no_grad()
def evaluate_single_image(
    dataloader: torch.utils.data.DataLoader,
    model: GenericModule,
    num: Optional[int] = None,
    callback: Optional[Callable] = None,
    progress: bool = True,
    apply_prior_mask: bool = True,
    mask_index: Optional[Tuple[int]] = None,
):
    metrics = MetricCollection(model.model.metrics())
    ppm = model.model.conf.pixel_per_meter
    metrics["directional_error"] = LateralLongitudinalError(ppm)
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

        if apply_prior_mask:
            pred["log_probs_unmasked"] = pred["log_probs"]
            pred["log_probs"] = mask_with_prior(
                pred["log_probs"],
                batch["map_mask"],
                batch["yaw_init"],
                dataloader.dataset.cfg.max_init_error_rotation,
            )
            if pred["log_probs"].ndim == 3:
                pred["xy_max"] = argmax_xy(pred["log_probs"]).float()
            else:
                pred["xyr_max"] = argmax_xyr(pred["log_probs"])
                pred["xy_max"] = pred["xyr_max"][:, :2]
                pred["yaw_max"] = pred["xyr_max"][:, 2]
        results = metrics(pred, batch)
        if callback is not None:
            callback(i, model, batch_, pred, results)
        del batch_, batch, pred
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
    apply_prior_mask: bool = True,
    mask_index: Optional[Tuple[int]] = None,
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
    metrics["directional_seq_error"] = LateralLongitudinalError(ppm, key="xy_seq")
    metrics["xy_seq_error"] = Location2DError("xy_seq", ppm)
    metrics["yaw_seq_error"] = AngleError("yaw_seq")
    metrics = metrics.to(model.device)

    for chunk_index, key in enumerate(tqdm(chunk_keys, disable=not progress)):
        indices = chunk2idx[key]
        aligner = RigidAligner(track_priors=viz, num_rotations=num_rotations)
        batches = []
        preds = []
        for i in indices:
            data = dataset[i]
            data = model.transfer_batch_to_device(data, model.device, 0)
            pred = model(collate([data]))

            if apply_prior_mask:
                pred["log_probs"] = mask_with_prior(
                    pred["log_probs"],
                    data["map_mask"],
                    data["yaw_init"],
                    dataset.cfg.max_init_error_rotation + 5,
                )
                pred["xyr_max"] = argmax_xyr(pred["log_probs"])
                pred["xy_max"] = pred["xyr_max"][:, :2]
                pred["yaw_max"] = pred["xyr_max"][:, 2]

            canvas = data["canvas"]
            data["xy_geo"] = xy = canvas.to_xy(data["xy"].double())
            data["yaw"] = yaw = data["roll_pitch_yaw"][-1].double()
            aligner.update(pred["log_probs"][0], canvas, xy, yaw)
            preds.append(
                {
                    k: pred[k][0]
                    for k in ["xyr_max", "xy_max", "yaw_max", "xy_expectation"]
                }
            )
            batches.append(data)
            if viz:
                preds[-1]["log_probs"] = pred["log_probs"][0]
            else:
                batches[-1].pop("image")
                batches[-1].pop("map")
        aligner.compute()
        xy_gt = torch.stack([b["xy_geo"] for b in batches])
        yaw_gt = torch.stack([b["yaw"] for b in batches])
        xy_seq, yaw_seq = aligner.transform(xy_gt, yaw_gt)
        results = []
        for i in range(len(indices)):
            preds[i]["xy_seq"] = batches[i]["canvas"].to_uv(xy_seq[i]).float()
            preds[i]["yaw_seq"] = yaw_seq[i].float()
            results.append(metrics(preds[i], batches[i]))
        if viz:
            callback(chunk_index, model, batches, preds, results, aligner)
        del aligner, preds, batches, results
    return metrics.cpu()


def run(
    stage: str,
    experiment: str,
    cfg=None,
    output_dir=None,
    sequential: bool = False,
    callback: Optional[Callable] = None,
    thresholds=(1, 3, 5),
    **kwargs,
):
    cfg = cfg or {}
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    default = default_cfg_sequential if sequential else default_cfg_single
    cfg = OmegaConf.merge(default, cfg)
    logger.info("Working on (%s, %s)", experiment, cfg)
    model = GenericModule.load_for_evaluation(experiment, cfg=cfg)

    dataset = KittiDataModule(cfg.get("data", {}))
    dataset.prepare_data()
    dataset.setup()

    if output_dir is not None:
        raise NotImplementedError
    kwargs = {**kwargs, "callback": callback}

    seed_everything(cfg.seed)
    if sequential:
        dset, chunk2idx = dataset.sequence_dataset(stage, cfg.chunking)
        metrics = evaluate_sequential(dset, chunk2idx, model, **kwargs)
    else:
        loader = dataset.dataloader(stage, shuffle=True, num_workers=1)
        metrics = evaluate_single_image(loader, model, **kwargs)
    logger.info("All results: %s", metrics.compute())
    keys = ["directional_error", "yaw_max_error"]
    if sequential:
        keys += ["directional_seq_error", "yaw_seq_error"]
    for k in keys:
        rec = metrics[k].recall(thresholds).double().numpy().round(2).tolist()
        logger.info("%s: %s at %s m/°", k, rec, thresholds)
    return metrics


def main(args):
    cfg = OmegaConf.from_cli(args.dotlist)
    experiment = args.experiment
    if experiment in experiments:
        experiment, override = experiments[experiment]
        cfg = OmegaConf.merge(cfg, OmegaConf.create(override))
    run(args.split, experiment, cfg, args.output_dir, args.sequential)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("dotlist", nargs="*")
    main(parser.parse_args())
