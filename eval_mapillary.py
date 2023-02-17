# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from itertools import islice
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torchmetrics import MetricCollection
from tqdm import tqdm

from . import logger
from .data.torch import collate
from .models.hough_voting import argmax_xyr, fuse_gps
from .models.metrics import AngleError, Location2DError
from .models.sequential import GPSAligner, RigidAligner
from .module import GenericModule


@torch.no_grad()
def evaluate_single_image(
    dataloader: torch.utils.data.DataLoader,
    model: GenericModule,
    num: Optional[int] = None,
    callback: Optional[Callable] = None,
    progress: bool = False,
    mask_index: Optional[Tuple[int]] = None,
    gps_sigma: float = 15.0,
):
    ppm = model.cfg.model.pixel_per_meter
    metrics = MetricCollection(model.model.metrics())
    metrics["xy_gps_error"] = Location2DError("xy_gps", ppm)
    metrics["xy_fused_error"] = Location2DError("xy_fused", ppm)
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

        (uv_gps,) = pred["xy_gps"] = batch["xy_gps"]
        log_prob_fused = fuse_gps(pred["log_probs"], uv_gps, ppm, sigma=gps_sigma)
        uvt_fused = argmax_xyr(log_prob_fused)
        pred["xy_fused"] = uvt_fused[..., :2]
        pred["yaw_fused"] = uvt_fused[..., -1]

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
    progress: bool = False,
    num_rotations: int = 512,
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
    metrics["xy_gps_error"] = Location2DError("xy_gps", ppm)
    metrics["xy_seq_error"] = Location2DError("xy_seq", ppm)
    metrics["xy_gps_seq_error"] = Location2DError("xy_gps_seq", ppm)
    metrics["yaw_seq_error"] = AngleError("yaw_seq")
    metrics["yaw_gps_seq_error"] = AngleError("yaw_gps_seq")
    metrics = metrics.to(model.device)

    keys_save = ["xyr_max", "xy_max", "yaw_max", "xy_expectation", "xy_gps"]
    if viz:
        keys_save.append("log_probs")

    for chunk_index, key in tqdm(chunk_keys, disable=not progress):
        indices = chunk2idx[key]
        aligner = RigidAligner(track_priors=viz, num_rotations=num_rotations)
        aligner_gps = GPSAligner(track_priors=viz, num_rotations=num_rotations)
        batches = []
        preds = []
        for i in indices:
            data = dataset[i]
            data = model.transfer_batch_to_device(data, model.device, 0)
            pred = model(collate([data]))

            canvas = data["canvas"]
            data["xy_geo"] = xy = canvas.to_xy(data["xy"].double())
            data["yaw"] = yaw = data["roll_pitch_yaw"][-1].double()
            aligner.update(pred["log_probs"][0], canvas, xy, yaw)

            (uv_gps) = pred["xy_gps"] = data["xy_gps"][None]
            xy_gps = canvas.to_xy(uv_gps.double())
            aligner_gps.update(xy_gps, data["accuracy_gps"], canvas, xy, yaw)

            preds.append({k: pred[k][0] for k in keys_save})
            batches.append(data)
            if not viz:
                batches[-1].pop("image")
                batches[-1].pop("map")

        aligner.compute()
        aligner_gps.compute()
        xy_gt = torch.stack([b["xy_geo"] for b in batches])
        yaw_gt = torch.stack([b["yaw"] for b in batches])
        xy_seq, yaw_seq = aligner.transform(xy_gt, yaw_gt)
        xy_gps_seq, yaw_gps_seq = aligner_gps.transform(xy_gt, yaw_gt)
        results = []
        for i in range(len(indices)):
            preds[i]["xy_seq"] = batches[i]["canvas"].to_uv(xy_seq[i]).float()
            preds[i]["yaw_seq"] = yaw_seq[i].float()
            preds[i]["xy_gps_seq"] = batches[i]["canvas"].to_uv(xy_gps_seq[i]).float()
            preds[i]["yaw_gps_seq"] = yaw_gps_seq[i].float()
            results.append(metrics(preds[i], batches[i]))
        if viz:
            callback(chunk_index, model, batches, preds, results, aligner)
        del aligner, aligner_gps, preds, batches, results
    return metrics.cpu()
