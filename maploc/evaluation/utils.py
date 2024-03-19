# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch
from omegaconf import OmegaConf

from ..utils.io import write_json
from ..utils.wrappers import Transform2D


def compute_recall(errors):
    num_elements = len(errors)
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(num_elements) + 1) / num_elements
    recall = np.r_[0, recall]
    errors = np.r_[0, errors]
    return errors, recall


def compute_auc(errors, recall, thresholds):
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t, side="right")
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        auc = np.trapz(r, x=e) / t
        aucs.append(auc * 100)
    return aucs


def write_dump(output_dir, experiment, cfg, results, metrics):
    dump = {
        "experiment": experiment,
        "cfg": OmegaConf.to_container(cfg),
        "results": results,
        "errors": {},
    }
    for k, m in metrics.items():
        if hasattr(m, "get_errors"):
            dump["errors"][k] = m.get_errors().numpy()
    write_json(output_dir / "log.json", dump)


def swap_uv_ij(uv, im_height=255):
    if uv is not None:
        ij = uv.clone()
        ij[..., 1] = im_height - uv[..., 1]
        return ij


def refactor_pred(pred):
    """
    Changes yaw convention
    Changes uv indices (origin top left) to ij indices (origin bottom left)
    Replaces uv_max, uvr_max, yaw_max -> Transform2D object
             uv_expectation, uvr_expectation, yaw_expectation -> Transform2D
    Changes map memory layout to allow indexing like map[i, j]
    """

    # changing yaw angle from north-clockwise to east-counterclockwise
    yaw_max = 90 - pred["yaw_max"][..., -1:][None]
    yaw_expectation = 90 - pred["yaw_expectation"][None]
    if "yaw_fused" in pred:
        pred["yaw_fused"] = 90 - pred["yaw_fused"][None]

    ij_max = swap_uv_ij(pred["uv_max"])
    ij_expectation = swap_uv_ij(pred["uv_expectation"])
    if "uv_fused" in pred:
        pred["ij_fused"] = swap_uv_ij(pred["uv_fused"])
        del pred["uv_fused"]

    pred["map_T_cam_max"] = Transform2D.from_degrees(yaw_max, ij_max)
    pred["map_T_cam_expectation"] = Transform2D.from_degrees(
        yaw_expectation, ij_expectation
    )
    del pred["yaw_max"], pred["uv_max"], pred["yaw_expectation"], pred["uv_expectation"]
    del pred["uvr_max"], pred["uvr_expectation"]

    pred["map"].update(
        {
            k: torch.rot90(torch.stack(pred["map"][k]), -1, dims=(-2, -1))
            for k in ["map_features", "log_prior"]
        }
    )
    pred["bev"].update(
        {
            k: torch.rot90(pred["bev"][k], -1, dims=(-2, -1))
            for k in ["output", "confidence"]
        }
    )
    pred.update(
        {
            k: torch.rot90(pred[k], -1, dims=(-2, -1))
            for k in ["features_bev", "valid_bev"]
        }
    )
    pred.update(
        {k: torch.rot90(pred[k], -1, dims=(-3, -2)) for k in ["scores", "log_probs"]}
    )

    if "log_probs_fused" in pred:
        pred["log_probs_fused"] = torch.rot90(
            pred["log_probs_fused"], -1, dims=(-3, -2)
        )
    if "scores_unmasked" in pred:
        pred["scores_unmasked"] = torch.rot90(
            pred["scores_unmasked"], -1, dims=(-3, -2)
        )

    return pred


def refactor_batch(batch):

    # Ideally, this happens in the dataloader.
    batch.update(
        {k: torch.rot90(batch[k], -1, dims=(-2, -1)) for k in ["map", "map_mask"]}
    )

    return batch
