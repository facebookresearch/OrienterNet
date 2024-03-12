# Copyright (c) Meta Platforms, Inc. and affiliates.

from copy import deepcopy

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


def swap_uv_ij(uv, im_height=256):
    if uv is not None:
        ij = uv.clone()
        ij[..., 1] = im_height - uv[..., 1]
        return ij


def refactor_model_output(batch_, pred_):
    """
    Changes yaw convention
    Changes uv indices (origin top left) to ij indices (origin bottom left)
    Replaces uv_max, uvr_max, yaw_max -> Transform2D object
             uv_expectation, uvr_expectation, yaw_expectation -> Transform2D
    Changes map memory layout to allow indexing like map[i, j]
    """
    batch = deepcopy(batch_)
    pred = deepcopy(pred_)

    # changing yaw angle from north-clockwise to east-counterclockwise
    def adjust_yaw(angle):
        return 90 - angle

    # change ij_gt + yaw to Transform2D object map_T_query
    yaw_gt = adjust_yaw(batch["roll_pitch_yaw"][..., -1])
    ij_gt = swap_uv_ij(batch["uv"])
    batch["map_T_query_gt"] = Transform2D.from_degrees(yaw_gt[None], ij_gt)
    batch.update(
        {
            "ij_gps": swap_uv_ij(batch.get("uv_gps")),
            "ij_init": swap_uv_ij(batch["uv_init"]),
        }
    )
    del batch["uv"], batch["uv_gps"], batch["uv_init"], batch["roll_pitch_yaw"]

    yaw_max = adjust_yaw(pred["yaw_max"][..., -1:])[None]
    ij_max = swap_uv_ij(pred["uv_max"])
    yaw_expectation = adjust_yaw(pred["yaw_expectation"])[None]
    ij_expectation = swap_uv_ij(pred["uv_expectation"])
    pred["map_T_query_max"] = Transform2D.from_degrees(yaw_max, ij_max)
    pred["map_T_query_expectation"] = Transform2D.from_degrees(
        yaw_expectation, ij_expectation
    )
    del pred["yaw_max"], pred["uv_max"], pred["yaw_expectation"], pred["uv_expectation"]
    del pred["uvr_max"], pred["uvr_expectation"]

    batch.update(
        {k: torch.rot90(batch[k], -1, dims=(-2, -1)) for k in ["map", "map_mask"]}
    )

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

    if "uv_fused" in pred:
        pred["ij_fused"] = swap_uv_ij(pred["uv_fused"])
        del pred["uv_fused"]
    if "yaw_fused" in pred:
        pred["yaw_fused"] = adjust_yaw(pred["yaw_fused"])[None]

    if "log_probs_fused" in pred:
        pred["log_probs_fused"] = torch.rot90(
            pred["log_probs_fused"], -1, dims=(-3, -2)
        )
    if "scores_unmasked" in pred:
        pred["scores_unmasked"] = torch.rot90(
            pred["scores_unmasked"], -1, dims=(-3, -2)
        )

    return batch, pred
