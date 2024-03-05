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

    batch["roll_pitch_yaw"][..., -1] = adjust_yaw(batch["roll_pitch_yaw"][..., -1])
    pred["yaw_max"][..., -1] = adjust_yaw(pred["yaw_max"][..., -1])
    pred["yaw_expectation"][..., -1] = adjust_yaw(pred["yaw_expectation"])
    batch.update(
        {
            "ij_gps": swap_uv_ij(batch.get("uv_gps")),
            "ij": swap_uv_ij(batch["uv"]),  # gt,
            "ij_init": swap_uv_ij(batch["uv_init"]),
        }
    )

    # change ij_gt + yaw to Transform2D object map_T_query
    batch["map_T_query_gt"] = Transform2D.from_degrees(
        batch["roll_pitch_yaw"][..., -1:], batch["ij"]
    )

    pred.update(
        {
            "ij_max": swap_uv_ij(pred["uv_max"]),
            "ij_expectation": swap_uv_ij(pred["uv_expectation"]),
        }
    )

    pred["map_T_query_max"] = Transform2D.from_degrees(
        pred["yaw_max"][None], pred["ij_max"]
    )
    pred["map_T_query_expectation"] = Transform2D.from_degrees(
        pred["yaw_expectation"][None], pred["ij_expectation"]
    )

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
    if "log_probs_fused" in pred:
        pred["log_probs_fused"] = torch.rot90(
            pred["log_probs_fused"], -1, dims=(-3, -2)
        )
    if "scores_unmasked" in pred:
        pred["scores_unmasked"] = torch.rot90(
            pred["scores_unmasked"], -1, dims=(-3, -2)
        )

    return batch, pred


def prepare_for_plotting(batch, pred):
    """Temporary function to undo the changes made to the model output"""
    batch_new = deepcopy(batch)
    pred_new = deepcopy(pred)

    # Change ij indexing to uv for matplotlib
    batch_new.update(
        {
            "uv_gps": swap_uv_ij(batch_new.get("ij_gps")),
            "uv": swap_uv_ij(batch_new["ij"]),  # gt,
            "uv_init": swap_uv_ij(batch_new["ij_init"]),
        }
    )
    pred_new.update(
        {
            "uv_max": swap_uv_ij(pred_new["ij_max"]),
            "uv_expectation": swap_uv_ij(pred_new["ij_expectation"]),
            "uvr_max": swap_uv_ij(pred_new["ijr_max"]),
            "uvr_expectation": swap_uv_ij(pred_new["ijr_expectation"]),
        }
    )

    # Convert map's memory layout to spatial layout for viz
    batch_new.update(
        {k: torch.rot90(batch_new[k], 1, dims=(-2, -1)) for k in ["map", "map_mask"]}
    )

    pred_new["map"].update(
        {
            k: torch.rot90(pred_new["map"][k], 1, dims=(-2, -1))
            for k in ["map_features", "log_prior"]
        }
    )
    pred_new["bev"].update(
        {
            k: torch.rot90(pred_new["bev"][k], 1, dims=(-2, -1))
            for k in ["output", "confidence"]
        }
    )
    pred_new.update(
        {
            k: torch.rot90(pred_new[k], 1, dims=(-2, -1))
            for k in ["features_bev", "valid_bev"]
        }
    )
    pred_new.update(
        {k: torch.rot90(pred_new[k], 1, dims=(-3, -2)) for k in ["scores", "log_probs"]}
    )

    if "ij_fused" in pred:
        pred_new["uv_fused"] = swap_uv_ij(pred_new["ij_fused"])
    if "log_probs_fused" in pred:
        pred_new["log_probs_fused"] = torch.rot90(
            pred_new["log_probs_fused"], 1, dims=(-3, -2)
        )
    if "scores_unmasked" in pred:
        pred_new["scores_unmasked"] = torch.rot90(
            pred_new["scores_unmasked"], 1, dims=(-3, -2)
        )

    return batch_new, pred_new
