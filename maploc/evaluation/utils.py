# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
from omegaconf import OmegaConf
from copy import deepcopy
import torch

from ..utils.io import write_json


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
            ij[...,1] = im_height - uv[...,1]
            return ij

def refactor_model_output(batch, pred):

    batch_new = deepcopy(batch)
    pred_new = deepcopy(pred)

    batch_new.update({
        'ij_gps': swap_uv_ij(batch_new.get('uv_gps')),
        'ij': swap_uv_ij(batch_new['uv']), # gt,
        'ij_init': swap_uv_ij(batch_new['uv_init'])
    })
    # TODO: change ij_gt + roll_pitch_yaw to Transform3D object map_T_query

    pred_new.update({
        'ij_max': swap_uv_ij(pred_new['uv_max']),
        'ij_expectation': swap_uv_ij(pred_new['uv_expectation']),
        'ijr_max': swap_uv_ij(pred_new['uvr_max']),
        'ijr_expectation': swap_uv_ij(pred_new['uvr_expectation'])
    })
    # TODO: convert uvr_max and uvr_expectations to Transform2D objects

    # Change map memory layout to allow indexing like map[i, j]
    batch_new.update({k: torch.rot90(batch_new[k], -1, dims=(-2, -1))
                      for k in ['map', 'map_mask']})

    pred_new["map"].update({k: torch.rot90(torch.stack(pred_new["map"][k]), -1, dims=(-2, -1))
                     for k in ['map_features', 'log_prior']})
    pred_new["bev"].update({k: torch.rot90(pred_new["bev"][k], -1, dims=(-2, -1))
                     for k in ['output', 'confidence']})
    pred_new.update({k: torch.rot90(pred_new[k], -1, dims=(-2, -1))
                     for k in ['features_bev', 'valid_bev']})
    pred_new.update({k: torch.rot90(pred_new[k], -1, dims=(-3, -2))
                     for k in ['scores', 'log_probs']
    })

    if 'uv_fused' in pred:
        pred_new['ij_fused'] = swap_uv_ij(pred_new['uv_fused'])
    if 'log_probs_fused' in pred:
        pred_new['log_probs_fused'] = torch.rot90(pred_new['log_probs_fused'], -1, dims=(-3, -2))
    if 'scores_unmasked' in pred:
        pred_new['scores_unmasked'] = torch.rot90(pred_new['scores_unmasked'], -1, dims=(-3,-2))

    return batch_new, pred_new


def prepare_for_plotting(batch, pred):

    batch_new = deepcopy(batch)
    pred_new = deepcopy(pred)

    # Change ij indexing to uv for matplotlib
    batch_new.update({
        'uv_gps': swap_uv_ij(batch_new.get('ij_gps')),
        'uv': swap_uv_ij(batch_new['ij']), # gt,
        'uv_init': swap_uv_ij(batch_new['ij_init'])
    })
    pred_new.update({
        'uv_max': swap_uv_ij(pred_new['ij_max']),
        'uv_expectation': swap_uv_ij(pred_new['ij_expectation']),
        'uvr_max': swap_uv_ij(pred_new['ijr_max']),
        'uvr_expectation': swap_uv_ij(pred_new['ijr_expectation'])
    })

    # Convert map's memory layout to spatial layout for viz
    batch_new.update({k: torch.rot90(batch_new[k], 1, dims=(-2, -1))
                   for k in ['map', 'map_mask']})

    pred_new["map"].update({k: torch.rot90(pred_new["map"][k], 1, dims=(-2, -1))
                     for k in ['map_features', 'log_prior']})
    pred_new["bev"].update({k: torch.rot90(pred_new["bev"][k], 1, dims=(-2, -1))
                     for k in ['output', 'confidence']})
    pred_new.update({k: torch.rot90(pred_new[k], 1, dims=(-2, -1))
                     for k in ['features_bev', 'valid_bev']})
    pred_new.update({k: torch.rot90(pred_new[k], 1, dims=(-3, -2))
                     for k in ['scores', 'log_probs']})

    if 'ij_fused' in pred:
        pred_new['uv_fused'] = swap_uv_ij(pred_new['ij_fused'])
    if 'log_probs_fused' in pred:
        pred_new['log_probs_fused'] = torch.rot90(pred_new['log_probs_fused'], 1, dims=(-3, -2))
    if 'scores_unmasked' in pred:
        pred_new['scores_unmasked'] = torch.rot90(pred_new['scores_unmasked'], 1, dims=(-3,-2))

    return batch_new, pred_new