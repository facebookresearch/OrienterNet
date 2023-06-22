# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
from omegaconf import OmegaConf

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
