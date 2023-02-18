import numpy as np

bev_override = {"model": {"num_rotations": 256, "apply_map_prior": False}}
experiments = {
    "BEV_MLY": (
        "bev1-osm2-mly13-n100_res101-vgg19_depth-bins33-nonorm-1to9_bs9-resize512_d8-nrot64-normvalid-prior-rep",
        bev_override,
    ),
    "BEV_MLY+KITTI": (
        "bev1-osm2-kitti-5m_res101-vgg19_depth-bins33-nonorm-1to9_bs9-f256_d8-nrot64-normvalid-prior-rep_ft-lr5",
        bev_override,
    ),
    "BEV_KITTI": (
        "bev1-osm2-kitti_res101-vgg19_depth-bins33-nonorm-1to9_bs9-f256_d8-nrot64-normvalid-prior-rep_ckpt",
        bev_override,
    ),
    "Retrieval_MLY": ("basic-osm2-mly13-n100_res101-vgg19_bs9-resize512_d256", {}),
}


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
