# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from collections import defaultdict

import numpy as np
import torch


def chunk_sequence_v2(
    data,
    names,
    indices,
    max_length=100,
    min_length=1,
    max_inter_dist=None,
    max_total_dist=None,
):
    indices = sorted(indices, key=lambda i: data.get("capture_time", names)[i].tolist())
    centers = torch.stack([data["t_c2w"][i][:2] for i in indices]).numpy()
    dists = np.linalg.norm(np.diff(centers, axis=0), axis=-1)
    chunks = [[indices[0]]]
    dist_total = 0
    for dist, idx in zip(dists, indices[1:]):
        dist_total += dist
        if (
            (max_inter_dist is not None and dist > max_inter_dist)
            or (max_total_dist is not None and dist_total > max_total_dist)
            or len(chunks[-1]) >= max_length
        ):
            chunks.append([])
            dist_total = 0
        chunks[-1].append(idx)
    chunks = list(filter(lambda c: len(c) >= min_length, chunks))
    chunks = sorted(chunks, key=len, reverse=True)
    return chunks


def chunk_sequence(names, views, max_inter_dist=10, max_length=100, min_length=5):
    names = sorted(names, key=lambda n: views[n]["capture_time"])
    centers = np.array([views[n]["t_c2w"][:2] for n in names])
    dists = np.linalg.norm(np.diff(centers, axis=0), axis=-1)
    chunks = [[names[0]]]
    for dist, name in zip(dists, names[1:]):
        if (max_inter_dist is not None and dist > max_inter_dist) or len(
            chunks[-1]
        ) >= max_length:
            chunks.append([])
        chunks[-1].append(name)
    chunks = list(filter(lambda c: len(c) >= min_length, chunks))
    chunks = sorted(chunks, key=len, reverse=True)
    return chunks


def extract_sequences(dumps, all_names, scenes=None, split_cam=True, **kw):
    seq_data = defaultdict(lambda: defaultdict(list))
    for sc, seq, name in all_names:
        seq_data[sc][seq].append(name)
    seq_data = dict(seq_data)

    if scenes is None:
        scenes = list(seq_data.keys())

    chunks = defaultdict(dict)
    for scene in scenes:
        seq2names = seq_data[scene]
        for seq_key in seq2names:
            views = dumps[scene][seq_key]["views"]
            cam2names = defaultdict(list)
            if split_cam:
                for name in seq2names[seq_key]:
                    cam2names[name.rsplit("_", 1)[-1].rsplit("-", 1)[-1]].append(name)
            else:
                cam2names["single"] = seq2names[seq_key]
            seq_chunks = []
            for names in cam2names.values():
                seq_chunks.extend(chunk_sequence(names, views, **kw))
            chunks[scene][seq_key] = seq_chunks
    return dict(chunks)


def unpack_batches(batches):
    images = [b["image"].permute(1, 2, 0) for b in batches]
    canvas = [b["canvas"] for b in batches]
    rasters = [b["map"] for b in batches]
    yaws = torch.stack([b["roll_pitch_yaw"][-1] for b in batches])
    uv_gt = torch.stack([b["xy"] for b in batches])
    xy_gt = torch.stack(
        [canv.to_xy(uv.cpu().double()) for uv, canv in zip(uv_gt, canvas)]
    )
    ret = [images, canvas, rasters, yaws, uv_gt, xy_gt.to(uv_gt)]
    if "xy_gps" in batches[0]:
        xy_gps = torch.stack(
            [c.to_xy(b["xy_gps"].cpu().double()) for b, c in zip(batches, canvas)]
        )
        ret.append(xy_gps.to(uv_gt))
    return ret
