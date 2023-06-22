# Copyright (c) Meta Platforms, Inc. and affiliates.

from collections import Counter, defaultdict
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from .parser import (
    filter_area,
    filter_node,
    filter_way,
    match_to_group,
    parse_area,
    parse_node,
    parse_way,
    Patterns,
)
from .reader import OSMData


def recover_hierarchy(counter: Counter) -> Dict:
    """Recover a two-level hierarchy from the flat group labels."""
    groups = defaultdict(dict)
    for k, v in sorted(counter.items(), key=lambda x: -x[1]):
        if ":" in k:
            prefix, group = k.split(":")
            if prefix in groups and isinstance(groups[prefix], int):
                groups[prefix] = {}
                groups[prefix][prefix] = groups[prefix]
                groups[prefix] = {}
            groups[prefix][group] = v
        else:
            groups[k] = v
    return dict(groups)


def bar_autolabel(rects, fontsize):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        width = rect.get_width()
        plt.gca().annotate(
            f"{width}",
            xy=(width, rect.get_y() + rect.get_height() / 2),
            xytext=(3, 0),  # 3 points vertical offset
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=fontsize,
        )


def plot_histogram(counts, fontsize, dpi):
    fig, ax = plt.subplots(dpi=dpi, figsize=(8, 20))

    labels = []
    for k, v in counts.items():
        if isinstance(v, dict):
            labels += list(v.keys())
            v = list(v.values())
        else:
            labels.append(k)
            v = [v]
        bars = plt.barh(
            len(labels) + -len(v) + np.arange(len(v)), v, height=0.9, label=k
        )
        bar_autolabel(bars, fontsize)

    ax.set_yticklabels(labels, fontsize=fontsize)
    ax.axes.xaxis.set_ticklabels([])
    ax.xaxis.tick_top()
    ax.invert_yaxis()
    plt.yticks(np.arange(len(labels)))
    plt.xscale("log")
    plt.legend(ncol=len(counts), loc="upper center")


def count_elements(elems: Dict[int, str], filter_fn, parse_fn) -> Dict:
    """Count the number of elements in each group."""
    counts = Counter()
    for elem in filter(filter_fn, elems.values()):
        group = parse_fn(elem.tags)
        if group is None:
            continue
        counts[group] += 1
    counts = recover_hierarchy(counts)
    return counts


def plot_osm_histograms(osm: OSMData, fontsize=8, dpi=150):
    counts = count_elements(osm.nodes, filter_node, parse_node)
    plot_histogram(counts, fontsize, dpi)
    plt.title("nodes")

    counts = count_elements(osm.ways, filter_way, parse_way)
    plot_histogram(counts, fontsize, dpi)
    plt.title("ways")

    counts = count_elements(osm.ways, filter_area, parse_area)
    plot_histogram(counts, fontsize, dpi)
    plt.title("areas")


def plot_sankey_hierarchy(osm: OSMData):
    triplets = []
    for node in filter(filter_node, osm.nodes.values()):
        label = parse_node(node.tags)
        if label is None:
            continue
        group = match_to_group(label, Patterns.nodes)
        if group is None:
            group = match_to_group(label, Patterns.ways)
        if group is None:
            group = "null"
        if ":" in label:
            key, tag = label.split(":")
            if tag == "yes":
                tag = key
        else:
            key = tag = label
        triplets.append((key, tag, group))
    keys, tags, groups = list(zip(*triplets))
    counts_key_tag = Counter(zip(keys, tags))
    counts_key_tag_group = Counter(triplets)

    key2tags = defaultdict(set)
    for k, t in zip(keys, tags):
        key2tags[k].add(t)
    key2tags = {k: sorted(t) for k, t in key2tags.items()}
    keytag2group = dict(zip(zip(keys, tags), groups))
    key_names = sorted(set(keys))
    tag_names = [(k, t) for k in key_names for t in key2tags[k]]

    group_names = []
    for k in key_names:
        for t in key2tags[k]:
            g = keytag2group[k, t]
            if g not in group_names and g != "null":
                group_names.append(g)
    group_names += ["null"]

    key2idx = dict(zip(key_names, range(len(key_names))))
    tag2idx = {kt: i + len(key2idx) for i, kt in enumerate(tag_names)}
    group2idx = {n: i + len(key2idx) + len(tag2idx) for i, n in enumerate(group_names)}

    key_counts = Counter(keys)
    key_text = [f"{k} {key_counts[k]}" for k in key_names]
    tag_counts = Counter(list(zip(keys, tags)))
    tag_text = [f"{t} {tag_counts[k, t]}" for k, t in tag_names]
    group_counts = Counter(groups)
    group_text = [f"{k} {group_counts[k]}" for k in group_names]

    fig = go.Figure(
        data=[
            go.Sankey(
                orientation="h",
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=key_text + tag_text + group_text,
                    x=[0] * len(key_names)
                    + [1] * len(tag_names)
                    + [2] * len(group_names),
                    color="blue",
                ),
                arrangement="fixed",
                link=dict(
                    source=[key2idx[k] for k, _ in counts_key_tag]
                    + [tag2idx[k, t] for k, t, _ in counts_key_tag_group],
                    target=[tag2idx[k, t] for k, t in counts_key_tag]
                    + [group2idx[g] for _, _, g in counts_key_tag_group],
                    value=list(counts_key_tag.values())
                    + list(counts_key_tag_group.values()),
                ),
            )
        ]
    )
    fig.update_layout(autosize=False, width=800, height=2000, font_size=10)
    fig.show()
    return fig
