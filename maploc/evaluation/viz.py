# Copyright (c) Meta Platforms, Inc. and affiliates.

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..osm.viz import Colormap, plot_nodes
from ..utils.io import write_torch_image
from ..utils.viz_2d import features_to_RGB, plot_images, save_plot
from ..utils.viz_localization import (
    add_circle_inset,
    likelihood_overlay,
    plot_dense_rotations,
    plot_pose,
)


def plot_example_single(
    idx,
    model,
    pred,
    data,
    results,
    plot_bev=True,
    out_dir=None,
    fig_for_paper=False,
    show_gps=False,
    show_fused=False,
    show_dir_error=False,
    show_masked_prob=False,
    refactored_input=False,
):

    # map_T_query (or m_T_q) is the Transform (trans and rot) of query in pixel space.
    # map_t_query (or m_t_q) is only the translation.
    # if refactored_input is True, then the translation is in ij format, else in uv.
    # m_r_q (or yaw) is the rotation only. If refactored_input, then this is
    # east-counterclockwise, else north-clockwise.

    if refactored_input:
        # The refactored input has rot and trans combined
        # into a Transform2D object: map_T_query
        scene, name, rasters, map_T_query_gt = (
            data[k] for k in ("scene", "name", "map", "map_T_query_gt")
        )

        m_t_q_gt = map_T_query_gt.t.squeeze(0)  # ij_gt
        yaw_gt = map_T_query_gt.angle.squeeze(0)  # m_r_q_gt

        m_t_gps = data.get("ij_gps")
        if show_fused and "ij_fused" in pred:
            m_t_q_pred = pred["ij_fused"]
            yaw_p = pred.get("yaw_fused")
        else:
            m_T_q_pred = pred["map_T_query_max"]
            m_t_q_pred = m_T_q_pred.t.squeeze(0)  # ij_p
            yaw_p = m_T_q_pred.angle.squeeze(0)  # m_r_q_pred

    else:
        # Temporary support for original formats
        scene, name, rasters, m_t_q_gt = (
            data[k] for k in ("scene", "name", "map", "uv")
        )
        yaw_gt = data["roll_pitch_yaw"][-1].numpy()  # m_r_q_gt
        m_t_gps = data.get("uv_gps")  # uv_gps
        if show_fused and "uv_fused" in pred:
            m_t_q_pred = pred["uv_fused"]
            yaw_p = pred.get("yaw_fused")
        else:
            m_t_q_pred = pred["uv_max"]  # uv_p
            yaw_p = pred.get("yaw_max")  # m_r_q_pred

    image = data["image"].permute(1, 2, 0)
    if "valid" in data:
        image = image.masked_fill(~data["valid"].unsqueeze(-1), 0.3)

    # if input is not refactored, lp_ijt and lp_ij are actually lp_uvt and lp_uv
    lp_ijt = lp_ij = pred["log_probs"]
    if show_fused and "log_probs_fused" in pred:
        lp_ijt = lp_ij = pred["log_probs_fused"]
    elif not show_masked_prob and "scores_unmasked" in pred:
        lp_ijt = lp_ij = pred["scores_unmasked"]
    has_rotation = lp_ijt.ndim == 3
    if has_rotation:
        lp_ij = lp_ijt.max(-1).values
    if lp_ij.min() > -np.inf:
        lp_ij = lp_ij.clip(min=np.percentile(lp_ij, 1))
    prob = lp_ij.exp()

    feats_map = pred["map"]["map_features"][0]
    (feats_map_rgb,) = features_to_RGB(feats_map.numpy())

    text1 = rf'$\Delta xy$: {results["xy_max_error"]:.1f}m'
    if has_rotation:
        text1 += rf', $\Delta\theta$: {results["yaw_max_error"]:.1f}°'
    if show_fused and "xy_fused_error" in results:
        text1 += rf', $\Delta xy_{{fused}}$: {results["xy_fused_error"]:.1f}m'
        text1 += rf', $\Delta\theta_{{fused}}$: {results["yaw_fused_error"]:.1f}°'
    if show_dir_error and "directional_error" in results:
        err_lat, err_lon = results["directional_error"]
        text1 += rf",  $\Delta$lateral/longitundinal={err_lat:.1f}m/{err_lon:.1f}m"
    if "xy_gps_error" in results:
        text1 += rf',  $\Delta xy_{{GPS}}$: {results["xy_gps_error"]:.1f}m'

    map_viz = Colormap.apply(rasters)
    overlay = likelihood_overlay(prob.numpy(), map_viz.mean(-1, keepdims=True))
    if refactored_input:
        map_viz, overlay, feats_map_rgb = [
            np.swapaxes(x, 0, 1) for x in (map_viz, overlay, feats_map_rgb)
        ]
    plot_images(
        [image, map_viz, overlay, feats_map_rgb],
        titles=[text1, "map", "likelihood", "neural map"],
        origins=["upper", "lower", "lower", "lower"] if refactored_input else None,
        dpi=75,
        cmaps="jet",
    )
    fig = plt.gcf()
    axes = fig.axes
    axes[1].images[0].set_interpolation("none")
    axes[2].images[0].set_interpolation("none")
    Colormap.add_colorbar()
    plot_nodes(1, rasters[2], refactored=refactored_input)

    if show_gps and m_t_gps is not None:
        plot_pose([1], m_t_gps, c="blue", refactored=refactored_input)
    plot_pose([1], m_t_q_gt, yaw_gt, c="red", refactored=refactored_input)
    plot_pose([1], m_t_q_pred, yaw_p, c="k", refactored=refactored_input)
    plot_dense_rotations(2, lp_ijt.exp(), refactored=refactored_input)
    inset_center = m_t_q_pred if results["xy_max_error"] < 5 else m_t_q_gt

    # Doesn't work for refactored axes conventions
    # axins = add_circle_inset(axes[2], inset_center, refactored=True)
    # axins.scatter(*ij_gt, lw=1, c="red", ec="k", s=50, zorder=15)

    axes[0].text(
        0.003,
        0.003,
        f"{scene}/{name}",
        transform=axes[0].transAxes,
        fontsize=3,
        va="bottom",
        ha="left",
        color="w",
    )
    plt.show()
    if out_dir is not None:
        name_ = name.replace("/", "_")
        p = str(out_dir / f"{scene}_{name_}_{{}}.pdf")
        save_plot(p.format("pred"))
        plt.close()

        if fig_for_paper:
            # !cp ../datasets/MGL/{scene}/images/{name}.jpg {out_dir}/{scene}_{name}.jpg
            plot_images([map_viz])
            plt.gca().images[0].set_interpolation("none")
            plot_nodes(0, rasters[2])
            plot_pose([0], m_t_q_gt, yaw_gt, c="red")
            plot_pose([0], m_t_q_pred, yaw_p, c="k")
            save_plot(p.format("map"))
            plt.close()
            plot_images([lp_ij], cmaps="jet")
            plot_dense_rotations(0, lp_ijt.exp())
            save_plot(p.format("loglikelihood"), dpi=100)
            plt.close()
            plot_images([overlay])
            plt.gca().images[0].set_interpolation("none")
            axins = add_circle_inset(plt.gca(), inset_center)
            axins.scatter(*m_t_q_gt, lw=1, c="red", ec="k", s=50)
            save_plot(p.format("likelihood"))
            plt.close()
            write_torch_image(
                p.format("neuralmap").replace("pdf", "jpg"), feats_map_rgb
            )
            write_torch_image(p.format("image").replace("pdf", "jpg"), image.numpy())

    if not plot_bev:
        return

    feats_q = pred["features_bev"]
    mask_bev = pred["valid_bev"]
    prior = None
    if "log_prior" in pred["map"]:
        prior = pred["map"]["log_prior"][0].sigmoid()
    if "bev" in pred and "confidence" in pred["bev"]:
        conf_q = pred["bev"]["confidence"]
    else:
        conf_q = torch.norm(feats_q, dim=0)
    conf_q = conf_q.masked_fill(~mask_bev, np.nan)
    (feats_q_rgb,) = features_to_RGB(feats_q.numpy(), masks=[mask_bev.numpy()])
    # feats_map_rgb, feats_q_rgb, = features_to_RGB(
    #     feats_map.numpy(), feats_q.numpy(), masks=[None, mask_bev])
    norm_map = torch.norm(feats_map, dim=0)
    origins = None
    if refactored_input:
        conf_q, feats_q_rgb, norm_map = [
            np.swapaxes(x, 0, 1) for x in [conf_q, feats_q_rgb, norm_map]
        ]
        if prior is not None:
            prior = np.swapaxes(prior, 0, 1)
        origins = ["lower", "lower", "lower"] + ([] if prior is None else ["lower"])
    plot_images(
        [conf_q, feats_q_rgb, norm_map] + ([] if prior is None else [prior]),
        titles=["BEV confidence", "BEV features", "map norm"]
        + ([] if prior is None else ["map prior"]),
        origins=origins,
        dpi=50,
        cmaps="jet",
    )
    plt.show()

    if out_dir is not None:
        save_plot(p.format("bev"))
        plt.close()


def plot_example_sequential(
    idx,
    model,
    pred,
    data,
    results,
    plot_bev=True,
    out_dir=None,
    fig_for_paper=False,
    show_gps=False,
    show_fused=False,
    show_dir_error=False,
    show_masked_prob=False,
):
    return
