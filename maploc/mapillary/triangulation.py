import logging
import os.path as osp
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from mapillary.vision.sfm.mapillary_sfm.dataset import (
    ClusterDataSet,
    ClusterDataSetBase,
    ClusterVRSDataset,
)
from mapillary.vision.sfm.mapillary_sfm.retriangulate import (
    cluster_common,
    cluster_reconstruct,
    PointsCuller,
    tracks_statistics_log,
)
from opensfm import (
    features_processing,
    io,
    matching,
    pairs_selection,
    pymap,
    reconstruction as orec,
    types,
    undistort,
)
from opensfm.dataset import UndistortedDataSet
from opensfm.dataset_base import DataSetBase

logger = logging.getLogger(__name__)


def compute_retrieval_pairs(
    data: DataSetBase, reconstruction: types.Reconstruction
) -> List[Tuple[str, str]]:
    shots = list(reconstruction.shots)
    exifs = {im: data.load_exif(im) for im in shots}
    config_override = {}
    config_override["symmetric_matching"] = False
    vlad_neighbors = data.config["cluster_intra_matching_count_regular"]
    vlad_preempt = data.config["cluster_intra_matching_preempt_regular"]
    config_override["matching_vlad_neighbors"] = vlad_neighbors
    config_override["matching_vlad_gps_neighbors"] = vlad_preempt
    config_override["matching_vlad_other_cameras"] = False
    pairs_list, _ = pairs_selection.match_candidates_from_metadata(
        shots, shots, exifs, data, config_override
    )
    return pairs_list


def retriangulate_generic(
    data: ClusterDataSetBase,
    reconstruction: types.Reconstruction,
    do_guided_matching: bool = True,
) -> Optional[Tuple[types.Reconstruction, pymap.TracksManager]]:
    shots = list(reconstruction.shots.keys())

    # enforce some flags
    backup_config = data.config.copy()
    data.config["triangulation_type"] = "ROBUST"

    # extract features
    start = time.time()
    features_processing.run_features_processing(data, shots, True)
    logger.info(f"Re-extracted features in {time.time()-start} seconds")

    pairs_to_match = set(compute_retrieval_pairs(data, reconstruction))

    # match features or load them
    exifs = {image: data.load_exif(image) for image in shots}
    start = time.time()
    match_config_override = {}
    poses = None
    if do_guided_matching:
        match_config_override = {
            "guided_filter_only": True,
        }
        poses = {shot.id: shot.pose for shot in reconstruction.shots.values()}
    pairs_matched = matching.match_images_with_pairs(
        data,
        match_config_override,
        exifs,
        list(pairs_to_match),
        poses=poses,
    )
    success_pairs = sum([1 for x in pairs_matched.values() if len(x) > 0])
    if len(pairs_to_match) > 0:
        logger.info(
            f"Classic-Matched {len(pairs_to_match)} pairs (success = {success_pairs} ({float(success_pairs)/len(pairs_to_match)*100}%))in {time.time()-start} seconds"
        )
        cluster_common.update_and_save_matches(data, shots, pairs_matched, False)
    else:
        logger.warning(
            "No pairs matched. Not running triangulation and returning. Try using with RetriangulateStages.TRIANGULATE"
        )
        return None

    # construct tracks and perform initial triangulation
    start = time.time()
    tracks_manager = cluster_reconstruct.construct_tracks_manager(
        data, shots, pairs_matched, data.config["guided_min_length_initial"]
    )
    _, retriangulated = orec.reconstruct_from_prior(
        data, tracks_manager, reconstruction
    )
    logger.info(
        f"Triangulated {len(retriangulated.points)} tracks in {time.time()-start} seconds"
    )
    tracks_statistics_log("After raw triangulation", retriangulated)

    # remove obs with high error
    start = time.time()
    culler = PointsCuller(retriangulated, data.config, data)
    culled = culler.remove_high_error()
    logger.info(f"Cleaned points in {time.time()-start} seconds")

    # remove points behind
    start = time.time()
    culled = culler.cull_points_behind()
    logger.info(f"Culled {culled} points behind in {time.time()-start} seconds")

    # remove points with low parallax
    start = time.time()
    culled = culler.cull_indecent_parallax_points()
    logger.info(
        f"Culled {culled} points with indecent parallax in {time.time()-start} seconds"
    )
    tracks_statistics_log("After culling high-error/behing/parralax", retriangulated)

    # remove points from short tracks
    start = time.time()
    culled = culler.cull_short_points()
    logger.info(f"Culled {culled} points too short in {time.time()-start} seconds")
    tracks_statistics_log("After culling short", retriangulated)

    # restore config
    data.config.update(backup_config)

    return retriangulated, tracks_manager


def retriangulate_reconstruction(
    data: ClusterDataSet, rec: types.Reconstruction, do_guided_matching: bool = True
):
    # Backup config and log levels
    backup_config = data.config.copy()
    loggers = [features_processing.logger, pairs_selection.logger, matching.logger]
    levels = [log.level for log in loggers]
    for log in loggers:
        log.setLevel(logging.WARNING)

    data.config["processes"] = data.config["read_processes"] = 30
    data.config["features_bake_segmentation"] = False
    data.config["matching_use_segmentation"] = False
    data.config["feature_min_frames"] = 6000
    data.config["hahog_upright"] = True
    if do_guided_matching:
        data.config["lowes_ratio"] = 0.95
    data.config["cluster_intra_matching_count_regular"] = 10
    data.config["cluster_intra_matching_preempt_regular"] = 60
    data.config["guided_matching_threshold"] = 0.002  # 0.006 by default
    data.config["guided_min_length_initial"] = 2
    data.config["guided_min_length_final"] = 2

    # mostly unused now
    extraction_size_thresh = 1.2
    extract_sizes = []
    for shot in rec.shots.values():
        is_pano = shot.camera.is_panorama(shot.camera.projection_type)
        size = data.config[
            "feature_process_size_panorama" if is_pano else "feature_process_size"
        ]
        extract_sizes.append(size)
    extraction_size_thresh_norm = extraction_size_thresh / np.median(extract_sizes)
    data.config["guided_extend_threshold"] = extraction_size_thresh_norm
    data.config["guided_extend_tracks"] = False

    logger.info("Retriangulating a reconstruction with %d shots.", len(rec.shots))
    retriangulated, tracks_new = retriangulate_generic(
        data,
        rec,
        do_guided_matching=do_guided_matching,
    )

    # Restore config and log levels
    data.config.update(backup_config)
    for log, level in zip(loggers, levels):
        log.setLevel(level)
    return retriangulated, tracks_new


class MinimalClusterDataSet(ClusterDataSet):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.exifs = {}
        self.camera_models = None
        self.load_config()
        self.load_exif_from_clusters()

    def load_exif(self, image: str) -> Dict[str, Any]:
        return self.exifs[image]

    def load_exif_from_clusters(self):
        for cluster_id in self.cluster_ids:
            (cluster,) = self.load_cluster_reconstruction_aligned(cluster_id)
            for image_key, shot in cluster.shots.items():
                self.exifs[image_key] = ClusterVRSDataset._convert_shot_to_exif(
                    None, cluster.reference, shot, shot.camera
                )
        self.image_list = list(self.exifs)


MinimalClusterDataSet.load_camera_models = ClusterVRSDataset.load_camera_models
MinimalClusterDataSet._init_camera_models = ClusterVRSDataset._init_camera_models
MinimalClusterDataSet._add_camera_models_with_old_ids = (
    ClusterVRSDataset._add_camera_models_with_old_ids
)


def undistort_reconstruction(
    data: ClusterDataSet, cluster_id: str, rec: types.Reconstruction, tracks=None
):
    undistorted_dir = Path(data.cluster_file_path(cluster_id, "undistorted"))
    undistorted_dir.mkdir(exist_ok=True, parents=True)
    undistorted_data_path = undistorted_dir / "cluster" / cluster_id
    udata = UndistortedDataSet(data, undistorted_data_path, io_handler=data.io_handler)
    udata._undistorted_image_path = lambda: osp.join(undistorted_dir, "images")

    data.config["depthmap_resolution"] = 1024
    data.config["processes"] = 30
    undistort.undistort_reconstruction_with_images(
        tracks, rec, data, udata, skip_images=False
    )
    (urec,) = udata.load_undistorted_reconstruction()

    with udata.io_handler.open_wt(undistorted_data_path / "aligned.json") as fout:
        io.json_dump(io.reconstructions_to_json([urec]), fout, minify=True)

    udata_cluster = MinimalClusterDataSet(undistorted_dir)
    (urec,) = udata_cluster.load_cluster_reconstruction_aligned(cluster_id)

    return udata_cluster, urec


def transfer_tracks(rec_undistorted, tracks_undistorted, rec_original):
    rec = deepcopy(rec_original)
    rec.set_points(rec_undistorted.points)
    tracks = pymap.TracksManager()
    for shot in rec.shots.values():
        observations = {}
        for subshot in rec_undistorted.shots.values():
            if shot.id not in subshot.id:
                continue
            for lm in subshot.get_valid_landmarks():
                observations[lm.id] = subshot.get_landmark_observation(lm)
        for track_id, obs in observations.items():
            tracks.add_observation(shot.id, track_id, obs)
    rec.add_correspondences_from_tracks_manager(tracks)
    return rec, tracks


def get_triangulated_reconstruction(
    data: ClusterDataSet,
    chunk_key: str,
    triangulated_dirname: str = "cluster_retriangulated",
) -> types.Reconstruction:
    def new_clusters_path_fn():
        return osp.join(data.data_path, triangulated_dirname)

    old_clusters_path_fn = data._clusters_path
    data._clusters_path = new_clusters_path_fn
    if data.cluster_reconstruction_aligned_exists(chunk_key):
        (rec,) = data.load_cluster_reconstruction_aligned(chunk_key)
        # bypass the cache, force a reload
        tracks = data.load_cluster_tracks_manager_cached.__wrapped__(data, chunk_key)
        rec.add_correspondences_from_tracks_manager(tracks)
        data._clusters_path = old_clusters_path_fn
    else:
        data._clusters_path = old_clusters_path_fn
        (rec_orig,) = data.load_cluster_reconstruction_aligned(chunk_key)
        logger.info("Undistorting the reconstruction %s", chunk_key)
        udata, urec_orig = undistort_reconstruction(data, chunk_key, rec_orig)
        logger.info("Retriangulating the reconstruction %s", chunk_key)
        urec, utracks = retriangulate_reconstruction(udata, urec_orig)
        udata.save_cluster_reconstruction_aligned(chunk_key, [urec])
        rec, tracks = transfer_tracks(urec, utracks, rec_orig)
        # rec, tracks = retriangulate_reconstruction(data, rec_orig)

        data._clusters_path = new_clusters_path_fn
        data.save_cluster_reconstruction_initial(chunk_key, [rec], tracks)
        data.save_cluster_reconstruction_aligned(chunk_key, [rec])
        data._clusters_path = old_clusters_path_fn
    return rec
