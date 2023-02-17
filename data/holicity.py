# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import functools
import math
import shutil
from collections import defaultdict
from multiprocessing import Pool

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from ..utils.geometry import to_homogeneous
from ..utils.io import read_image
from ..utils.wrappers import Pose

############
# GPS-world projection
# Adapted from Yichao Zhou https://github.com/zhou13/holicity

Offset = np.array(
    [
        [
            [51.48653658, -0.15787351],
            [51.49552138, -0.15750779],
            [51.50450284, -0.15714251],
            [51.5134942, -0.15678146],
            [51.522482, -0.15641665],
            [51.53147821, -0.15603593],
        ],
        [
            [51.48629708, -0.14348133],
            [51.4952862, -0.14311014],
            [51.50427476, -0.14273894],
            [51.51326159, -0.14237039],
            [51.52225695, -0.14200299],
            [51.53124625, -0.14162871],
        ],
        [
            [51.48606092, -0.12908871],
            [51.49505511, -0.12871558],
            [51.50405668, -0.12834281],
            [51.51303504, -0.1279733],
            [51.52202077, -0.12760482],
            [51.53100586, -0.12723739],
        ],
        [
            [51.485832, -0.11469519],
            [51.49482364, -0.11431929],
            [51.50381857, -0.11394519],
            [51.51280529, -0.11357625],
            [51.5217936, -0.11320131],
            [51.53077853, -0.11282795],
        ],
        [
            [51.48560681, -0.10029999],
            [51.49459821, -0.09992166],
            [51.50358628, -0.09954092],
            [51.51256733, -0.09916756],
            [51.52156189, -0.09879787],
            [51.53055204, -0.09841437],
        ],
        [
            [51.48537856, -0.08590049],
            [51.49436934, -0.08552391],
            [51.50335272, -0.08514402],
            [51.51233467, -0.08477007],
            [51.52132308, -0.08439054],
            [51.53031843, -0.08400393],
        ],
        [
            [51.48514379, -0.07149651],
            [51.49412855, -0.07111692],
            [51.50311466, -0.0707391],
            [51.51210374, -0.07036401],
            [51.52108874, -0.06998574],
            [51.53007984, -0.06960159],
        ],
        [
            [51.48490644, -0.05709119],
            [51.49389141, -0.05671215],
            [51.50287896, -0.05633446],
            [51.51186803, -0.05595725],
            [51.5208552, -0.05557937],
            [51.5298418, -0.05520037],
        ],
    ]
)


def model2gps(X):
    # [-40, -21] ~ [21, 20]
    x, y = X
    x0, y0 = math.floor(x / 1000), math.floor(y / 1000)
    xp, yp = x0 + 4, y0 + 3
    Oo = Offset[xp, yp]
    Ox = Offset[xp + 1, yp]
    Oy = Offset[xp, yp + 1]
    Oxy = Offset[xp + 1, yp + 1]
    dx, dy = x / 1000 - x0, y / 1000 - y0
    return (
        Oo * (1 - dx) * (1 - dy)
        + Ox * dx * (1 - dy)
        + Oy * (1 - dx) * dy
        + Oxy * dx * dy
    )


def gps2model(Y0):
    def risk(X):
        Y = model2gps(X)
        return Y - Y0

    try:
        result = least_squares(risk, np.r_[0, 0], gtol=1e-12, verbose=0)
        return result.x
    except IndexError:
        return None


############


class Projection:
    def project(self, geo):
        if len(geo.shape) == 1:
            return gps2model(geo)
        elif len(geo.shape) == 2:
            xy = []
            for g in geo:
                xy_i = gps2model(g)
                if xy_i is None:
                    continue
                xy.append(xy_i)
            return np.stack(xy, 0)
        else:
            raise ValueError(geo.shape)

    def unproject(self, xy):
        if len(xy.shape) == 1:
            return model2gps(xy)
        elif len(xy.shape) == 2:
            return np.stack([model2gps(xy_i) for xy_i in xy], 0)
        else:
            raise ValueError(xy.shape)


def build_perspective_4x4mat(loc, yaw, pitch):
    """Computes 4x4 world-to-camera transformation matrix"""
    yaw = -yaw * np.pi / 180 + np.pi / 2
    pitch = pitch * np.pi / 180
    return lookat(
        loc,
        [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)],
        [0, 0, 1],
    )


def lookat(position, forward, up):
    """Computes 4x4 transformation matrix to put camera looking at look point."""
    c = np.asarray(position).astype(float)
    w = -np.asarray(forward).astype(float)
    u = np.cross(up, w)
    v = np.cross(w, u)
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    w /= np.linalg.norm(w)
    return np.r_[u, u.dot(-c), v, v.dot(-c), w, w.dot(-c), 0, 0, 0, 1].reshape(4, 4)


def proj_pano(d, image_size):
    # assume that Z is up, Y is forward
    d = d.T  # Nx3 to 3xN
    pitch = np.arctan2(d[2], np.linalg.norm(d[:2], axis=0))
    yaw = np.arctan2(d[0], d[1])
    x = (yaw + np.pi) / (np.pi * 2)
    y = 1 - (pitch + np.pi / 2) / np.pi
    xy_norm = np.stack([x, y], -1)
    xy = xy_norm * np.array(image_size)
    return xy, xy_norm


def render_perspective(pano, cam_data, yaw, pitch, image_size):
    pano_yaw = np.deg2rad(cam_data["pano_yaw"])
    tilt_yaw = cam_data["tilt_yaw"]
    tilt_pitch = np.deg2rad(cam_data["tilt_pitch"])

    # perspective -> CV: (x,y,z) -> (x,-y,-z)
    R_cam2cv = np.diag([1, -1, -1])
    R_w2cam = R_cam2cv @ build_perspective_4x4mat(np.zeros(3), yaw, pitch)[:3, :3]
    axis = np.cross([np.cos(tilt_yaw), np.sin(tilt_yaw), 0], [0, 0, 1])
    # Z is up, Y is forward, X is right
    R_w2pano = (
        Rotation.from_rotvec(-axis * tilt_pitch) * Rotation.from_euler("z", -pano_yaw)
    ).as_matrix()
    R_w2pano = (
        Rotation.from_euler("z", -pano_yaw) * Rotation.from_rotvec(-axis * tilt_pitch)
    ).as_matrix()
    R_cam2pano = R_w2pano @ R_w2cam.T

    grid = np.stack(np.indices(image_size[::-1]), -1)[..., ::-1]
    w, h = image_size
    fx, fy, cx, cy = w / 2, h / 2, w / 2, h / 2.0  # 90 FOV
    p2d_norm = (grid.reshape(-1, 2) - np.array([cx, cy])) / np.array([fx, fy])

    bearing_cam = to_homogeneous(p2d_norm)
    bearing_cam = bearing_cam / np.linalg.norm(bearing_cam, axis=-1, keepdims=True)
    bearing_pano = bearing_cam @ R_cam2pano.T
    # Z is up, Y is forward, X is right

    pano_size = pano.shape[:2][::-1]
    p2d_pano, p2d_pano_norm = proj_pano(bearing_pano, pano_size)
    p2d_pano_grid = p2d_pano.reshape(grid.shape).astype(np.float32)

    image = cv2.remap(
        pano,
        *p2d_pano_grid.transpose(2, 0, 1),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )
    return image


def render_pano(idx, pano2prefixes, root, camera_dir, image_dir):
    pano_name, prefixes = pano2prefixes[idx]

    pano = read_image("./data/panorama/" + pano_name + ".jpg")
    prefixes_new = []
    for prefix in prefixes:
        with np.load(root / f"camr/{prefix}_camr.npz") as cam_data:
            cam_data = dict(cam_data)

        pitch = np.random.randint(-15, 16)
        prefix_new = "_".join(prefix.split("_")[:-1] + [f"{pitch:+03d}"])
        cam_data["pitch"] = pitch
        cam_data["R"] = build_perspective_4x4mat(
            cam_data["loc"], cam_data["yaw"], pitch
        )
        render = render_perspective(pano, cam_data, cam_data["yaw"], pitch, (512, 512))

        (camera_dir / prefix_new).parent.mkdir(exist_ok=True, parents=True)
        (image_dir / prefix_new).parent.mkdir(exist_ok=True, parents=True)
        np.savez(camera_dir / f"{prefix_new}_camr.npz", **cam_data)
        cv2.imwrite(str(image_dir / f"{prefix_new}_imag.jpg"), render[:, :, ::-1])
        prefixes_new.append(prefix_new)
    print("Done pano", idx)
    return prefixes_new


def run_rendering(root, out_dir):
    # root = Path('buckets/llp_datasets/holicity/')
    # out_dir = Path('./data/holicity_renders')

    with open(root / "split/filelist.txt") as fid:
        prefixes = [line.strip("\n") for line in fid]

    camera_dir = out_dir / "camr"
    image_dir = out_dir / "image"
    if camera_dir.exists():
        shutil.rmtree(camera_dir)
    if image_dir.exists():
        shutil.rmtree(image_dir)

    pano2prefixes = defaultdict(list)
    for prefix in prefixes:
        pano = "_".join(prefix.split("_")[:-3])
        pano2prefixes[pano].append(prefix)
    pano2prefixes = dict(pano2prefixes)
    panos = list(pano2prefixes)
    print(f"Doing {len(panos)} panoramas")

    fn = functools.partial(
        render_pano,
        pano2prefixes=list(pano2prefixes.items()),
        root=root,
        camera_dir=camera_dir,
        image_dir=image_dir,
    )
    with Pool(10) as p:
        prefixes_new = p.map(fn, range(len(panos)))

    prefixes_new = [p for pp in prefixes_new for p in pp]
    with open(out_dir / "split/filelist.txt", "w") as fp:
        fp.write("\n".join(prefixes_new))


def prepare_views(root, proj):
    with open(root / "split/filelist.txt") as fid:
        names = [line.strip("\n") for line in fid]
    camera = {
        "id": 1,
        "model": "PINHOLE",
        "width": 512,
        "height": 512,
        "params": [256, 256, 256, 256],
    }
    dump = {
        "cameras": {camera["id"]: camera},
        "views": {},
    }
    proj_orig = Projection()
    for name in tqdm(names):
        with np.load(root / f"camr/{name}_camr.npz") as cam_data:
            cam_data = dict(cam_data)
        T_w2c = Pose.from_4x4mat(np.diag([1, -1, -1, 1]) @ cam_data["R"])
        T_c2w = T_w2c.inv()
        gps = proj_orig.unproject(T_c2w.t[:2])
        xy = proj.project(gps)
        T_c2w.t = np.r_[xy, T_c2w.t[-1]]
        dump["views"][name + "_imag.jpg"] = {
            "camera_id": camera["id"],
            "R_c2w": T_c2w.R,
            "t_c2w": T_c2w.t,
            "gps": gps,
        }
    return dump
