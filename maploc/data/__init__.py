from .kitti.dataset import KittiDataModule
from .mapillary.dataset import MapillaryDataModule

modules = {"mapillary": MapillaryDataModule, "kitti": KittiDataModule}
