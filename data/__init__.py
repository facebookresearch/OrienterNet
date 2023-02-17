from .kitti import KittiDataModule
from .mapillary import MapillaryDataModule

modules = {"mapillary": MapillaryDataModule, "kitti": KittiDataModule}
