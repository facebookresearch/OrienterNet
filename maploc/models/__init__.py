# Copyright (c) Meta Platforms, Inc. and affiliates.

# Adapted from PixLoc, Paul-Edouard Sarlin, ETH Zurich
# https://github.com/cvg/pixloc
# Released under the Apache License 2.0

import inspect

from .base import BaseModel


def get_class(mod_name, base_path, BaseClass):
    """Get the class object which inherits from BaseClass and is defined in
    the module named mod_name, child of base_path.
    """
    mod_path = "{}.{}".format(base_path, mod_name)
    mod = __import__(mod_path, fromlist=[""])
    classes = inspect.getmembers(mod, inspect.isclass)
    # Filter classes defined in the module
    classes = [c for c in classes if c[1].__module__ == mod_path]
    # Filter classes inherited from BaseModel
    classes = [c for c in classes if issubclass(c[1], BaseClass)]
    assert len(classes) == 1, classes
    return classes[0][1]


def get_model(name):
    if name == "localizer":
        name = "localizer_basic"
    elif name == "rotation_localizer":
        name = "localizer_basic_rotation"
    elif name == "bev_localizer":
        name = "localizer_bev_plane"
    return get_class(name, __name__, BaseModel)
