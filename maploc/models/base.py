# Copyright (c) Meta Platforms, Inc. and affiliates.

# Adapted from PixLoc, Paul-Edouard Sarlin, ETH Zurich
# https://github.com/cvg/pixloc
# Released under the Apache License 2.0

"""
Base class for trainable models.
"""

from abc import ABCMeta, abstractmethod
from copy import copy

import omegaconf
from omegaconf import OmegaConf
from torch import nn


class BaseModel(nn.Module, metaclass=ABCMeta):
    """
    What the child model is expect to declare:
        default_conf: dictionary of the default configuration of the model.
        It recursively updates the default_conf of all parent classes, and
        it is updated by the user-provided configuration passed to __init__.
        Configurations can be nested.

        required_data_keys: list of expected keys in the input data dictionary.

        strict_conf (optional): boolean. If false, BaseModel does not raise
        an error when the user provides an unknown configuration entry.

        _init(self, conf): initialization method, where conf is the final
        configuration object (also accessible with `self.conf`). Accessing
        unknown configuration entries will raise an error.

        _forward(self, data): method that returns a dictionary of batched
        prediction tensors based on a dictionary of batched input data tensors.

        loss(self, pred, data): method that returns a dictionary of losses,
        computed from model predictions and input data. Each loss is a batch
        of scalars, i.e. a torch.Tensor of shape (B,).
        The total loss to be optimized has the key `'total'`.

        metrics(self, pred, data): method that returns a dictionary of metrics,
        each as a batch of scalars.
    """

    base_default_conf = {
        "name": None,
        "trainable": True,  # if false: do not optimize this model parameters
        "freeze_batch_normalization": False,  # use test-time statistics
    }
    default_conf = {}
    required_data_keys = []
    strict_conf = True

    def __init__(self, conf):
        """Perform some logic and call the _init method of the child model."""
        super().__init__()
        default_conf = OmegaConf.merge(
            self.base_default_conf, OmegaConf.create(self.default_conf)
        )
        if self.strict_conf:
            OmegaConf.set_struct(default_conf, True)

        # fixme: backward compatibility
        if "pad" in conf and "pad" not in default_conf:  # backward compat.
            with omegaconf.read_write(conf):
                with omegaconf.open_dict(conf):
                    conf["interpolation"] = {"pad": conf.pop("pad")}

        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        self.conf = conf = OmegaConf.merge(default_conf, conf)
        OmegaConf.set_readonly(conf, True)
        OmegaConf.set_struct(conf, True)
        self.required_data_keys = copy(self.required_data_keys)
        self._init(conf)

        if not conf.trainable:
            for p in self.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)

        def freeze_bn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

        if self.conf.freeze_batch_normalization:
            self.apply(freeze_bn)

        return self

    def forward(self, data):
        """Check the data and call the _forward method of the child model."""

        def recursive_key_check(expected, given):
            for key in expected:
                assert key in given, f"Missing key {key} in data"
                if isinstance(expected, dict):
                    recursive_key_check(expected[key], given[key])

        recursive_key_check(self.required_data_keys, data)
        return self._forward(data)

    @abstractmethod
    def _init(self, conf):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def _forward(self, data):
        """To be implemented by the child class."""
        raise NotImplementedError

    def loss(self, pred, data):
        """To be implemented by the child class."""
        raise NotImplementedError

    def metrics(self):
        return {}  # no metrics
