# Copyright (c) Meta Platforms, Inc. and affiliates.

import math
from typing import Optional

import torch


def checkpointed(cls, do=True):
    """Adapted from the DISK implementation of Michał Tyszkiewicz."""
    assert issubclass(cls, torch.nn.Module)

    class Checkpointed(cls):
        def forward(self, *args, **kwargs):
            super_fwd = super(Checkpointed, self).forward
            if any((torch.is_tensor(a) and a.requires_grad) for a in args):
                return torch.utils.checkpoint.checkpoint(super_fwd, *args, **kwargs)
            else:
                return super_fwd(*args, **kwargs)

    return Checkpointed if do else cls


class GlobalPooling(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind == "mean":
            self.fn = torch.nn.Sequential(
                torch.nn.Flatten(2), torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten()
            )
        elif kind == "max":
            self.fn = torch.nn.Sequential(
                torch.nn.Flatten(2), torch.nn.AdaptiveMaxPool1d(1), torch.nn.Flatten()
            )
        else:
            raise ValueError(f"Unknown pooling type {kind}.")

    def forward(self, x):
        return self.fn(x)


@torch.jit.script
def make_grid(
    w: float,
    h: float,
    step_x: float = 1.0,
    step_y: float = 1.0,
    orig_x: float = 0,
    orig_y: float = 0,
    y_up: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    x, y = torch.meshgrid(
        [
            torch.arange(orig_x, w + orig_x, step_x, device=device),
            torch.arange(orig_y, h + orig_y, step_y, device=device),
        ],
        indexing="xy",
    )
    if y_up:
        y = y.flip(-2)
    grid = torch.stack((x, y), -1)
    return grid


@torch.jit.script
def rotmat2d(angle: torch.Tensor) -> torch.Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    R = torch.stack([c, -s, s, c], -1).reshape(angle.shape + (2, 2))
    return R


@torch.jit.script
def rotmat2d_grad(angle: torch.Tensor) -> torch.Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    R = torch.stack([-s, -c, c, -s], -1).reshape(angle.shape + (2, 2))
    return R


def deg2rad(x):
    return x * math.pi / 180


def rad2deg(x):
    return x * 180 / math.pi
