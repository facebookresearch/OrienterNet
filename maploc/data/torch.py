# Copyright (c) Meta Platforms, Inc. and affiliates.

import collections
import os

import torch
from torch.utils.data import get_worker_info
from torch.utils.data._utils.collate import (
    default_collate_err_msg_format,
    np_str_obj_array_pattern,
)
from lightning_fabric.utilities.seed import pl_worker_init_function
from lightning_utilities.core.apply_func import apply_to_collection
from lightning_fabric.utilities.apply_func import move_data_to_device


def collate(batch):
    """Difference with PyTorch default_collate: it can stack other tensor-like objects.
    Adapted from PixLoc, Paul-Edouard Sarlin, ETH Zurich
    https://github.com/cvg/pixloc
    Released under the Apache License 2.0
    """
    if not isinstance(batch, list):  # no batching
        return batch
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel, device=elem.device)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]
    else:
        # try to stack anyway in case the object implements stacking.
        try:
            return torch.stack(batch, 0)
        except TypeError as e:
            if "expected Tensor as element" in str(e):
                return batch
            else:
                raise e


def set_num_threads(nt):
    """Force numpy and other libraries to use a limited number of threads."""
    try:
        import mkl
    except ImportError:
        pass
    else:
        mkl.set_num_threads(nt)
    torch.set_num_threads(1)
    os.environ["IPC_ENABLE"] = "1"
    for o in [
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]:
        os.environ[o] = str(nt)


def worker_init_fn(i):
    info = get_worker_info()
    pl_worker_init_function(info.id)
    num_threads = info.dataset.cfg.get("num_threads")
    if num_threads is not None:
        set_num_threads(num_threads)


def unbatch_to_device(data, device="cpu"):
    data = move_data_to_device(data, device)
    data = apply_to_collection(data, torch.Tensor, lambda x: x.squeeze(0))
    data = apply_to_collection(
        data, list, lambda x: x[0] if len(x) == 1 and isinstance(x[0], str) else x
    )
    return data
