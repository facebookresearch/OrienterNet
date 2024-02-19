# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from pathlib import Path

import pytorch_lightning  # noqa: F401

formatter = logging.Formatter(
    fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

logger = logging.getLogger("maploc")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

pl_logger = logging.getLogger("pytorch_lightning")
if len(pl_logger.handlers):
    pl_logger.handlers[0].setFormatter(formatter)

repo_dir = Path(__file__).parent.parent
EXPERIMENTS_PATH = repo_dir / "experiments/"
DATASETS_PATH = repo_dir / "datasets/"
