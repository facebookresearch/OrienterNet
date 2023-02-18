# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from uuid import uuid4

import hydra
import torch.distributed.launcher as pet
from hydra.core.hydra_config import HydraConfig
from hydra_plugins.flow_launcher_plugin.config import Mode
from omegaconf import DictConfig  # @manual

from .train import train


@hydra.main(config_path="conf", config_name="bev")
def main(cfg: DictConfig) -> None:
    # import pickle
    # print(pickle.dumps(train))
    # return
    if cfg.training.elastic:
        gpus = cfg.training.trainer.gpus
        lc = pet.LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=1 if gpus in (None, 0) else gpus,
            rdzv_backend="zeus",
            run_id=f"recon_init_{uuid4()}",
            max_restarts=0,
            start_method="spawn",
        )
        cfg_hydra = HydraConfig.get()
        is_flow = str(cfg_hydra.launcher.mode) == str(Mode.flow)
        job_id = cfg_hydra.job.id if is_flow else None
        pet.elastic_launch(lc, entrypoint=train)(cfg, job_id)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
