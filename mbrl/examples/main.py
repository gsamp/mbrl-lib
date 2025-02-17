# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# GEORGIA BEGIN
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
# GEORGIA END

import hydra
import numpy as np
import omegaconf
from omegaconf import OmegaConf
import torch

import mbrl.algorithms.mbpo as mbpo
import mbrl.algorithms.mbpo_backwards as mbpo_backwards
import mbrl.algorithms.pets as pets
import mbrl.algorithms.planet as planet
import mbrl.util.mujoco as mujoco_util

# GEORGIA BEGIN
import wandb
# GEORGIA END

@hydra.main(config_path="conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    # GEORGIA BEGIN
    wandb.init(project="cs229-project",
               config=OmegaConf.to_container(cfg),
               settings=wandb.Settings(start_method="fork"))
    # GEORGIA END

    env, term_fn, reward_fn = mujoco_util.make_env(cfg)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.algorithm.name == "pets":
        return pets.train(env, term_fn, reward_fn, cfg)
    if cfg.algorithm.name == "mbpo":
        test_env, *_ = mujoco_util.make_env(cfg)
        return mbpo.train(env, test_env, term_fn, cfg)
    if cfg.algorithm.name == "mbpo_backwards":
        test_env, *_ = mujoco_util.make_env(cfg)
        return mbpo_backwards.train(env, test_env, term_fn, cfg)
    if cfg.algorithm.name == "planet":
        return planet.train(env, cfg)


if __name__ == "__main__":
    run()
