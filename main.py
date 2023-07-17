import os

import torch
import hydra
from omegaconf import DictConfig
from hydra.utils import call, instantiate
from snn_attack import SNN_ATTACK 


@hydra.main(config_path='./config', config_name='default', version_base=None)
def main(cfg: DictConfig):
    # print(cfg)
    device = cfg.parameters.general_setup.visible_device
    os.environ['CUDA_VISIBLE_DEVICES']=device
    print("using GPU device:", device)
    data, model = call(cfg.dataset_and_model.dataset_and_model)
    # paras = instantiate(cfg.parameters)
    # print(paras)
    attacker = SNN_ATTACK(cfg, model)
    output = attacker.run(data)
    
    
    




if __name__ == '__main__':
    main()