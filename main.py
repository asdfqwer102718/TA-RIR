import os
import argparse
import random

import numpy.random
import torch.distributed as dist
import torch
from torch_geometric.loader import DataLoader

from data.process_data import load_dataset
from trainer import Trainer
from dataloader import ReverbDataset
from model import TARIR_main
from utils.utils import load_config


def main(args):
    # 加载配置文件
    config = load_config(args.config_path)

    # 这里输出和使用的模型有关的一些附加说明
    if global_rank == 0:
        print(config.instruction)
        print()

    if global_rank == 0:
        print(config)

    # 读取的是pickle文件，存有IR和混响语音以及其他信息
    train_info_list, valid_info_list, test_info_list = load_dataset(config.dataset.params.pickle_path)
    train_dataset = ReverbDataset(train_info_list,
        # num_of_IRs_train,
        config.dataset.params, use_noise=True)
    valid_dataset = ReverbDataset(valid_info_list,
        # num_of_IRs_val,
        config.dataset.params, use_noise=True)
    # load model
    model = TARIR_main(config.model.params)

    # run trainer
    trainer = Trainer(model, local_rank, global_rank, world_size, train_dataset,
        valid_dataset, config.train.params, config.eval.params, args)

    trainer.train()


if __name__ == "__main__":
    cpu_only = False
    use_ddp = True

    seed = 3407
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default='cpu')
    parser.add_argument("--save_name", type=str, default="m")
    parser.add_argument("--resume_step", type=int, default=0)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default="config.yaml")
    args = parser.parse_args()


    if not cpu_only and use_ddp:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        global_rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        # 配置DDP（DDP对模型的处理写在初始化里了）
        dist.init_process_group(backend='nccl', world_size=world_size)  # world_size是gpu数量
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0
        global_rank = 0
        world_size = 0
        device = 'cpu'
        if not cpu_only:
            torch.cuda.set_device(int(args.device))
            args.device = torch.device(f'cuda:{args.device}')


    main(args)
