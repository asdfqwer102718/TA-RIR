import os
import argparse
import random
from datetime import datetime

import numpy.random
import torch
from torch_geometric.data import DataLoader

from data.process_data import load_speech_dataset
from dataloader import ReverbDataset
from model import TARIR_main
from utils.utils import load_config
from tester import Tester


def load_dataset(dataset_path):
    """
    dataset_path 要求下面有三个文件夹：train, val, test
    返回值是三个列表，先后表示训练集、验证集、测试集，每个列表里都是绝对路径
    :param dataset_path:
    :return:
    """
    d = {}
    subset = 'test'
    dir_d = os.path.join(dataset_path, subset)
    d[subset] = os.listdir(dir_d)
    d[subset] = [os.path.join(dir_d, i) for i in d[subset]]

    return d['test']


def main(args):
    # 加载配置文件

    if torch.cuda.is_available() and use_cuda:
        torch.cuda.set_device(int(device))
        args.device = torch.device(f"cuda:{device}")
    else:
        args.device = "cpu"

    # 读取的是pickle文件，存有IR和混响语音以及其他信息
    test_info_list = load_dataset(config.dataset.params.pickle_path)

    # load dataset
    num_of_IRs_test = int(config.dataset.params.num_of_scene_train * 0.1) * config.dataset.params.num_IRs_per_scene
    test_info_list = test_info_list[:num_samples_to_select]
    testDataset = ReverbDataset(test_info_list,
        config.dataset.params, use_noise=True, isTest=True)

    test_dataloader = DataLoader(
        testDataset,
        batch_size=config.train.params.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.train.params.num_workers,
    )

    print("Number of RIR data", num_of_IRs_test)
    print("Number of speech data", len(test_info_list))
    print("Number of batches", len(test_dataloader))

    model = TARIR_main(config.model.params)

    tester = Tester(model, test_dataloader, checkpoint_path, config.train.params, args)
    tester.test()


if __name__ == "__main__":
    use_cuda = False
    device = 0
    checkpoint_dir = ''    # put your checkpoint path here
    config_path = "config_ours.yaml"
    test_result_save_dir = f"{datetime.now().strftime('%y%m%d-%H%M%S')}"
    # 配置要选择的样本数量
    num_samples_to_select = 32768

    print(f"test_result_save_dir: {test_result_save_dir}")
    save_name_prefix = 'ours'      # test_dir下面保存的测试路径的前缀

    test_epoches = [137]

    # 模型参数设置
    seed = 3407
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default='cpu')
    parser.add_argument("--resume_step", type=int, default=0)
    parser.add_argument("--checkpoint_path", type=str, default=None)

    config = load_config(config_path)
    print(config)
    for epoch in test_epoches:
        args = parser.parse_args()
        checkpoint_path = os.path.join(checkpoint_dir, f'epoch-{epoch}.pt')
        setattr(args, 'save_name', save_name_prefix)
        setattr(args, 'save_name_suffix', f'-{epoch}')
        setattr(args, 'test_result_save_dir', test_result_save_dir)
        setattr(args, 'epoch_model', epoch)
        print()
        print(f"test the model of epoch {epoch}")
        main(args)
