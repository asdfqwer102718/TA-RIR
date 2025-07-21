# ======================================================================
# 本代码从pickle中获取数据
# ======================================================================
import os.path
import pickle

from torch.utils.data import Dataset
from utils.graph import *
import torch.nn.functional as F


class ReverbDataset(Dataset):
    """MONO RIR"""

    def __init__(
            self,
            pickle_files,
            config,
            use_noise,
            isTest=False
    ):
        """
        Args
            rir_file : list of rir audio files
            source_files : list of speech files
            isTest: 是否是正在测试。如果是，__getitem__会输出房间名字
        """
        self.pickle_files = pickle_files
        self.config = config
        self.isTest = isTest
        self.use_noise = use_noise  # Add white noise to handle noisy environment
        # self.num_of_irs = num_of_irs  # 数据集中IR的数量

        self.rir_length = int(config.rir_duration * config.sr)  # 1 sec = 48000 samples
        self.input_signal_length = config.input_length  # 131070 samples

        with open(os.path.join(config.pickle_path, 'scene2points.pickle'), 'rb') as f:
            self.scene_2_points = pickle.load(f)  # 每个场景到bounding_box点坐标的映射 {str -> torch.Tensor}

    def __len__(self):
        return len(self.pickle_files)

    def __getitem__(self, idx):
        """
        获取数据
        :param idx:
        :return: d: {"rir": rir, "flipped_rir": flipped_rir, "source": source, "noise": noise, "snr_db": snr_db}
        graph: 
        room_info_vector: 房间顶点和源、接收器位置坐标拼成的一维向量

        """
        filepath = self.pickle_files[idx]

        with open(filepath, 'rb') as f:
            d = pickle.load(f)
        file_name, _ = os.path.splitext(os.path.basename(self.pickle_files[idx]))
        scene_name: str = file_name.split('_')[0]
        room_vertices = self.scene_2_points[scene_name]
        graph = get_graph(room_vertices, d['source_point'], d['receiver_point'])
        d['source_point'] = d['source_point'].unsqueeze(0)
        d['receiver_point'] = d['receiver_point'].unsqueeze(0)
        room_info_vector = torch.cat((room_vertices, d['source_point'], d['receiver_point']), dim=0)
        room_info_vector = room_info_vector.view(-1)

        current_length = room_info_vector.size(0)  # 获取批次大小和当前长度
        # 将房间信息向量改为128维，少了补0，多了截断
        # 如果长度不足128，进行零填充；如果超过128，进行截断
        if current_length < self.config.room_info_len_after_padding:
            # 计算需要填充多少个零
            padding_size = self.config.room_info_len_after_padding - current_length
            # 进行零填充
            room_info_vector = F.pad(room_info_vector, (0, padding_size), "constant", 0)
        else:
            # 截断到目标长度
            room_info_vector = room_info_vector[:self.config.room_info_len_after_padding]
        room_info_vector = room_info_vector.float()

        # 获取scene信息
        if self.isTest:
            return d, graph, room_info_vector, filepath
        else:
            return d, graph, room_info_vector
