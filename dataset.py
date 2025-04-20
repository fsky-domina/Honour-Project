import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat  # 用于加载.mat文件
from sklearn.preprocessing import StandardScaler

class EMGDataset(Dataset):
    def __init__(self, input_dir, label_dir, emg_scaler=None, pressure_scaler=None, mode='train'):
        """
        初始化数据集
        :param input_dir: 输入信号目录（包含.mat文件）
        :param label_dir: 标签信号目录（包含.mat文件）
        :param emg_scaler: 预定义的肌电信号标准化器
        :param pressure_scaler: 预定义的压力信号标准化器
        :param mode: 数据集模式（train/test）
        """
        # 获取匹配的文件列表
        self.input_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.mat')])
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.mat')])
        
        # 验证文件匹配
        assert len(self.input_files) == len(self.label_files), "输入和标签文件数量不一致"
        for i, l in zip(self.input_files, self.label_files):
            assert i.split('_')[0] == l.split('_')[0], f"文件不匹配: {i} vs {l}"

        self.input_dir = input_dir
        self.label_dir = label_dir

        # 数据标准化处理
        if mode == 'train':
            self.emg_scaler = StandardScaler()
            self.pressure_scaler = StandardScaler()
        else:
            # 测试模式使用预训练的scaler
            #assert emg_scaler is not None and pressure_scaler is not None
            self.emg_scaler = StandardScaler()
            self.pressure_scaler = StandardScaler()
            #self.emg_scaler = emg_scaler
            #self.pressure_scaler = pressure_scaler

    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        # 加载数据
        input_data = loadmat(os.path.join(self.input_dir, self.input_files[idx]))
        label_data = loadmat(os.path.join(self.label_dir, self.label_files[idx]))
        
        # 检查变量是否存在
        if 'subdata_2' not in input_data or 'subdata_1' not in label_data:
            raise KeyError(f"File {self.input_files[idx]} or {self.label_files[idx]} is missing required variables.")
        
        input_data = input_data['subdata_2'].T
        label_data = label_data['subdata_1'].T
        
        # 检查数据是否为空
        if input_data.size == 0 or label_data.size == 0:
            raise ValueError(f"File {self.input_files[idx]} or {self.label_files[idx]} contains empty data.")
        
        # 检查数据形状
        if input_data.ndim != 2 or input_data.shape[1] != 1:
            raise ValueError(f"Input data in file {self.input_files[idx]} has incorrect shape: {input_data.shape}")
        if label_data.ndim != 2 or label_data.shape[1] != 1:
            raise ValueError(f"Label data in file {self.label_files[idx]} has incorrect shape: {label_data.shape}")
        
        # 检查数据是否包含非法值
        if np.isnan(input_data).any() or np.isinf(input_data).any():
            raise ValueError(f"Input data in file {self.input_files[idx]} contains NaN or Inf values.")
        if np.isnan(label_data).any() or np.isinf(label_data).any():
            raise ValueError(f"Label data in file {self.label_files[idx]} contains NaN or Inf values.")
        
        # 转换为torch.Tensor
        input_data = torch.FloatTensor(input_data)  # [T, 1]
        label_data = torch.FloatTensor(label_data)  # [T, 1]
        
        return input_data, label_data

    def _scale_data(self, data, scaler):
        """
        对单个样本进行归一化
        :param data: 单个样本的数据，形状为 [T, 1]
        :param scaler: 标准化器
        :return: 归一化后的数据
        """
        original_shape = data.shape
        data = data.reshape(-1, 1)  # 转换为 [T * 1, 1] 以便标准化
        if isinstance(scaler, StandardScaler):
            data = scaler.fit_transform(data) if not hasattr(scaler, 'mean_') else scaler.transform(data)
        return data.reshape(original_shape)  # 恢复原始形状 [T, 1]

"""
def collate_fn(batch):
    #动态填充数据
    #:param batch: 一个批次的数据，包含多个 (input, label) 对
    #:return: 填充后的 inputs 和 labels
    inputs = [item[0] for item in batch]  # List of [T, 1] tensors
    labels = [item[1] for item in batch]  # List of [T, 1] tensors

    # 动态填充
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)  # [B, T_max, 1]
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)  # [B, T_max, 1]

    return inputs, labels
"""

def get_dataloaders(train_input_dir, train_label_dir, test_input_dir, test_label_dir, batch_size=32):
    """
    获取训练和测试DataLoader
    :param train_input_dir: 训练输入信号目录
    :param train_label_dir: 训练标签信号目录
    :param test_input_dir: 测试输入信号目录
    :param test_label_dir: 测试标签信号目录
    :param batch_size: 批次大小
    :return: train_loader, test_loader, train_dataset
    """
    # 加载训练集
    train_dataset = EMGDataset(train_input_dir, train_label_dir, mode='train')
    
    # 加载测试集（使用训练集的scaler）
    test_dataset = EMGDataset(
        test_input_dir, test_label_dir,
        mode='test'
    )
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader, train_dataset