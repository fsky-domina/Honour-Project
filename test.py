import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from dataset import EMGDataset
from model import EMGTransformer
import os
#python ./Transformer3/test.py
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用 GPU ID 为 1 的设备

def test_model(model_path, test_input_dir, test_label_dir, device, save_path):
    # 加载模型
    model = EMGTransformer().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载测试集
    test_dataset = EMGDataset(
        test_input_dir, test_label_dir,
        mode='test'
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    
    # 确保存储路径存在
    os.makedirs(save_path, exist_ok=True)
    
    with torch.no_grad():
        for i, (emg, pressure) in enumerate(test_loader):
            emg = emg.to(device)
            outputs = model(emg).cpu().numpy()
            outputs = outputs.squeeze(2)
            
            # 保存为 .mat 文件
            save_file_path = os.path.join(save_path, f"{i}.mat")
            save_file_path = os.path.join(save_path, f"{i}.mat")
            savemat(save_file_path, {
                'predicted': outputs,
                'input': emg.cpu().numpy().squeeze(2),  # 假设 emg 的形状为 [batch_size, T, 1]
                'label': pressure.numpy().squeeze(2)  # 假设 pressure 的形状为 [batch_size, T, 1]
            })
            
            print(f"Saved results for {i} to {save_file_path}")
            
            if i >= 300:  # 仅处理前两个样本
                break

if __name__ == "__main__":
    device = torch.device("cuda")
    test_model(
        model_path='D:\model3/best_model.pth',
        test_input_dir='./Pdata/train/input',
        test_label_dir='./Pdata/train/label',
        device=device,
        save_path='./Transformer3/results'
    )