import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from dataset import get_dataloaders
from model import EMGTransformer
import matplotlib.pyplot as plt
#python ./Transformer3/train.py
# 训练配置
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用 GPU ID 为 0 的设备
print("当前使用的设备:", torch.cuda.current_device())
print("Available GPUs:", torch.cuda.device_count())
device = torch.device("cuda")
batch_size = 1
lr = 8e-5
weight_decay = 1e-5
patience = 10000

# 获取数据
train_loader, test_loader, train_dataset = get_dataloaders(
    train_input_dir="./PData/train/input",
    train_label_dir="./PData/train/label",
    test_input_dir="./PData/debug/input",
    test_label_dir="./PData/debug/label",
    batch_size=batch_size
)

# 初始化模型
model = EMGTransformer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
writer = SummaryWriter('logs')

# 检查是否从保存的模型恢复训练
model_save_path = './Transformer3/models/best_model0.pth'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
start_epoch = 1
best_loss = float('inf')
train_losses = []
val_losses = []

if os.path.exists(model_save_path):
    print("找到保存的模型，正在恢复训练...")
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['loss']
    train_dataset.emg_scaler = checkpoint['emg_scaler']
    train_dataset.pressure_scaler = checkpoint['pressure_scaler']
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    print(f"从 Epoch {start_epoch} 开始继续训练。")

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (emg, pressure) in enumerate(loader):
        emg = emg.to(device)
        pressure = pressure.to(device)
        
        optimizer.zero_grad()
        outputs = model(emg)
        loss = criterion(outputs, pressure)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} Loss: {loss.item():.4f}")
    
    return total_loss / len(loader.dataset)

def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for emg, pressure in loader:
            emg = emg.to(device)
            pressure = pressure.to(device)
            outputs = model(emg)
            total_loss += criterion(outputs, pressure).item() * emg.size(1)
    
    return total_loss / len(loader.dataset)

# 训练循环
counter = 0
model_save_path_new = './Transformer3/models/best_model.pth'
for epoch in range(start_epoch, 10000):
    train_loss = train_epoch(model, train_loader, optimizer, device)
    train_losses.append(train_loss)
    
    # 记录到TensorBoard
    # writer.add_scalar('Loss/train', train_loss, epoch)
    # writer.add_scalar('Loss/val', val_loss, epoch)
    
    # 保存最佳模型
    if train_loss < best_loss:
        best_loss = train_loss
        counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
            'emg_scaler': train_dataset.emg_scaler,
            'pressure_scaler': train_dataset.pressure_scaler,
            'train_losses': train_losses,
            'val_losses': val_losses
        }, model_save_path_new)
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(f'./Transformer3/losses/loss_{epoch}.png')  # 保存图像

writer.close()