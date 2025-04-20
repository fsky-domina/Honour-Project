import torch
import torch.nn as nn


class TimeReduction1(nn.Module):
    def __init__(self, in_channels=8, reduction_ratio=100):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=reduction_ratio,
            stride=reduction_ratio
        )
    
    def forward(self, x):
        # x: [B, T=1000, C=1]
        #print(x.shape)
        self.original_length = x.size(1)  # 获取输入序列的实际长度
        x = x.permute(0, 2, 1)  # [B, C, T]
        x = self.conv(x)        # [B, 1, T//20=50]
        return x.permute(0, 2, 1)  # [B, 50, 1]
    
class TimeRecovery(nn.Module):
    def __init__(self, in_channels=8, expansion_ratio=20):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=expansion_ratio,
            stride=expansion_ratio
        )
    
    def forward(self, x):
        # x: [B, T=50, C=8]
        x = x.permute(0, 2, 1)  # [B, C, T]
        x = self.conv_transpose(x)  # [B, C, T*expansion_ratio]
        return x.permute(0, 2, 1)  # [B, T*expansion_ratio, C]

class TimeReduction(nn.Module):
    def __init__(self, in_channels=8, reduction_ratio=5):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=reduction_ratio,
            stride=reduction_ratio
        )
    
    def forward(self, x):
        # x: [B, T=1000, C=1]
        #print(x.shape)
        self.original_length = x.size(1)  # 获取输入序列的实际长度
        x = x.permute(0, 2, 1)  # [B, C, T]
        x = self.conv(x)        # [B, 1, T//20=50]
        return x.permute(0, 2, 1)  # [B, 50, 1]
    


class EMGTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=32, num_layers=1, nhead=2):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model//4),
            nn.GELU(),
            nn.LayerNorm(d_model//4),
            nn.Linear(d_model//4, d_model//4),
            nn.GELU(),
            nn.LayerNorm(d_model//4)
        )

        self.time_compress1 = TimeReduction1()
        self.pos_encoder = nn.Embedding(60000, d_model//4)

        self.middle_proj = nn.Sequential(
            nn.Linear(d_model//4, d_model//2),
            nn.GELU(),
            nn.LayerNorm(d_model//2),
            nn.Linear(d_model//2, d_model//4),
            nn.GELU(),
            nn.LayerNorm(d_model//4)
        )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model//4,
            nhead=nhead,
            dim_feedforward=8,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.time_recovery = TimeRecovery()
        
        # 输出头
        self.output_head = nn.Sequential(
            nn.Linear(d_model//4, 16),
            nn.ReLU(),
            nn.LayerNorm(16),
            nn.Linear(16, 1)
        )
    
    def forward(self, src):
        # src: [B, 1000, 1]
        x = self.input_proj(src)  # [B, 1000, d_model]

        x = self.time_compress1(x)
        self.original_length = x.size(1)
        
        # 位置编码
        positions = torch.arange(self.original_length, device=src.device).expand(x.size(0), -1)
        x = x + self.pos_encoder(positions)

        x = self.middle_proj(x)
        
        # Transformer处理
        x = self.encoder(x)  # [B, 50, d_model]

        x = self.time_recovery(x)
        
        # 输出预测
        return self.output_head(x)  # [B, 50, 1]