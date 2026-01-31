
import torch
import torch.nn as nn
import math

class ECAModule(nn.Module):
    """
    Efficient Channel Attention Module
    """
    def __init__(self, channels, gamma=2, b=1):
        super(ECAModule, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        y = self.avg_pool(x) # (B, C, 1)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2) # (B, C, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class TransECANet(nn.Module):
    def __init__(self, num_features, num_classes, d_model=64, nhead=4, num_layers=2):
        super(TransECANet, self).__init__()
        
        # 1. Feature Extraction (1D CNN)
        # Input: (B, 1, num_features)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU()
        
        # 2. ECA Module
        self.eca = ECAModule(channels=d_model)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Classification Head
        # We use Global Average Pooling before FC
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x: (B, num_features) -> Reshape to (B, 1, num_features)
        x = x.unsqueeze(1)
        
        # CNN
        x = self.conv1(x) # (B, d_model, num_features)
        x = self.bn1(x)
        x = self.relu(x)
        
        # ECA
        x = self.eca(x)   # (B, d_model, num_features)
        
        # Transformer expects (B, Seq, Feature)
        # Here we treat 'channels' as features? No, Transformer usually processes a sequence.
        # Option A: Sequence = num_features (Length), Embedding = d_model.
        # Current shape: (B, d_model, Length). 
        # Permute to (B, Length, d_model)
        x = x.permute(0, 2, 1) # (B, num_features, d_model)
        
        x = self.transformer(x) # (B, num_features, d_model)
        
        # Global Pooling over sequence
        # Permute back for pooling: (B, d_model, num_features)
        x = x.permute(0, 2, 1)
        x = self.global_pool(x) # (B, d_model, 1)
        x = x.squeeze(-1)       # (B, d_model)
        
        out = self.fc(x)
        return out
