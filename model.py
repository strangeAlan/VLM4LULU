import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, features, label1, label2, label3):

        self.features = features
        self.label1 = label1
        self.label2 = label2
        self.label3 = label3

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.label1[idx], self.label2[idx], self.label3[idx]
    
# 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,dropout_rate=0.3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3_1 = nn.Linear(hidden_size, output_size)  # 输出给 label1
        self.fc3_2 = nn.Linear(hidden_size, output_size)  # 输出给 label2
        self.fc3_3 = nn.Linear(hidden_size, output_size)  # 输出给 label3

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x1 = self.fc3_1(x)
        x2 = self.fc3_2(x)
        x3 = self.fc3_3(x)
        return x1, x2, x3
    
class MLP2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.3):
        super(MLP2, self).__init__()
        # 4层隐藏层（移除批归一化）
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)  # 维度扩张
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)  # 维度收缩
        self.fc4 = nn.Linear(hidden_size, hidden_size // 2)  # 进一步压缩
        
        # 输出层
        self.fc_out1 = nn.Linear(hidden_size // 2, output_size)
        self.fc_out2 = nn.Linear(hidden_size // 2, output_size)
        self.fc_out3 = nn.Linear(hidden_size // 2, output_size)
        
        self.leaky_relu = nn.LeakyReLU(0.1)  # 避免死神经元
        self.dropout = nn.Dropout(dropout_rate)  # 保留Dropout防过拟合

    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        x = self.leaky_relu(x)
        
        # 输出
        out1 = self.fc_out1(x)
        out2 = self.fc_out2(x)
        out3 = self.fc_out3(x)
        
        return out1, out2, out3
    
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size=16, output_size=2,dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.out1 = nn.Linear(hidden_size, output_size)
        self.out2 = nn.Linear(hidden_size, output_size)
        self.out3 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.out1(x), self.out2(x), self.out3(x)
    
class MultiTaskModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,dropout_rate=0.3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # 每个任务一个小 MLP
        self.task1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.task2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.task3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        h = self.shared(x)
        return self.task1(h), self.task2(h), self.task3(h)
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskAttn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=2,dropout_rate=None):
        super(MultiTaskAttn, self).__init__()
        # 基础层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # 三个任务头，每个输出二分类
        self.task_heads = nn.ModuleList([nn.Linear(hidden_size, output_size) for _ in range(3)])

        # 注意力权重层：根据输入动态分配三个任务的权重
        self.task_attn = nn.Linear(hidden_size, 3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        # 三个任务输出
        outs = [head(x) for head in self.task_heads]  # list of [batch, 2]

        # 注意力权重
        attn_weights = torch.softmax(self.task_attn(x), dim=-1)  # [batch, 3]
        attn_weights = torch.softmax(self.task_attn(x), dim=-1)
        # 可选：综合预测（加权组合三个任务输出）
        # combined = sum(w.unsqueeze(-1) * o for w, o in zip(attn_weights.T, outs))  # [batch, 2]

        return outs, attn_weights

