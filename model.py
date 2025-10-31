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
        # 第一层：线性 -> 激活 -> Dropout
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        # 第二层：维度扩张
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        # 第三层：维度收缩
        x = self.fc3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        # 第四层：压缩特征
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