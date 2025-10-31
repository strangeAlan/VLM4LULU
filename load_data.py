import torch
from sklearn.model_selection import train_test_split
import numpy as np
def load_dataset(path):
    data = torch.load(path)
    # 将 features 和 labels 分别存储到变量中
    features = data['features']  # 对应张量
    features = features.to(dtype=torch.float32)
    labels = data['labels']      # 对应列表
    # 初始化三个类别的列表
    label1, label2, label3 = [], [], []
    # 遍历每个标签，将其拆分为三类
    for label in labels:
        # 将整数转为字符串并补齐为三位二进制格式
        binary_label = f"{label:03}"  # 确保是三位，例如 1 -> '001', 110 -> '110'
        # 按位拆分并转换为整数
        label1.append(int(binary_label[0]))
        label2.append(int(binary_label[1]))
        label3.append(int(binary_label[2]))
    label1 = torch.tensor(label1, dtype=torch.float32)
    label2 = torch.tensor(label2, dtype=torch.float32)
    label3 = torch.tensor(label3, dtype=torch.float32)
    return features, label1, label2, label3
def split_dataset(features, label1, label2, label3, test_size=0.2, random_state=42):
    # 针对极不平衡的label1（77正4负）进行分层抽样：确保测试集至少包含1个负样本
    # 手动控制分层逻辑（避免因样本太少导致分层失效）
    # 1. 分离label1的正负样本
    
    pos_mask = (label1 == 1)
    neg_mask = (label1 == 0)
    
    X_pos, y1_pos, y2_pos, y3_pos = features[pos_mask], label1[pos_mask], label2[pos_mask], label3[pos_mask]
    X_neg, y1_neg, y2_neg, y3_neg = features[neg_mask], label1[neg_mask], label2[neg_mask], label3[neg_mask]
    
    # 2. 测试集负样本取1个（总负样本4个，留1个测试，3个训练）
    # 测试集正样本按比例取：总样本81，测试集20%约16个，减去1个负样本，取15个正样本
    X_neg_train, X_neg_test, y1_neg_train, y1_neg_test = train_test_split(
        X_neg, y1_neg, test_size=1, random_state=42, stratify=y1_neg
    )
    X_pos_train, X_pos_test, y1_pos_train, y1_pos_test = train_test_split(
        X_pos, y1_pos, test_size=15, random_state=42, stratify=y1_pos
    )
    
    # 3. 合并正负样本，组成训练集和测试集（保持其他标签对应关系）
    X_train = np.concatenate([X_pos_train, X_neg_train], axis=0)
    X_test = np.concatenate([X_pos_test, X_neg_test], axis=0)
    
    y1_train = np.concatenate([y1_pos_train, y1_neg_train], axis=0)
    y1_test = np.concatenate([y1_pos_test, y1_neg_test], axis=0)
    
    # 提取对应其他标签的训练/测试集
    y2_train = np.concatenate([y2_pos[pos_mask[pos_mask]][:len(X_pos_train)], y2_neg[neg_mask[neg_mask]][:len(X_neg_train)]], axis=0)
    y2_test = np.concatenate([y2_pos[pos_mask[pos_mask]][len(X_pos_train):], y2_neg[neg_mask[neg_mask]][len(X_neg_train):]], axis=0)
    
    y3_train = np.concatenate([y3_pos[pos_mask[pos_mask]][:len(X_pos_train)], y3_neg[neg_mask[neg_mask]][:len(X_neg_train)]], axis=0)
    y3_test = np.concatenate([y3_pos[pos_mask[pos_mask]][len(X_pos_train):], y3_neg[neg_mask[neg_mask]][len(X_neg_train):]], axis=0)
    
    # 输出训练集和测试集样本数及label1分布
    # print(f"类别一：训练集样本数: {len(X_train)} (正样本: {sum(y1_train==1)}, 负样本: {sum(y1_train==0)})|测试集样本数: {len(X_test)} (正样本: {sum(y1_test==1)}, 负样本: {sum(y1_test==0)})")
    # print(f"类别二：训练集样本数: {len(X_train)} (正样本: {sum(y2_train==1)}, 负样本: {sum(y2_train==0)})|测试集样本数: {len(X_test)} (正样本: {sum(y2_test==1)}, 负样本: {sum(y2_test==0)})")
    # print(f"类别三：训练集样本数: {len(X_train)} (正样本: {sum(y3_train==1)}, 负样本: {sum(y3_train==0)})|测试集样本数: {len(X_test)} (正样本: {sum(y3_test==1)}, 负样本: {sum(y3_test==0)})")
    
    return X_train, X_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test