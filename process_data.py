import torch
from collections import Counter
# 读取 data.pt 文件
data = torch.load('data.pt')

# 将 features 和 labels 分别存储到变量中
features = data['features']  # 对应张量
labels = data['labels']      # 对应列表

# 打印简单验证信息
print("Features 类型：", type(features), "形状：", features.shape)
print("Labels 类型：", type(labels), "长度：", len(labels))
print("feature:",features)
print("label:",labels)
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
# 换成torch类型
# label1 = torch.tensor(label1, dtype=torch.float32)
# label2 = torch.tensor(label2, dtype=torch.float32)
# label3 = torch.tensor(label3, dtype=torch.float32)
# 统计每个类别中 0 和 1 的数量
label1_count = Counter(label1)
label2_count = Counter(label2)
label3_count = Counter(label3)

# 打印结果验证
print("Label1 拆分结果:", label1)
print("Label2 拆分结果:", label2)
print("Label3 拆分结果:", label3)

print("\n统计结果：")
print(f"Label1: 0 的数量 = {label1_count[0]}, 1 的数量 = {label1_count[1]}")
print(f"Label2: 0 的数量 = {label2_count[0]}, 1 的数量 = {label2_count[1]}")
print(f"Label3: 0 的数量 = {label3_count[0]}, 1 的数量 = {label3_count[1]}")