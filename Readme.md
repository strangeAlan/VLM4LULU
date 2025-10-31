# 路怒症危险行为检测项目（VLM-based）

基于视觉语言模型（VLM）提取的特征，结合多层感知机（MLP）实现三类危险行为的二分类检测，适配小样本场景，支持结果复现与模型灵活选择。

## 环境依赖

- Python 3.8
- torch==1.12.1+cu116
- torchvision==0.13.1+cu116
- numpy==1.23.5
- scikit-learn==1.2.2

## 运行方式

1.  **基础运行：**
    ```bash
    python main.py
    ```

2.  **自定义参数运行：**
    ```bash
    python main.py --model mlp --hidden_size 32 --epochs 30
    ```