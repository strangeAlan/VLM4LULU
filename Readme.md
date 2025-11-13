# 🚗 路怒症危险行为检测项目（VLM-based）

> **基于视觉语言模型（VLM）提取的单模态特征，结合轻量化 Adapter 与分类层实现三类危险行为的二分类检测**

本项目适配小样本场景，支持灵活的模型选择、结果复现与蒸馏训练。项目整合了 **VLM 特征提取** 与 **下游分类任务** 两大核心模块。

---

## 📋 项目概述

### 🎯 核心功能

- **🔍 VLM 特征提取**  
  使用 `InternVL`（图像）与 `VideoMAEv2`（视频）预训练模型作为教师模型，生成高质量单模态特征  

- **🎯 行为检测**  
  在提取特征基础上，引入 **Adapter 层** 进行特征对齐与压缩，再接 **分类层** 完成小样本二分类，支持多任务学习与注意力机制  

## 🏗️ 技术架构

```text
原始视频/图像
    ↓
[VLM 特征提取器]
    ↓ (InternVL + VideoMAEv2)
特征向量
    ↓
[Adapter 层]
    ↓
[分类层]
    ↓
三类危险行为检测 (Label1, Label2, Label3)
```

---

## 📦 安装（Installation）

### 环境要求

请参考 [single_modality/INSTALL.md](./single_modality/INSTALL.md) 进行环境配置。

> 💡 **推荐配置**：Python 3.8 + PyTorch 1.12.1（CUDA 11.6）

### 快速安装

```bash
# 安装依赖
cd single_modality
pip install -r requirements.txt
```

### 模型权重准备

运行 InternVideo2 预训练前，需要准备以下模型权重：

- **[InternVL-6B 视觉编码器](https://huggingface.co/OpenGVLab/InternVL/blob/main/internvl_c_13b_224px.pth)**
- **[VideoMAEv2-g](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md)**

将模型路径配置到 `single_modality/models/internvl_clip_vision.py` 和 `single_modality/models/videomae.py` 中。

---

## 📊 数据集（Datasets）

数据集准备与预处理说明见 [single_modality/DATASET.md](./single_modality/DATASET.md)。

> 📝 **数据集格式**：项目使用 `data.pt` 文件存储特征和标签，格式为：
> ```python
> {
>     'features': torch.Tensor,  # 特征张量
>     'labels': List[int]        # 标签列表（三位二进制编码）
> }
> ```

---

## 🎨 模型库（Model ZOO）

所有预训练模型、特征提取脚本与下游模型结构见 [single_modality/MODEL_ZOO.md](./single_modality/MODEL_ZOO.md)。

---

## 🔧 环境依赖（Dependencies）

### 核心依赖

```text
- Python == 3.8
- torch == 1.12.1+cu116
- torchvision == 0.13.1+cu116
- numpy == 1.23.5
- scikit-learn == 1.2.2
```

> 📦 其他依赖请参考 `single_modality/requirements.txt` 或 `single_modality/INSTALL.md`

---


## 🚀 快速开始（Quick Start）

> **我们已提供一份预提取的 VLM 特征文件 `data.pt`**，您可以**跳过VLM 特征提取步骤**，直接运行验证！

### 一键运行

```bash
python main.py
```

### 预期输出

```
Test Results:
Label1 - Acc: 93.75%, F1: 0.9677
Label2 - Acc: 75.00%, F1: 0.3333
Label3 - Acc: 62.50%, F1: 0.5714
平均Accuracy: 77.08%, 平均F1-score: 0.6242
```

> 秒级完成 | 无需 GPU | 结果完全可复现

## 💡 功能模块

### 1️⃣ VLM 特征提取（Feature Extraction）

使用 `InternVL` 和 `VideoMAEv2` 预训练模型作为教师，提取高质量单模态特征。

#### 📝 特征提取流程

特征提取脚本位于 `single_modality/extrac_feature.py`，支持以下功能：

- ✅ 从预训练模型中提取特征
- ✅ 支持多种数据集格式（Kinetics, SSV2, UCF101, HMDB51 等）
- ✅ 支持分布式特征提取
- ✅ 支持多种精度保存（float16, float32, bfloat16）

#### 🔨 使用方法

```bash
cd single_modality

# 基础特征提取
python extrac_feature.py \
    --model internvideo2_ap_small_patch14_224 \
    --finetune /path/to/checkpoint.pth \
    --data_path /path/to/dataset \
    --extract_features \
    --feature_split test \
    --feature_save_dir ./features/ \
    --batch_size 64 \
    --num_frames 16 \
    --sampling_rate 1
```

#### ⚙️ 关键参数

| 参数 | 说明 |
|------|------|
| `--extract_features` | 启用特征提取模式 |
| `--feature_split` | 数据集划分（train/val/test/all） |
| `--feature_save_dir` | 特征保存目录 |
| `--feature_dtype` | 特征精度（float16/float32/bfloat16） |
| `--sampling_rate` | 采样率（1 表示稀疏采样） |
| `--num_frames` | 视频帧数 |

#### 📌 注意事项

- 运行前请修改脚本中的 `DATA_PATH` 为你的数据集路径
- `--sampling_rate 1` 表示**稀疏采样**
- 特征提取会自动保存为 `.pt` 格式，包含特征、标签和元数据

---

### 2️⃣ 预训练（Pre-Training）

使用 `InternVL` 与 `VideoMAEv2` 预训练模型作为教师，进行单模态特征提取器的自监督预训练。

```bash
cd single_modality
bash ./scripts/pretraining/1B_pt.sh
```

#### ⚠️ 注意事项

- 运行前请修改脚本中的 `DATA_PATH` 为你的数据集路径
- `--sampling_rate 1` 表示**稀疏采样**
- 训练过程中自动保存最新 checkpoint，建议设置较大的 `--save_ckpt_freq`
- 默认教师模型：
  - **InternVideo2-1B / 6B** → 使用 `InternVL-C-13B` + `VideoMAEv2-g`

---

### 3️⃣ 微调（Finetuning）

在预训练特征提取器基础上，对下游分类任务进行全量或参数高效微调。

```bash
cd single_modality
bash ./scripts/finetuning/full_tuning/k400/1B_ft_k710_ft_k400_f8.sh
```

#### ⚠️ 注意事项

- 修改 `DATA_PATH`、`PREFIX` 和 `MODEL_PATH`
- 使用 `--use_checkpoint` 与 `--checkpoint_num` 可节省显存
- 自动评估最佳 checkpoint（`--test_best`）
- 支持多段多裁剪测试：`--test_num_segment` / `--test_num_crop`
- 仅评估模式：添加 `--eval`

---

### 4️⃣ 蒸馏（Distillation）

采用教师-学生蒸馏框架，提升小模型性能与对齐能力。

```bash
cd single_modality
bash ./scripts/distillation/B14_dist_1B_stage2.sh
```

#### ✨ 特点

- 与预训练设置相似，但使用 `MLP_Decoder` 增强特征对齐
- 默认教师模型：`InternVideo2-1B`

#### ⚠️ 注意事项

- 修改 `DATA_PATH`
- `--sampling_rate 1` 为稀疏采样
- 自动保存最新 checkpoint

---

### 5️⃣ 危险行为检测（Main Task）⭐

在提取的 VLM 特征上训练 MLP 分类器，实现三类路怒危险行为的二分类。

#### 🎯 任务说明

- **三类危险行为**：Label1, Label2, Label3（每个标签为二分类）
- **标签格式**：三位二进制编码（例如：`001`, `110`）
- **模型支持**：多种 MLP 架构（MLP, MLP2, SimpleMLP, MultiTaskModel, MultiTaskAttn）

#### 🚀 基础运行

```bash
python main.py
```

#### ⚙️ 自定义参数运行

```bash
python main.py \
    --model mlp \
    --hidden_size 32 \
    --epochs 300 \
    --batch_size 16 \
    --lr 1e-3 \
    --data_path data.pt \
    --dropout 0.3 \
    --seed 1
```

#### 📋 常用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型类型 | `mlp` |
| | 可选：`mlp`, `mlp2`, `SimpleMLP`, `MTM`, `att` | |
| `--hidden_size` | MLP 隐藏层维度 | `16` |
| `--epochs` | 训练轮数 | `300` |
| `--batch_size` | 批次大小 | `16` |
| `--lr` | 学习率 | `0.001` |
| `--data_path` | 数据文件路径（.pt 格式） | `data.pt` |
| `--dropout` | Dropout 率 | `0.3` |
| `--output_size` | 输出类别数 | `2` |
| `--seed` | 随机种子 | `1` |
| `--use_l1` | 是否使用 L1 正则化 | `False` |

#### 🎨 支持的模型架构

1. **MLP**：基础多层感知机，共享隐藏层，三个独立输出头
2. **MLP2**：深层 MLP（4 层），带维度扩张与压缩
3. **SimpleMLP**：简单单层 MLP
4. **MTM (MultiTaskModel)**：多任务模型，共享底层，独立任务头
5. **att (MultiTaskAttn)**：带注意力机制的多任务模型

#### 📊 输出结果

训练完成后会输出：

- 每个标签的准确率（Accuracy）和 F1 分数
- 平均准确率和平均 F1 分数

示例输出：
```
Test Results:
Label1 - Acc: 85.00%, F1: 0.8234
Label2 - Acc: 90.00%, F1: 0.8765
Label3 - Acc: 88.00%, F1: 0.8543
平均Accuracy: 87.67%, 平均F1-score: 0.8514
```

---

## 📁 目录结构

```
VLM4LULU/
├── 📄 main.py                    # 主入口（危险行为检测）
├── 📄 model.py                   # MLP 模型定义
├── 📄 load_data.py               # 数据加载与预处理
├── 📄 process_data.py            # 数据处理工具
├── 📄 data.pt                    # 特征数据文件
│
├── 📂 single_modality/           # VLM 特征提取模块
│   ├── 📄 extrac_feature.py     # 特征提取脚本
│   ├── 📄 README.md              # VLM 模块说明
│   ├── 📄 INSTALL.md              # 安装指南
│   ├── 📄 DATASET.md              # 数据集说明
│   ├── 📄 MODEL_ZOO.md           # 模型库
│   │
│   ├── 📂 models/                 # 模型定义
│   │   ├── internvl_clip_vision.py
│   │   ├── videomae.py
│   │   ├── internvideo2_*.py
│   │   └── ...
│   │
│   ├── 📂 engines/                # 训练引擎
│   │   ├── engine_for_extract.py
│   │   ├── engine_for_pretraining.py
│   │   ├── engine_for_finetuning.py
│   │   └── engine_for_distill.py
│   │
│   ├── 📂 datasets/               # 数据集处理
│   │   ├── kinetics.py
│   │   ├── ssv2.py
│   │   └── ...
│   │
│   └── 📂 scripts/                # 训练脚本
│       ├── pretraining/
│       ├── finetuning/
│       ├── distillation/
│       └── extract/
│
├── 📂 features/                  # 存放提取的 VLM 特征（可选）
├── 📂 outputs/                    # 训练日志与模型（可选）
└── 📄 Readme.md                   # 本文件
```

---

## 📚 引用（Citation）

如使用本项目，请考虑引用以下工作：

```bibtex
@inproceedings{chen2024internvl,
  title={Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks},
  author={Chen, Zhe and Wu, Jiannan and Wang, Wenhai and Su, Weijie and Chen, Guo and Xing, Sen and Zhong, Muyan and Zhang, Qinglong and Zhu, Xizhou and Lu, Lewei and others},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={24185--24198},
  year={2024}
}

@inproceedings{wang2023videomae,
  title={Videomae v2: Scaling video masked autoencoders with dual masking},
  author={Wang, Limin and Huang, Bingkun and Zhao, Zhiyu and Tong, Zhan and He, Yinan and Wang, Yi and Wang, Yali and Qiao, Yu},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={14549--14560},
  year={2023}
}
```

---

## 📄 许可证（License）

[MIT License](./LICENSE)

---

## 💬 支持与反馈

> 💡 如需技术支持或报告问题，请提交 Issue 或联系维护者。

---

<div align="center">

**⭐ 如果这个项目对你有帮助，欢迎 Star！⭐**

</div>
