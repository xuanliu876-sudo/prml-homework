# PRML Homework 4 代码说明

本文件夹包含三个代码文件，分别用于数据下载、模型训练和结果绘图。

## 1. download_multi30k.py

用于下载 Multi30k 英德翻译数据集，并保存到本地 `multi30k_data/` 文件夹。

运行：

```bash
python download_multi30k.py
```

## 2. mini_transformer_multi30k.py

用于训练小型 Transformer 英德翻译模型，是本次作业的核心代码。

主要功能：

- 读取数据集并构建词表；
- 实现并训练 Transformer 模型；
- 支持 `baseline` 和 `no_residual` 两种模式；
- 保存模型、训练日志，并输出翻译示例。

运行 Baseline：

```bash
python mini_transformer_multi30k.py --mode baseline --epochs 20
```

运行无残差模型：

```bash
python mini_transformer_multi30k.py --mode no_residual --epochs 20
```

## 3. plot_results.py

用于读取训练结果并绘制实验对比曲线，包括训练损失、验证损失和验证准确率。

运行：

```bash
python plot_results.py
```

## 4. 推荐运行顺序

```bash
python download_multi30k.py
python mini_transformer_multi30k.py --mode baseline --epochs 20
python mini_transformer_multi30k.py --mode no_residual --epochs 20
python plot_results.py
```

## 5. 依赖库

```bash
pip install torch datasets matplotlib
```
