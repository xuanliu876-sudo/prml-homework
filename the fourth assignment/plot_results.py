# plot_results.py
# 用于绘制 Transformer 残差连接消融实验结果图
# 运行前请确保已经有：
# baseline_mini_transformer.pt
# no_residual_mini_transformer.pt

from pathlib import Path

import torch
import matplotlib.pyplot as plt


def load_log(path):
    ckpt = torch.load(path, map_location="cpu")

    train_losses = ckpt["train_losses"]
    valid_losses = ckpt["valid_losses"]
    valid_accs = ckpt["valid_accs"]

    return train_losses, valid_losses, valid_accs


def plot_single_curve(values, ylabel, title, save_path):
    epochs = list(range(1, len(values) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, values, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_compare_curve(values1, values2, label1, label2, ylabel, title, save_path):
    min_len = min(len(values1), len(values2))
    epochs = list(range(1, min_len + 1))

    values1 = values1[:min_len]
    values2 = values2[:min_len]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, values1, marker="o", label=label1)
    plt.plot(epochs, values2, marker="s", label=label2)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    baseline_path = Path("baseline_mini_transformer.pt")
    no_residual_path = Path("no_residual_mini_transformer.pt")

    if not baseline_path.exists():
        raise FileNotFoundError("找不到 baseline_mini_transformer.pt，请先运行 baseline 实验。")

    if not no_residual_path.exists():
        raise FileNotFoundError("找不到 no_residual_mini_transformer.pt，请先运行 no_residual 实验。")

    baseline_train, baseline_valid, baseline_acc = load_log(baseline_path)
    no_res_train, no_res_valid, no_res_acc = load_log(no_residual_path)

    baseline_acc_percent = [x * 100 for x in baseline_acc]
    no_res_acc_percent = [x * 100 for x in no_res_acc]

    # 单独绘制 Baseline
    plot_single_curve(
        baseline_train,
        ylabel="Train Loss",
        title="Baseline Training Loss",
        save_path="baseline_train_loss.png"
    )

    plot_single_curve(
        baseline_valid,
        ylabel="Valid Loss",
        title="Baseline Validation Loss",
        save_path="baseline_valid_loss.png"
    )

    plot_single_curve(
        baseline_acc_percent,
        ylabel="Valid Token Accuracy (%)",
        title="Baseline Validation Token Accuracy",
        save_path="baseline_valid_acc.png"
    )

    # 单独绘制 No-Residual
    plot_single_curve(
        no_res_train,
        ylabel="Train Loss",
        title="No-Residual Training Loss",
        save_path="no_residual_train_loss.png"
    )

    plot_single_curve(
        no_res_valid,
        ylabel="Valid Loss",
        title="No-Residual Validation Loss",
        save_path="no_residual_valid_loss.png"
    )

    plot_single_curve(
        no_res_acc_percent,
        ylabel="Valid Token Accuracy (%)",
        title="No-Residual Validation Token Accuracy",
        save_path="no_residual_valid_acc.png"
    )

    # 绘制对比图
    plot_compare_curve(
        baseline_train,
        no_res_train,
        label1="Baseline with Residual",
        label2="No Residual",
        ylabel="Train Loss",
        title="Training Loss Comparison",
        save_path="comparison_train_loss.png"
    )

    plot_compare_curve(
        baseline_valid,
        no_res_valid,
        label1="Baseline with Residual",
        label2="No Residual",
        ylabel="Valid Loss",
        title="Validation Loss Comparison",
        save_path="comparison_valid_loss.png"
    )

    plot_compare_curve(
        baseline_acc_percent,
        no_res_acc_percent,
        label1="Baseline with Residual",
        label2="No Residual",
        ylabel="Valid Token Accuracy (%)",
        title="Validation Token Accuracy Comparison",
        save_path="comparison_valid_acc.png"
    )

    print("绘图完成！已生成以下图片：")
    print("1. baseline_train_loss.png")
    print("2. baseline_valid_loss.png")
    print("3. baseline_valid_acc.png")
    print("4. no_residual_train_loss.png")
    print("5. no_residual_valid_loss.png")
    print("6. no_residual_valid_acc.png")
    print("7. comparison_train_loss.png")
    print("8. comparison_valid_loss.png")
    print("9. comparison_valid_acc.png")


if __name__ == "__main__":
    main()