import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ======== 数据生成 ========
def make_moons_3d(n_samples=500, noise=0.1, random_state=None):
    """生成3D双月形数据集，返回样本数为 2*n_samples"""
    rng = np.random.default_rng(random_state)

    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)

    # 第二个月牙通过坐标翻转和平移生成
    X = np.vstack([
        np.column_stack([x, y, z]),
        np.column_stack([-x, y - 1, -z])
    ])
    labels = np.hstack([
        np.zeros(n_samples, dtype=int),
        np.ones(n_samples, dtype=int)
    ])

    # 添加高斯噪声
    X += rng.normal(scale=noise, size=X.shape)
    return X, labels


# ======== 全局参数设置 ========
RANDOM_STATE = 42
TRAIN_PER_CLASS = 500   # 每类训练样本数，总计1000
TEST_PER_CLASS = 250    # 每类测试样本数，总计500
NOISE = 0.2

os.makedirs("figures", exist_ok=True)

X_train, y_train = make_moons_3d(
    n_samples=TRAIN_PER_CLASS,
    noise=NOISE,
    random_state=RANDOM_STATE
)

X_test, y_test = make_moons_3d(
    n_samples=TEST_PER_CLASS,
    noise=NOISE,
    random_state=RANDOM_STATE + 1
)


# ======== 3D数据可视化 ========
def plot_3d_data(X, y, title, save_path):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        X[:, 0], X[:, 1], X[:, 2],
        c=y, cmap='viridis', marker='o', alpha=0.65, s=12
    )
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


plot_3d_data(
    X_train, y_train,
    '3D Make Moons - Training Data (1000 samples)',
    'figures/fig1_train_data.png'
)

plot_3d_data(
    X_test, y_test,
    '3D Make Moons - Test Data (500 samples)',
    'figures/fig2_test_data.png'
)


# ======== 辅助函数：模型评估与分组柱状图 ========
def evaluate_model(model, X_train, y_train, X_test, y_test):
    """训练模型并计算各项指标"""
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "train_acc": accuracy_score(y_train, y_pred_train),
        "test_acc": accuracy_score(y_test, y_pred_test),
        "precision": precision_score(y_test, y_pred_test),
        "recall": recall_score(y_test, y_pred_test),
        "f1": f1_score(y_test, y_pred_test),
        "confusion_matrix": confusion_matrix(y_test, y_pred_test).tolist(),
        "classification_report": classification_report(y_test, y_pred_test, output_dict=True)
    }
    return metrics, y_pred_test


def plot_grouped_bar(xlabels, train_vals, test_vals, xlabel, ylabel, title, save_path, ylim=None):
    """绘制训练/测试准确率的分组柱状图"""
    x = np.arange(len(xlabels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, train_vals, width, label='Train Accuracy')
    bars2 = ax.bar(x + width / 2, test_vals, width, label='Test Accuracy')

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    if ylim is not None:
        ax.set_ylim(*ylim)

    # 在柱状图顶部标注数值
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.005,
                f"{h:.4f}",
                ha='center',
                va='bottom',
                fontsize=9
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# 5折分层交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# 存储所有最优模型的评估结果
results = {}


# ======== 决策树：搜索最佳深度 ========
dt_results = {}
dt_depths = [3, 5, 10, None]

best_dt_test = -1.0
best_dt_depth = None
best_dt_model = None
best_dt_metrics = None
best_dt_pred = None

for depth in dt_depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
    metrics, y_pred = evaluate_model(dt, X_train, y_train, X_test, y_test)

    dt_results[str(depth)] = {
        "train_acc": metrics["train_acc"],
        "test_acc": metrics["test_acc"]
    }

    print(f"DT depth={depth}: train={metrics['train_acc']:.4f}, test={metrics['test_acc']:.4f}")

    # 记录测试集最优的深度
    if metrics["test_acc"] > best_dt_test:
        best_dt_test = metrics["test_acc"]
        best_dt_depth = depth
        best_dt_model = clone(dt)
        best_dt_metrics = metrics
        best_dt_pred = y_pred

results[f"Decision Tree (depth={best_dt_depth})"] = best_dt_metrics


# ======== AdaBoost：网格搜索最佳超参数（交叉验证） ========
ada_search_space = {
    "base_depth": [1, 2, 3, 4, 5, 10],
    "n_estimators": [50, 100, 200, 300, 500],
    "learning_rate": [0.1, 0.3, 0.5, 0.8, 1.0]
}

ada_cv_records = []
best_ada_cv_score = -1.0
best_ada_params = None
best_ada_model = None

for depth in ada_search_space["base_depth"]:
    for n_est in ada_search_space["n_estimators"]:
        for lr in ada_search_space["learning_rate"]:
            base_tree = DecisionTreeClassifier(
                max_depth=depth,
                random_state=RANDOM_STATE
            )

            ada = AdaBoostClassifier(
                estimator=base_tree,
                n_estimators=n_est,
                learning_rate=lr,
                random_state=RANDOM_STATE
            )

            cv_scores = cross_val_score(
                ada, X_train, y_train,
                cv=cv,
                scoring="accuracy",
                n_jobs=None
            )
            mean_cv = cv_scores.mean()

            ada_cv_records.append({
                "base_depth": depth,
                "n_estimators": n_est,
                "learning_rate": lr,
                "cv_acc_mean": float(mean_cv)
            })

            if mean_cv > best_ada_cv_score:
                best_ada_cv_score = mean_cv
                best_ada_params = {
                    "base_depth": depth,
                    "n_estimators": n_est,
                    "learning_rate": lr
                }
                best_ada_model = clone(ada)

print("\nBest AdaBoost params from CV:")
print(best_ada_params)
print(f"Best AdaBoost CV accuracy: {best_ada_cv_score:.4f}")

# 用最优参数在完整训练集上训练并在测试集上评估
ada_best_metrics, y_pred_ada = evaluate_model(
    best_ada_model, X_train, y_train, X_test, y_test
)

results[
    f"AdaBoost + DT (depth={best_ada_params['base_depth']}, "
    f"n={best_ada_params['n_estimators']}, "
    f"lr={best_ada_params['learning_rate']})"
] = ada_best_metrics

# 固定最优深度和学习率，绘制不同弱分类器数量的效果对比
ada_plot_depth = best_ada_params["base_depth"]
ada_plot_lr = best_ada_params["learning_rate"]

ada_results = {}
for n_est in [50, 100, 200]:
    ada_tmp = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=ada_plot_depth, random_state=RANDOM_STATE),
        n_estimators=n_est,
        learning_rate=ada_plot_lr,
        random_state=RANDOM_STATE
    )
    metrics, _ = evaluate_model(ada_tmp, X_train, y_train, X_test, y_test)
    ada_results[str(n_est)] = {
        "train_acc": metrics["train_acc"],
        "test_acc": metrics["test_acc"]
    }
    print(
        f"AdaBoost plot-setting depth={ada_plot_depth}, lr={ada_plot_lr}, n={n_est}: "
        f"train={metrics['train_acc']:.4f}, test={metrics['test_acc']:.4f}"
    )


# ======== SVM：对每种核函数分别搜索最优参数（交叉验证） ========
svm_search_spaces = {
    "Linear": {
        "C": [0.1, 1, 10, 100]
    },
    "Polynomial (d=3)": {
        "C": [0.1, 1, 10],
        "gamma": ["scale", 0.1, 1.0],
        "coef0": [0.0, 1.0]
    },
    "RBF": {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", 0.01, 0.1, 1.0]
    },
    "Sigmoid": {
        "C": [0.1, 1, 10],
        "gamma": ["scale", 0.01, 0.1, 1.0],
        "coef0": [0.0, 1.0]
    }
}

svm_best_params = {}
svm_predictions = {}
svm_plot_train = []
svm_plot_test = []
svm_plot_names = []

for kernel_name, search_space in svm_search_spaces.items():
    best_cv_score = -1.0
    best_params = None
    best_model = None

    # 线性核
    if kernel_name == "Linear":
        for C in search_space["C"]:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("svc", SVC(kernel="linear", C=C, random_state=RANDOM_STATE))
            ])
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=None)
            mean_cv = cv_scores.mean()
            if mean_cv > best_cv_score:
                best_cv_score = mean_cv
                best_params = {"C": C}
                best_model = clone(model)

    # 多项式核
    elif kernel_name == "Polynomial (d=3)":
        for C in search_space["C"]:
            for gamma in search_space["gamma"]:
                for coef0 in search_space["coef0"]:
                    model = Pipeline([
                        ("scaler", StandardScaler()),
                        ("svc", SVC(
                            kernel="poly",
                            degree=3,
                            C=C,
                            gamma=gamma,
                            coef0=coef0,
                            random_state=RANDOM_STATE
                        ))
                    ])
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=None)
                    mean_cv = cv_scores.mean()
                    if mean_cv > best_cv_score:
                        best_cv_score = mean_cv
                        best_params = {"C": C, "gamma": gamma, "coef0": coef0}
                        best_model = clone(model)

    # 径向基核（高斯核）
    elif kernel_name == "RBF":
        for C in search_space["C"]:
            for gamma in search_space["gamma"]:
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("svc", SVC(
                        kernel="rbf",
                        C=C,
                        gamma=gamma,
                        random_state=RANDOM_STATE
                    ))
                ])
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=None)
                mean_cv = cv_scores.mean()
                if mean_cv > best_cv_score:
                    best_cv_score = mean_cv
                    best_params = {"C": C, "gamma": gamma}
                    best_model = clone(model)

    # Sigmoid核
    elif kernel_name == "Sigmoid":
        for C in search_space["C"]:
            for gamma in search_space["gamma"]:
                for coef0 in search_space["coef0"]:
                    model = Pipeline([
                        ("scaler", StandardScaler()),
                        ("svc", SVC(
                            kernel="sigmoid",
                            C=C,
                            gamma=gamma,
                            coef0=coef0,
                            random_state=RANDOM_STATE
                        ))
                    ])
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=None)
                    mean_cv = cv_scores.mean()
                    if mean_cv > best_cv_score:
                        best_cv_score = mean_cv
                        best_params = {"C": C, "gamma": gamma, "coef0": coef0}
                        best_model = clone(model)

    svm_best_params[kernel_name] = {
        "best_cv_accuracy": float(best_cv_score),
        "best_params": best_params
    }

    metrics, y_pred = evaluate_model(best_model, X_train, y_train, X_test, y_test)
    results[f"SVM ({kernel_name})"] = metrics
    svm_predictions[kernel_name] = y_pred
    svm_plot_names.append(kernel_name)
    svm_plot_train.append(metrics["train_acc"])
    svm_plot_test.append(metrics["test_acc"])

    print(f"\nBest SVM params for {kernel_name}: {best_params}")
    print(f"Best CV accuracy for {kernel_name}: {best_cv_score:.4f}")
    print(f"SVM {kernel_name}: train={metrics['train_acc']:.4f}, test={metrics['test_acc']:.4f}")


# ======== 绘制对比图 ========

# 图3：决策树不同深度对比
plot_grouped_bar(
    xlabels=['3', '5', '10', 'None'],
    train_vals=[dt_results[str(d)]["train_acc"] for d in [3, 5, 10, None]],
    test_vals=[dt_results[str(d)]["test_acc"] for d in [3, 5, 10, None]],
    xlabel='Max Depth',
    ylabel='Accuracy',
    title='Decision Tree: Accuracy vs Max Depth',
    save_path='figures/fig3_dt_depth.png',
    ylim=(0.6, 1.02)
)

# 图4：AdaBoost不同弱分类器数量对比
plot_grouped_bar(
    xlabels=['50', '100', '200'],
    train_vals=[ada_results[str(n)]["train_acc"] for n in [50, 100, 200]],
    test_vals=[ada_results[str(n)]["test_acc"] for n in [50, 100, 200]],
    xlabel='Number of Estimators',
    ylabel='Accuracy',
    title=f'AdaBoost + DT (depth={ada_plot_depth}, lr={ada_plot_lr}): Accuracy vs Number of Estimators',
    save_path='figures/fig4_adaboost.png',
    ylim=(0.6, 1.02)
)

# 图5：SVM不同核函数对比（均使用各自最优参数）
plot_grouped_bar(
    xlabels=svm_plot_names,
    train_vals=svm_plot_train,
    test_vals=svm_plot_test,
    xlabel='Kernel',
    ylabel='Accuracy',
    title='SVM: Accuracy vs Kernel Function (Best Parameters)',
    save_path='figures/fig5_svm_kernels.png',
    ylim=(0.4, 1.05)
)

# 图6：所有最优模型的测试准确率总对比
all_names = list(results.keys())
all_test_acc = [results[name]["test_acc"] for name in all_names]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(range(len(all_names)), all_test_acc)
ax.set_xticks(range(len(all_names)))
ax.set_xticklabels(all_names, rotation=25, ha='right', fontsize=9)
ax.set_ylabel('Test Accuracy')
ax.set_title('Overall Comparison of All Methods (Best Parameters)')
ax.set_ylim(0.4, 1.05)

for bar, val in zip(bars, all_test_acc):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.01,
        f'{val:.4f}',
        ha='center',
        va='bottom',
        fontsize=8
    )

plt.tight_layout()
plt.savefig('figures/fig6_overall.png', dpi=150)
plt.close()

# 图7：各最优模型的混淆矩阵
preds = {
    f"DT (depth={best_dt_depth})": best_dt_pred,
    f"AdaBoost (depth={best_ada_params['base_depth']}, n={best_ada_params['n_estimators']})": y_pred_ada,
    "SVM Linear": svm_predictions["Linear"],
    "SVM Polynomial (d=3)": svm_predictions["Polynomial (d=3)"],
    "SVM RBF": svm_predictions["RBF"],
    "SVM Sigmoid": svm_predictions["Sigmoid"]
}

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for ax, (name, pred) in zip(axes.flat, preds.items()):
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(name, fontsize=10)

plt.suptitle('Confusion Matrices of Best Models', fontsize=14)
plt.tight_layout()
plt.savefig('figures/fig7_confusion.png', dpi=150)
plt.close()


# ======== 保存结果到JSON ========
serializable_results = {}
for k, v in results.items():
    serializable_results[k] = {
        "train_acc": v["train_acc"],
        "test_acc": v["test_acc"],
        "precision": v["precision"],
        "recall": v["recall"],
        "f1": v["f1"],
        "confusion_matrix": v["confusion_matrix"]
    }

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(serializable_results, f, indent=2, ensure_ascii=False)

with open("dt_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "all_depth_results": dt_results,
        "best_depth": str(best_dt_depth),
        "best_test_acc": best_dt_test
    }, f, indent=2, ensure_ascii=False)

with open("ada_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "best_params": best_ada_params,
        "best_cv_accuracy": best_ada_cv_score,
        "plot_results": ada_results,
        "cv_records": ada_cv_records
    }, f, indent=2, ensure_ascii=False)

with open("svm_results.json", "w", encoding="utf-8") as f:
    json.dump(svm_best_params, f, indent=2, ensure_ascii=False)

# 打印最终汇总
print("\n===== 最终结果汇总（最优参数） =====")
for name, r in serializable_results.items():
    print(
        f"{name}: "
        f"train_acc={r['train_acc']:.4f}, "
        f"test_acc={r['test_acc']:.4f}, "
        f"precision={r['precision']:.4f}, "
        f"recall={r['recall']:.4f}, "
        f"f1={r['f1']:.4f}"
    )

print("\n最优决策树深度:", best_dt_depth)
print("最优AdaBoost参数:", best_ada_params)
print("各核函数最优SVM参数:", svm_best_params)
print("\n所有图表已保存至 figures/")