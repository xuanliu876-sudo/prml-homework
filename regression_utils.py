import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# 读取训练集和测试集数据
def load_data(file_path="Data4Regression.xlsx"):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"找不到文件: {file_path}")
    train_df = pd.read_excel(file_path, sheet_name="Training Data")
    test_df = pd.read_excel(file_path, sheet_name="Test Data")
    X_train = train_df["x"].values.reshape(-1, 1)
    y_train = train_df["y_complex"].values.reshape(-1, 1)
    X_test = test_df["x_new"].values.reshape(-1, 1)
    y_test = test_df["y_new_complex"].values.reshape(-1, 1)
    return X_train, y_train, X_test, y_test


# 计算均方误差
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# 计算均方根误差
def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


# 计算平均绝对误差
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


# 给输入数据添加偏置项
def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


# 使用最小二乘法求解线性回归参数
def least_squares(Xb, y):
    return np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ y


# 使用梯度下降法求解线性回归参数
def gradient_descent(Xb, y, lr=0.01, epochs=5000):
    n, d = Xb.shape
    w = np.zeros((d, 1))
    for _ in range(epochs):
        y_pred = Xb @ w
        grad = (2 / n) * Xb.T @ (y_pred - y)
        w = w - lr * grad
    return w


# 使用牛顿法求解线性回归参数
def newton_method(Xb, y, max_iter=10, tol=1e-12):
    n, d = Xb.shape
    w = np.zeros((d, 1))
    H = (2 / n) * (Xb.T @ Xb)
    H_inv = np.linalg.pinv(H)
    for _ in range(max_iter):
        y_pred = Xb @ w
        grad = (2 / n) * Xb.T @ (y_pred - y)
        step = H_inv @ grad
        w_new = w - step
        if np.linalg.norm(w_new - w) < tol:
            w = w_new
            break
        w = w_new
    return w


# 根据指定方法生成线性回归预测结果
def linear_predictions(X_train, y_train, X_test, method="ls"):
    Xb_train = add_bias(X_train)
    Xb_test = add_bias(X_test)
    if method == "ls":
        w = least_squares(Xb_train, y_train)
        name = "Least Squares Linear Regression"
    elif method == "gd":
        w = gradient_descent(Xb_train, y_train)
        name = "Gradient Descent Linear Regression"
    elif method == "newton":
        w = newton_method(Xb_train, y_train)
        name = "Newton Method Linear Regression"
    else:
        raise ValueError("未知线性方法")
    train_pred = Xb_train @ w
    test_pred = Xb_test @ w
    return {
        "w": w,
        "train_pred": train_pred,
        "test_pred": test_pred,
        "train_mse": mse(y_train, train_pred),
        "name": name
    }


# 根据指定方法返回线性回归完整结果
def linear_result(X_train, y_train, X_test, y_test, method="ls"):
    Xb_train = add_bias(X_train)
    Xb_test = add_bias(X_test)
    if method == "ls":
        w = least_squares(Xb_train, y_train)
        title = "Least Squares Linear Regression"
    elif method == "gd":
        w = gradient_descent(Xb_train, y_train)
        title = "Gradient Descent Linear Regression"
    elif method == "newton":
        w = newton_method(Xb_train, y_train)
        title = "Newton Method Linear Regression"
    else:
        raise ValueError("未知线性方法")
    train_pred = Xb_train @ w
    test_pred = Xb_test @ w
    return {
        "w": w,
        "train_pred": train_pred,
        "test_pred": test_pred,
        "train_mse": mse(y_train, train_pred),
        "test_mse": mse(y_test, test_pred),
        "title": title
    }


# 构造多项式特征矩阵
def polynomial_features(X, degree):
    features = [np.ones((X.shape[0], 1))]
    for p in range(1, degree + 1):
        features.append(X ** p)
    return np.hstack(features)


# 使用最小二乘法拟合多项式回归
def fit_polynomial_least_squares(X, y, degree):
    X_poly = polynomial_features(X, degree)
    return np.linalg.pinv(X_poly.T @ X_poly) @ X_poly.T @ y


# 使用多项式模型进行预测
def predict_polynomial(X, w, degree):
    X_poly = polynomial_features(X, degree)
    return X_poly @ w


# 构造多项式表达式字符串
def build_equation_string(w, zero_tol=1e-10):
    terms = []
    for i in range(len(w)):
        coef = float(w[i, 0])
        if abs(coef) < zero_tol:
            continue
        if i == 0:
            terms.append(f"{coef:.12f}")
        elif i == 1:
            terms.append(f"({coef:.12f})*x")
        else:
            terms.append(f"({coef:.12f})*x^{i}")
    if len(terms) == 0:
        return "y = 0"
    return "y = " + " + ".join(terms)


# 返回多项式回归实验结果
def polynomial_result(X_train, y_train, X_test, y_test, degree_range=range(2, 13)):
    best_degree = None
    best_test_mse = float("inf")
    best_w = None
    all_results = []
    for degree in degree_range:
        w = fit_polynomial_least_squares(X_train, y_train, degree)
        train_pred = predict_polynomial(X_train, w, degree)
        test_pred = predict_polynomial(X_test, w, degree)
        train_mse = mse(y_train, train_pred)
        test_mse = mse(y_test, test_pred)
        all_results.append((degree, train_mse, test_mse))
        if test_mse < best_test_mse:
            best_test_mse = test_mse
            best_degree = degree
            best_w = w
    return {
        "best_degree": best_degree,
        "best_test_mse": best_test_mse,
        "best_w": best_w,
        "all_results": all_results,
        "equation": build_equation_string(best_w)
    }


# 核回归模型类
class KernelRegression:
    # 初始化核回归模型
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.X_train = None
        self.y_train = None

    # 训练核回归模型
    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float).reshape(-1, 1)
        self.y_train = np.asarray(y, dtype=float).reshape(-1, 1)

    # 定义高斯核函数
    def _gaussian_kernel(self, u):
        return np.exp(-0.5 * u ** 2)

    # 使用核回归模型进行预测
    def predict(self, X):
        X = np.asarray(X, dtype=float).reshape(-1, 1)
        distances = (X - self.X_train.T) / self.bandwidth
        weights = self._gaussian_kernel(distances)
        weight_sums = np.sum(weights, axis=1, keepdims=True)
        weight_sums = np.where(weight_sums == 0, 1e-12, weight_sums)
        return (weights @ self.y_train) / weight_sums


# 划分训练集和验证集
def train_valid_split(X, y, valid_ratio=0.2, random_state=42):
    np.random.seed(random_state)
    n = len(X)
    indices = np.random.permutation(n)
    valid_size = int(n * valid_ratio)
    valid_idx = indices[:valid_size]
    train_idx = indices[valid_size:]
    return X[train_idx], y[train_idx], X[valid_idx], y[valid_idx]


# 选择最佳带宽参数
def select_best_bandwidth(X_train, y_train, bandwidth_candidates, valid_ratio=0.2):
    X_subtrain, y_subtrain, X_valid, y_valid = train_valid_split(X_train, y_train, valid_ratio=valid_ratio, random_state=42)
    best_h = None
    best_valid_mse = float("inf")
    records = []
    for h in bandwidth_candidates:
        model = KernelRegression(bandwidth=h)
        model.fit(X_subtrain, y_subtrain)
        y_valid_pred = model.predict(X_valid)
        current_mse = mse(y_valid, y_valid_pred)
        records.append((h, current_mse))
        if current_mse < best_valid_mse:
            best_valid_mse = current_mse
            best_h = h
    return best_h, records


# 返回核回归实验结果
def kernel_result(X_train, y_train, X_test, y_test, bandwidth_candidates=None, valid_ratio=0.2):
    if bandwidth_candidates is None:
        bandwidth_candidates = np.linspace(0.05, 3.0, 60)
    best_h, records = select_best_bandwidth(X_train, y_train, bandwidth_candidates, valid_ratio=valid_ratio)
    model = KernelRegression(bandwidth=best_h)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    result_df = pd.DataFrame({
        "X_test": X_test.flatten(),
        "y_test_true": y_test.flatten(),
        "y_test_pred": y_test_pred.flatten(),
        "abs_error": np.abs(y_test.flatten() - y_test_pred.flatten())
    })
    return {
        "best_h": best_h,
        "records": records,
        "model": model,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "train_mse": mse(y_train, y_train_pred),
        "train_rmse": rmse(y_train, y_train_pred),
        "train_mae": mae(y_train, y_train_pred),
        "test_mse": mse(y_test, y_test_pred),
        "test_rmse": rmse(y_test, y_test_pred),
        "test_mae": mae(y_test, y_test_pred),
        "result_df": result_df
    }


# 绘制线性回归拟合图
def plot_linear_fit(X_train, y_train, X_test, y_test, w, title):
    x_plot = np.linspace(min(X_train.min(), X_test.min()), max(X_train.max(), X_test.max()), 500).reshape(-1, 1)
    xb_plot = add_bias(x_plot)
    y_plot = xb_plot @ w
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train, y_train, label="Training Data", s=20)
    plt.scatter(X_test, y_test, label="Test Data", s=20, alpha=0.7)
    plt.plot(x_plot, y_plot, label=title)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 绘制多项式回归误差曲线图
def plot_polynomial_mse(all_results):
    degrees = [item[0] for item in all_results]
    train_mse_list = [item[1] for item in all_results]
    test_mse_list = [item[2] for item in all_results]
    plt.figure(figsize=(8, 6))
    plt.plot(degrees, train_mse_list, marker="o", label="Train MSE")
    plt.plot(degrees, test_mse_list, marker="s", label="Test MSE")
    plt.title("MSE under Different Polynomial Degrees")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("MSE")
    plt.xticks(degrees)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 绘制最佳多项式拟合图
def plot_polynomial_fit(X_train, y_train, X_test, y_test, best_w, best_degree):
    x_plot = np.linspace(min(X_train.min(), X_test.min()), max(X_train.max(), X_test.max()), 500).reshape(-1, 1)
    y_plot = predict_polynomial(x_plot, best_w, best_degree)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train, y_train, label="Training Data", s=20)
    plt.scatter(X_test, y_test, label="Test Data", s=20, alpha=0.7)
    plt.plot(x_plot, y_plot, label=f"Best Polynomial Fit (Degree={best_degree})")
    plt.title("Best Polynomial Regression Fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 绘制核回归带宽选择图
def plot_kernel_bandwidth(records):
    hs = [r[0] for r in records]
    errors = [r[1] for r in records]
    plt.figure(figsize=(8, 5))
    plt.plot(hs, errors, marker="o")
    plt.xlabel("Bandwidth h")
    plt.ylabel("Validation MSE")
    plt.title("Bandwidth Selection")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# 绘制核回归拟合图
def plot_kernel_fit(X_train, y_train, X_test, y_test, model):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, label="Train Data", alpha=0.7)
    plt.scatter(X_test, y_test, label="Test Data", alpha=0.7)
    x_min = min(X_train.min(), X_test.min())
    x_max = max(X_train.max(), X_test.max())
    x_plot = np.linspace(x_min, x_max, 500).reshape(-1, 1)
    y_plot = model.predict(x_plot)
    plt.plot(x_plot, y_plot, linewidth=2, label=f"Kernel Regression (h={model.bandwidth:.4f})")
    plt.title("Kernel Regression on Data4Regression")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# 打印线性回归结果并绘图
def print_linear_result(X_train, y_train, X_test, y_test, result):
    print(f"w0 = {result['w'][0, 0]:.6f}")
    print(f"w1 = {result['w'][1, 0]:.6f}")
    print(f"Train MSE = {result['train_mse']:.6f}")
    print(f"Test MSE = {result['test_mse']:.6f}")
    plot_linear_fit(X_train, y_train, X_test, y_test, result["w"], result["title"])