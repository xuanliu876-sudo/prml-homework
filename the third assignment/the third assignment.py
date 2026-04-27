import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# 1. 基本参数设置

TRAIN_FILE = "LSTM-Multivariate_pollution.csv"
TEST_FILE = "pollution_test_data1.csv"

LOOK_BACK = 24
BATCH_SIZE = 64
EPOCHS = 80
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
WEIGHT_DECAY = 1e-5
PATIENCE = 10

USE_ATTENTION_MODEL = False  # True为Attention-LSTM，False为Basic-LSTM

MODEL_TAG = "attention_lstm" if USE_ATTENTION_MODEL else "basic_lstm"
OUTPUT_DIR = f"outputs_{MODEL_TAG}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
print("Output directory:", OUTPUT_DIR)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# 2. 读取与预处理数据

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)

    df.columns = [col.strip() for col in df.columns]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

    for col in df.columns:
        if col not in ["date", "wnd_dir"]:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].ffill()
                df[col] = df[col].bfill()

    if "wnd_dir" in df.columns:
        df["wnd_dir"] = df["wnd_dir"].ffill()
        df["wnd_dir"] = df["wnd_dir"].bfill()

    return df


train_df = load_and_preprocess(TRAIN_FILE)
test_df = load_and_preprocess(TEST_FILE)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Train columns:", train_df.columns.tolist())
print("Test columns:", test_df.columns.tolist())


# 3. 原始数据可视化

def plot_raw_data(df, output_dir):
    if "date" in df.columns:
        x = df["date"]
        x_label = "Date"
    else:
        x = np.arange(len(df))
        x_label = "Index"

    if "pollution" in df.columns:
        plt.figure(figsize=(12, 5))
        plt.plot(x, df["pollution"], label="Pollution / PM2.5")
        plt.xlabel(x_label)
        plt.ylabel("Pollution / PM2.5")
        plt.title("Raw PM2.5 Pollution Concentration Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(output_dir, "raw_pollution_series.png")
        plt.savefig(save_path, dpi=300)
        plt.show()

        print("Saved raw pollution plot to:", save_path)

    raw_feature_cols = [
        col for col in ["dew", "temp", "press", "wnd_spd", "snow", "rain"]
        if col in df.columns
    ]

    if len(raw_feature_cols) > 0:
        plt.figure(figsize=(12, 6))
        for col in raw_feature_cols:
            plt.plot(x, df[col], label=col)

        plt.xlabel(x_label)
        plt.ylabel("Raw Feature Value")
        plt.title("Raw Meteorological Variables Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(output_dir, "raw_features_series.png")
        plt.savefig(save_path, dpi=300)
        plt.show()

        print("Saved raw features plot to:", save_path)


plot_raw_data(train_df, OUTPUT_DIR)


# 4. 风向编码

if "wnd_dir" in train_df.columns:
    le = LabelEncoder()

    all_wnd_dir = pd.concat(
        [
            train_df["wnd_dir"].astype(str),
            test_df["wnd_dir"].astype(str)
        ],
        axis=0
    )

    le.fit(all_wnd_dir)

    train_df["wnd_dir"] = le.transform(train_df["wnd_dir"].astype(str))
    test_df["wnd_dir"] = le.transform(test_df["wnd_dir"].astype(str))

    print("Wind direction classes:", list(le.classes_))


# 5. 选择特征与预测目标

candidate_features = [
    "pollution",
    "dew",
    "temp",
    "press",
    "wnd_dir",
    "wnd_spd",
    "snow",
    "rain"
]

features = [col for col in candidate_features if col in train_df.columns]

target_col = "pollution"

if target_col not in features:
    raise ValueError("训练集中必须包含pollution列，因为目标是预测PM2.5污染浓度。")

target_index = features.index(target_col)

print("Used features:", features)
print("Target column:", target_col)


# 6. 数据归一化

scaler = MinMaxScaler(feature_range=(0, 1))

train_values = train_df[features].values.astype("float32")
train_scaled = scaler.fit_transform(train_values)

test_has_target = target_col in test_df.columns

if test_has_target:
    test_values = test_df[features].values.astype("float32")
else:
    test_temp = test_df.copy()
    test_temp[target_col] = 0.0
    test_temp = test_temp[features]
    test_values = test_temp.values.astype("float32")

test_scaled = scaler.transform(test_values)


# 7. 构造时间序列样本

def create_sequences(data, look_back, target_index):
    X, y = [], []

    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back, :])
        y.append(data[i + look_back, target_index])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


X_all, y_all = create_sequences(train_scaled, LOOK_BACK, target_index)

# 按时间顺序划分训练集和验证集，避免时间序列数据泄漏
train_size = int(len(X_all) * 0.8)

X_train = X_all[:train_size]
y_train = y_all[:train_size]

X_val = X_all[train_size:]
y_val = y_all[train_size:]

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)


class AirQualityDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if self.y is None:
            return self.X[index]
        return self.X[index], self.y[index]


train_dataset = AirQualityDataset(X_train, y_train)
val_dataset = AirQualityDataset(X_val, y_val)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# 8. 定义LSTM模型

class BasicLSTMForecastModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(BasicLSTMForecastModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)

        last_out = out[:, -1, :]

        last_out = self.layer_norm(last_out)
        last_out = self.dropout(last_out)

        prediction = self.fc(last_out)

        return prediction.squeeze(-1)


class AttentionLSTMForecastModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(AttentionLSTMForecastModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)

        attention_scores = self.attention(out)

        attention_weights = torch.softmax(attention_scores, dim=1)

        context = torch.sum(attention_weights * out, dim=1)

        context = self.layer_norm(context)
        context = self.dropout(context)

        prediction = self.fc(context)

        return prediction.squeeze(-1)


input_size = len(features)

if USE_ATTENTION_MODEL:
    model = AttentionLSTMForecastModel(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    model_name = "Attention-LSTM"
else:
    model = BasicLSTMForecastModel(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    model_name = "Basic-LSTM"

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5
)

print("Model type:", model_name)
print(model)


# 9. 训练模型

train_losses = []
val_losses = []

best_val_loss = float("inf")
best_model_path = os.path.join(
    OUTPUT_DIR,
    f"best_{MODEL_TAG}_pollution_model.pth"
)
early_stop_counter = 0

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        loss.backward()

        # 梯度裁剪，缓解LSTM训练中的梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_train_loss += loss.item() * X_batch.size(0)

    avg_train_loss = total_train_loss / len(train_dataset)

    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_val_loss += loss.item() * X_batch.size(0)

    avg_val_loss = total_val_loss / len(val_dataset)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    scheduler.step(avg_val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {avg_train_loss:.6f} "
        f"Val Loss: {avg_val_loss:.6f} "
        f"LR: {current_lr:.8f}"
    )

    # 保存验证集最优模型，并进行Early Stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        early_stop_counter += 1

    if early_stop_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch + 1}.")
        break


model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
print("Loaded best model from:", best_model_path)


# 10. 绘制训练损失曲线

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title(f"Training and Validation Loss ({model_name})")
plt.legend()
plt.grid(True)
plt.tight_layout()

loss_plot_path = os.path.join(
    OUTPUT_DIR,
    f"{MODEL_TAG}_training_validation_loss.png"
)
plt.savefig(loss_plot_path, dpi=300)
plt.show()

print("Saved loss plot to:", loss_plot_path)


# 11. 反归一化函数与评价函数

def inverse_transform_pollution(scaled_pollution, scaler, target_index, feature_num):
    # scaler基于全部特征拟合，因此反归一化时需要构造完整特征维度
    dummy = np.zeros((len(scaled_pollution), feature_num), dtype=np.float32)
    dummy[:, target_index] = scaled_pollution

    inversed = scaler.inverse_transform(dummy)

    return inversed[:, target_index]


def evaluate_predictions(true_values, pred_values, dataset_name):
    mae = mean_absolute_error(true_values, pred_values)
    rmse = math.sqrt(mean_squared_error(true_values, pred_values))

    print(f"\n{dataset_name} Results:")
    print(f"{dataset_name} MAE: {mae:.4f}")
    print(f"{dataset_name} RMSE: {rmse:.4f}")

    return mae, rmse


# 12. 验证集预测与评价

model.eval()

with torch.no_grad():
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    val_pred_scaled = model(X_val_tensor).cpu().numpy()

val_true = inverse_transform_pollution(
    y_val,
    scaler,
    target_index,
    len(features)
)

val_pred = inverse_transform_pollution(
    val_pred_scaled,
    scaler,
    target_index,
    len(features)
)

val_mae, val_rmse = evaluate_predictions(
    val_true,
    val_pred,
    "Validation"
)

plt.figure(figsize=(10, 5))
plt.plot(val_true[:300], label="True Pollution")
plt.plot(val_pred[:300], label="Predicted Pollution")
plt.xlabel("Time Step")
plt.ylabel("Pollution / PM2.5")
plt.title(f"Validation: True vs Predicted Pollution ({model_name})")
plt.legend()
plt.grid(True)
plt.tight_layout()

val_plot_path = os.path.join(
    OUTPUT_DIR,
    f"{MODEL_TAG}_validation_true_vs_predicted.png"
)
plt.savefig(val_plot_path, dpi=300)
plt.show()

print("Saved validation prediction plot to:", val_plot_path)


# 13. 测试集预测

test_mae = None
test_rmse = None

if len(test_scaled) <= LOOK_BACK:
    print("\n测试集长度小于或等于LOOK_BACK，无法构造测试序列。")
else:
    X_test, y_test_scaled = create_sequences(
        test_scaled,
        LOOK_BACK,
        target_index
    )

    test_dataset = AirQualityDataset(
        X_test,
        y_test_scaled if test_has_target else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_pred_scaled_list = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            if test_has_target:
                X_batch, _ = batch
            else:
                X_batch = batch

            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)

            test_pred_scaled_list.extend(outputs.cpu().numpy())

    test_pred_scaled = np.array(test_pred_scaled_list)

    test_pred = inverse_transform_pollution(
        test_pred_scaled,
        scaler,
        target_index,
        len(features)
    )

    result_df = pd.DataFrame({
        "Predicted_Pollution": test_pred
    })

    if test_has_target:
        test_true = inverse_transform_pollution(
            y_test_scaled,
            scaler,
            target_index,
            len(features)
        )

        result_df["True_Pollution"] = test_true

        test_mae, test_rmse = evaluate_predictions(
            test_true,
            test_pred,
            "Test"
        )

        plt.figure(figsize=(10, 5))
        plt.plot(test_true, label="True Pollution")
        plt.plot(test_pred, label="Predicted Pollution")
        plt.xlabel("Time Step")
        plt.ylabel("Pollution / PM2.5")
        plt.title(f"Test: True vs Predicted Pollution ({model_name})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        test_plot_path = os.path.join(
            OUTPUT_DIR,
            f"{MODEL_TAG}_test_true_vs_predicted.png"
        )
        plt.savefig(test_plot_path, dpi=300)
        plt.show()

        print("Saved test prediction plot to:", test_plot_path)

    prediction_path = os.path.join(
        OUTPUT_DIR,
        f"{MODEL_TAG}_pollution_prediction_results.csv"
    )
    result_df.to_csv(
        prediction_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nPrediction results saved to:", prediction_path)


# 14. 保存评价指标

metrics_data = [
    {
        "Model": model_name,
        "Dataset": "Validation",
        "MAE": val_mae,
        "RMSE": val_rmse
    }
]

if test_mae is not None and test_rmse is not None:
    metrics_data.append(
        {
            "Model": model_name,
            "Dataset": "Test",
            "MAE": test_mae,
            "RMSE": test_rmse
        }
    )

metrics_df = pd.DataFrame(metrics_data)

metrics_path = os.path.join(
    OUTPUT_DIR,
    f"{MODEL_TAG}_metrics_results.csv"
)
metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

print("\nMetrics saved to:", metrics_path)
print(metrics_df)


# 15. 保存最终模型

final_model_path = os.path.join(
    OUTPUT_DIR,
    f"final_{MODEL_TAG}_pollution_model.pth"
)
torch.save(model.state_dict(), final_model_path)

print("Final model saved to:", final_model_path)


# 16. 输出实验说明

print("\nExperiment Summary:")
print(f"Model: {model_name}")
print(f"Look-back window: {LOOK_BACK} hours")
print(f"Input features: {features}")
print(f"Target: {target_col}")
print(f"Dropout: {DROPOUT}")
print(f"Weight decay: {WEIGHT_DECAY}")

if USE_ATTENTION_MODEL:
    print("Improvement method: Temporal attention mechanism based on LSTM outputs.")
else:
    print("Baseline method: Basic LSTM using the last hidden state.")
