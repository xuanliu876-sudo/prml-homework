import os

# 必须放在所有 Hugging Face 相关导入之前
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset

cache_dir = r"D:\Multi30k_Transformer\hf_cache"
save_dir = r"D:\Multi30k_Transformer\multi30k_data"

print("开始下载Multi30k数据集...")
print("HF_ENDPOINT =", os.environ.get("HF_ENDPOINT"))

dataset = load_dataset(
    "bentrevett/multi30k",
    cache_dir=cache_dir
)

print(dataset)
print("示例数据：")
print(dataset["train"][0])

dataset.save_to_disk(save_dir)

print(f"下载并保存完成：{save_dir}")