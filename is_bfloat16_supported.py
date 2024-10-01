import os

import torch


# def is_bfloat16_supported():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     try:
#         x = torch.randn(3, device=device, dtype=torch.bfloat16)
#         return True
#     except RuntimeError as e:
#         return False


# is_bfloat16_supported = is_bfloat16_supported()

# print("is_bfloat16_supported? ", is_bfloat16_supported)

model_name = "Qwen2-1.5B-Instruct-Abliterated_v1"  # 模型名称
modle_path = "model/" + model_name  # 模型路径
dataset_name = "yuwangdianti"  # 数据集名称
dataset_file = "/mnt/s/worklib/llm/tools/llama3-txt2json-dataset-maker/novel/" + dataset_name + "/dataset.json"  # 数据文件路径

trained_model_name = "adapter_model/" + f"{model_name}_{dataset_name}".replace(
    "/", "_"
)

os.makedirs(trained_model_name, exist_ok=True)