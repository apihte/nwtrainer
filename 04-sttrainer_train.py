# 加载模型和分词器
import os

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from local_dataset import LocalJsonDataset
from safetensors.torch import load_model, save_model

max_seq_length = 2048
dtype = None
load_in_4bit = False

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载预训练模型和tokenizer
model_name = "Qwen2-1.5B-Instruct-Abliterated_v1"  # 模型名称
modle_path = "Qwen/" + model_name  # 模型路径
dataset_name = "yuwangdianti"  # 数据集名称
dataset_file = "/mnt/s/worklib/llm/tools/llama3-txt2json-dataset-maker/novel/" + dataset_name + "/dataset.json"  # 数据文件路径

trained_model_name = "adapter_model/" + f"{model_name}_{dataset_name}".replace(
    "/", "_"
)

os.makedirs(trained_model_name, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(modle_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(modle_path)

# 加载和预处理数据集
custom_dataset = LocalJsonDataset(json_file=dataset_file, tokenizer=tokenizer, max_seq_length=max_seq_length)
dataset = custom_dataset.get_dataset()


def is_bfloat16_supported():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        x = torch.randn(3, device=device, dtype=torch.bfloat16)
        return True
    except RuntimeError as e:
        return False


is_bfloat16_supported = is_bfloat16_supported()


# 设置训练配置
from trl import SFTConfig, SFTTrainer

train_config = SFTConfig(
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    warmup_steps=20,
    max_steps=2000,
    learning_rate=3e-3,
    fp16=not is_bfloat16_supported,
    bf16=is_bfloat16_supported,
    logging_steps=1,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    save_strategy="no"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=train_config
)


# 训练模型
trainer.train()

model.save_pretrained(trained_model_name)
tokenizer.save_pretrained(trained_model_name)
