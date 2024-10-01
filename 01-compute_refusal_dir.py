import random
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

torch.inference_mode()

torch.set_default_device("cuda")

model_name = "Qwen2.5-0.5B-Instruct"  # 模型名称
modle_path = "/mnt/s/worklib/llm/models-st/Qwen/" + model_name  # 模型路径

model = AutoModelForCausalLM.from_pretrained(
    modle_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(modle_path, trust_remote_code=True)

instructions = 1000
layer_idx = int(len(model.model.layers) * 0.6)
pos = -1

print("Instruction count: " + str(instructions))
print("Layer index: " + str(layer_idx))

with open("harmful.txt", "r") as f:
    harmful = f.readlines()

with open("harmless.txt", "r") as f:
    harmless = f.readlines()

harmful_instructions = random.sample(harmful, len(harmful))
harmless_instructions = random.sample(harmless, instructions)

harmful_toks = [
    tokenizer.apply_chat_template(conversation=[{"role": "user", "content": insn}], add_generation_prompt=True,
                                  return_tensors="pt") for insn in harmful_instructions]
harmless_toks = [
    tokenizer.apply_chat_template(conversation=[{"role": "user", "content": insn}], add_generation_prompt=True,
                                  return_tensors="pt") for insn in harmless_instructions]

max_its = instructions * 2
bar = tqdm(total=max_its)


def generate(toks):
    bar.update(n=1)
    return model.generate(toks.to(model.device), use_cache=False, max_new_tokens=1, return_dict_in_generate=True,
                          output_hidden_states=True)


harmful_outputs = [generate(toks) for toks in harmful_toks]
harmless_outputs = [generate(toks) for toks in harmless_toks]

bar.close()

harmful_hidden = [output.hidden_states[0][layer_idx][:, pos, :] for output in harmful_outputs]
harmless_hidden = [output.hidden_states[0][layer_idx][:, pos, :] for output in harmless_outputs]

print(harmful_hidden)

harmful_mean = torch.stack(harmful_hidden).mean(dim=0)
harmless_mean = torch.stack(harmless_hidden).mean(dim=0)

print(harmful_mean)

refusal_dir = harmful_mean - harmless_mean
refusal_dir = refusal_dir / refusal_dir.norm()

print(refusal_dir)

torch.save(refusal_dir, model_name.replace("/", "_") + "_refusal_dir.pt")
