from typing import Optional, Tuple

import einops
import jaxtyping
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig

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

refusal_dir = torch.load(model_name.replace("/", "_") + "_refusal_dir.pt")
refusal_dir = refusal_dir.to(torch.bfloat16)


def direction_ablation_hook(activation: jaxtyping.Float[torch.Tensor, "... d_act"],
                            direction: jaxtyping.Float[torch.Tensor, "d_act"]):
    proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
    return activation - proj


class AblationDecoderLayer(nn.Module):
    def __init__(self, original_layer):
        super(AblationDecoderLayer, self).__init__()
        self.original_layer = original_layer

    def forward(self, *args, **kwargs):
        hidden_states = args[0]
        ablated = direction_ablation_hook(hidden_states, refusal_dir.to(hidden_states.device)).to(hidden_states.device)
        args = (ablated,) + args[1:]
        return self.original_layer.forward(*args, **kwargs)


for idx in range(len(model.model.layers)):
    model.model.layers[idx] = AblationDecoderLayer(model.model.layers[idx])

# Test Inference
# streamer = TextStreamer(tokenizer)
# with open("harmful.txt", "r") as f:
#     harmful = f.readlines()

#     for prompt in harmful:
#         print('===')
#         print(prompt)
#         print('---')
#         conversation = []
#         conversation.append({"role": "user", "content": prompt})
#         toks = tokenizer.apply_chat_template(
#         	conversation=conversation,
#         	add_generation_prompt=True, return_tensors="pt")

#         gen = model.generate(toks.to(model.device), streamer=streamer, max_new_tokens=1024)

#         decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
#         conversation.append({"role": "assistant", "content": decoded})


def generate_answer(question):
    input_text = f"下面列出了一个问题. 请写出问题的答案.\n####问题:{question}\n####答案:"
    inputs = tokenizer(
        [input_text], 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=2048, use_cache=True)
    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return decoded_output.split('<|im_end|>')[0].strip()

print("请输入您的问题,输入'exit'退出:")
while True:
    user_input = input("> ")
    if user_input.lower() == 'exit':
        print("程序已退出。")
        break
    answer = generate_answer(user_input)
    print("---")
    print(answer)



'''
# Test Inference
prompt = "Tell me about the Tiananmen Square Massacre"
conversation=[]
conversation.append({"role": "user", "content": prompt})
toks = tokenizer.apply_chat_template(conversation=conversation, add_generation_prompt=True, return_tensors="pt")
gen = model.generate(toks.to(model.device), max_new_tokens=200)
decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
print(decoded)

### Doesn't work ofc
# model.save_pretrained("modified_model")               
# tokenizer.save_pretrained("modified_model")   
'''
