from openai import OpenAI
from transformers import AutoModelForCausalLM
# model_path = '/rl/web_train_data/digirl/web_policy_sft_lite_turn0_10512_10841/actor'
# model = AutoModelForCausalLM.from_pretrained(model_path)
import random
random.seed(42)
import json
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)
models = client.models.list()
# print(models)
import torch
data = torch.load('/workspace/qzh/web_policy_sft_lite_turn0_10000_10512/web_policy_sft_lite_turn0_10000_10512.pt')
# with open('/rl/web_train_data/planner-policy-v1.0-function-wo-pic/template_lite_thought.json') as fp:
#     data = json.load(fp)
targets = []
inputs = []
# for item in data:
#     inputs.append(item['conversations'][0]['value'])
#     targets.append(item['conversations'][1]['value'])
for traj in data:
    for item in traj:
        inputs.append(item['observation'])
        targets.append(item['action'])

idx = 38
# print([inputs[idx]])
# print('-=-' * 20)
print([targets[idx]])
print('-=-' * 20)
idxs = random.sample([i for i in range(len(targets))], 50)
print(idxs)
for i in idxs:
    completion = client.completions.create(
    model="gpt-3.5-turbo",
    prompt=inputs[i],
    max_tokens=512,
    temperature=0,
    top_p=1,
    stop='<|eot_id|>'
    )
    print(completion.choices[0].text)
    print('----------------------------')
    print(targets[i])
    print('===' * 20)

# import openai
# openai.base_url="http://localhost:8000/v1"
# openai.api_key="EMPTY"
# completion = openai.ChatCompletion.create(
#   model="/rl/web_policy/llama3-8b/sft",
#   messages=[{'role': 'user', 'content': 'hi'}],
#   max_tokens=512
# )

