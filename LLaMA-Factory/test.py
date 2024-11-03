import requests
import json
import random
random.seed(42)
# 设置请求的URL
url = "http://localhost:8000/v1/chat/completions"

# 请求头，指定内容类型为 JSON
headers = {
    "Content-Type": "application/json",
}

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
    # 请求体数据
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": inputs[i],
            }
        ],
        "do_sample": True,
        "temperature": 0,
        "top_p": 0,
        "n": 1,
        "max_tokens": 0,
        "stop": "<|eot_id|>",
        "stream": False
    }

    # 发送POST请求
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # 检查请求是否成功
    if response.status_code == 200:
        # 解析返回的JSON数据
        data = response.json()
        print(data['choices'][0]['message']['content'])
        print('----------------------------')
        print(targets[i])
        print('===' * 20)
    else:
        print(f"请求失败，状态码: {response.status_code}")
