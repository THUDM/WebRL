import ast
import numpy as np
from transformers import AutoTokenizer
import jsonlines
from vllm import LLM, SamplingParams
import argparse
import os
# Set environment variables
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
import time
from tqdm import tqdm
import json
import torch
import random

def compare_dicts(dict1, dict2):
    if isinstance(dict1, dict) and isinstance(dict2, dict):
        if dict1.keys() != dict2.keys():
            return False
        for key in dict1:
            if key == 'message':
                continue
            if not compare_dicts(dict1[key], dict2[key]):
                return False
        return True
    else:
        return dict1 == dict2
    return False

def remove_comments(code):
    # 按行分割代码
    for key in ['exit(','do(','go_backward(']:
        if key in code:
            return key + code.split(key)[-1]
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('#'):
            # 跳过注释行
            continue
        else:
            # 返回非注释行及其后面的部分
            return '\n'.join(lines[i:])
    return ''

def parse_function_call(expression):
    expression = remove_comments(expression)
    # 将字符串解析为 AST
    expression = expression.strip()  # 清除两边的空白
    tree = ast.parse(expression, mode='eval')

    # 提取函数名称
    func_call = tree.body
    if not isinstance(func_call, ast.Call):
        return {
            "operation": expression,
        }

    func_name = func_call.func.id
    result = {
        "operation": func_name,
    }

    # 提取参数
    args = func_call.args
    kwargs = func_call.keywords

    for kw in kwargs:
        if func_name == "do" and kw.arg == "action":
            result["action"] = ast.literal_eval(kw.value)
        # elif func_name == "do" and kw.arg == "argument":
        #     result["argument"] = ast.literal_eval(kw.value)
        else:
            if "kwargs" not in result:
                result["kwargs"] = {}
            if kw.arg == "element":
                try:
                    # 解析元素的内部函数
                    inner_func = kw.value
                    if isinstance(inner_func, ast.Call) and inner_func.func.id == 'find_element_by_instruction':
                        for inner_kw in inner_func.keywords:
                            if inner_kw.arg == "instruction":
                                result["kwargs"]["instruction"] = ast.literal_eval(inner_kw.value)
                    else:
                        result["kwargs"][kw.arg] = ast.literal_eval(inner_func)
                except Exception:
                    result["kwargs"][kw.arg] = ast.literal_eval(kw.value)
            else:
                result["kwargs"][kw.arg] = ast.literal_eval(kw.value)

    return result


model_path = "/rl/web_policy/llama3-8b/sft_template_0_7_wo_selector_0_3"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# with open('/rl/web_train_data/planner-policy-v1.0-function-wo-pic/real_template_success_test.json') as fp:
#     data = json.load(fp)
# targets = []
# inputs = []
# for item in data:
#     inputs.append(item['conversations'][0]['value'])
#     targets.append(item['conversations'][1]['value'])
    

# data = torch.load('/workspace/qzh/digirl-master/logs/webarena_test_offline.pt')
# inputs = []
# targets = []
# for trace in data:
#     for item in trace:
#         inputs.append(item['observation'])
#         targets.append(item['action'])

train_items_chosen = torch.load('/rl/web_train_data/digirl/webarena_template.pt')
webs_idx = {'reddit': [0, 1838],
 'map': [1838, 3088],
 'oss': [3088, 4303],
 'cms': [4303, 5466],
 'gitlab': [5466, 6258]}

with open('/workspace/qzh/Pipeline/config_files/train_tasks_rm.json') as fp:
    content = json.load(fp)
task_rm = {}
for item in content:
    task = item['intent']
    if task not in task_rm:
        task_rm[task] = True
with open('/workspace/qzh/Pipeline/config_files/train_tasks_wo_selector.json') as fp:
    content = json.load(fp)
task_wo_selector = {}
for item in content:
    task = item['intent']
    if task not in task_wo_selector:
        task_wo_selector[task] = True
with open('/workspace/qzh/filtered_0_7.json') as fp:
    save_task = json.load(fp)
task_0_7 = {}
for task in save_task:
    if task not in task_0_7:
        task_0_7[task] = True
print(len(task_0_7), len(task_rm), len(task_wo_selector))
stats = {}
for web in webs_idx:
    idx = webs_idx[web]
    stats[web] = {'0_7': [], '0_3_wo_selector': [], '0_3_rm': []}
    for item in train_items_chosen[idx[0]:idx[1]]:
        task = item[0]['task']
        if task in task_0_7:
            stats[web]['0_7'].append(item)
        else:
            if task in task_wo_selector:
                stats[web]['0_3_wo_selector'].append(item)
            elif task in task_rm:
                stats[web]['0_3_rm'].append(item)
                    
llm = LLM(model=model_path, trust_remote_code=True, gpu_memory_utilization=0.95, tensor_parallel_size=8)
print(tokenizer.eos_token)
sampling_params = SamplingParams(temperature=0, max_tokens=8192, stop=["<|endoftext|>","<|im_end|>","<|end|>", "<|eot_id|>",tokenizer.eos_token])

for web in stats:
    keys = ['0_7', '0_3_wo_selector']
    info = stats[web]
    for key in keys:
        inputs = []
        targets = []
        items = info[key]
        random.seed(42)
        random.shuffle(items)
        for trace in items[:100]:
            for item in trace:
                inputs.append(item['observation'])
                targets.append(item['action'])
        outputs = llm.generate(inputs, sampling_params)
        generated_text = [output.outputs[0].text for output in outputs]
        correct_num = 0
        wrong_num = 0
        for i, (target, text) in enumerate(zip(targets, generated_text)):
            try:
                target = parse_function_call(target)
                text = parse_function_call(text)
                if compare_dicts(text, target):
                    correct_num += 1
                else:
                    wrong_num += 1
            except:
                wrong_num += 1
        print(web, key, correct_num, wrong_num, correct_num / (correct_num + wrong_num))