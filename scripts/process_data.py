import json
from tqdm import tqdm
import os
import glob
import re
import random
import uuid
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import signal
import time
import requests
import numpy as np
from accelerate import init_empty_weights, infer_auto_device_map

def save_jsonl(content, path, mode='w'):
    with open(path, mode) as f:
        for i, line in enumerate(content):
            if i == len(content) - 1:
                f.write(json.dumps(line))
            else:
                f.write(json.dumps(line) + "\n")
                
def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

import torch.nn as nn

class LlamaAgent():
    def __init__(self, device, accelerator=None, policy_lm = "gpt2", 
                cache_dir = '~/.cache', dropout = 0.5, TEMPLATE = None, use_lora=False,
                do_sample = True, temperature = 1.0, max_new_tokens = 32, use_bfloat16 = False, eos_str = None):
        if use_bfloat16:
            self.model = AutoModelForCausalLM.from_pretrained(policy_lm, use_cache=False,
                                                              torch_dtype = torch.bfloat16)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(policy_lm, use_cache=False, torch_dtype = torch.bfloat16, trust_remote_code=True)
        
        self.template = TEMPLATE
        self.policy_lm = policy_lm
        self.target_critic = None
        self.tokenizer = AutoTokenizer.from_pretrained(policy_lm, trust_remote_code=True, use_cache=False)
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.padding_side = 'right'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.device = device
        self.dropout = torch.nn.Dropout(p=dropout)
        self.softmax = torch.nn.Softmax(dim= -1)
        self.do_sample = do_sample
        self.temperature = temperature
        self.accelerator = accelerator
        self.max_new_tokens = max_new_tokens
        self.eos_str = eos_str
    
    def func(self, response):
        if 'do(' in response:
            action_idx = response.index('do(')
        elif 'go_backward' in response:
            action_idx = response.index('go_backward')
        elif 'scroll' in response:
            action_idx = response.index('scroll')
        elif 'login' in response:
            action_idx = response.index('login')
        elif 'exit' in response:
            action_idx = response.index('exit')
        else:
            action_idx = 0
        if action_idx != 0:
            action_idx = len(self.tokenizer.tokenize(response[:action_idx]))
        return action_idx

    def get_log_prob(self, observation, action):
        device = self.device
        input_texts = [obs + act for obs, act in zip(observation, action)]
        input_ids = self.tokenizer(input_texts, return_tensors='pt', padding=True, max_length=16384, truncation = True).to(device)
        
        obs_lengths = [len(self.tokenizer(obs, return_tensors='pt')['input_ids'][0]) for obs in observation]
        
        outputs = self.model(labels = input_ids["input_ids"],
                            **input_ids)
        log_probs = outputs.logits.log_softmax(dim=-1)
        
        action_tokenized = self.tokenizer(action, return_tensors='pt', padding=True, truncation=True, add_special_tokens=False).to(device)
        action_ids_padded = action_tokenized.input_ids
        action_attention_mask = action_tokenized.attention_mask

        max_action_length = action_ids_padded.shape[1]
        batch_size = len(action)
        action_log_probs = torch.zeros((batch_size, max_action_length)).to(device)

        for i in range(batch_size):
            action_start_idx = obs_lengths[i] - 1
            action_end_idx = action_start_idx + action_attention_mask[i].sum().item()  
            action_log_probs[i, :action_end_idx-action_start_idx] = log_probs[i, action_start_idx:action_end_idx, :].gather(
                1, action_ids_padded[i, :action_end_idx-action_start_idx].unsqueeze(-1)
            ).squeeze(-1)

        log_prob = action_log_probs * action_attention_mask
        res_log_prob = []
        for i in range(len(action)):
            # action_idx = self.func(action[i])
            action_idx = 0
            action_log_prob = log_prob[i, action_idx:]
            res_log_prob.append(action_log_prob)
        res_log_prob = torch.stack(res_log_prob)
        # log_prob = log_prob.mean(dim = 1).flatten()
        return res_log_prob

def add_mc_return(trajectory, gamma = 0.9):
    """
    add trajectory reward to the dict of each interaction
    """
    trajectory_rewards = np.array([d["reward"] for d in trajectory]).reshape(1, -1)
    gamma_row = np.cumprod(np.ones((1, trajectory_rewards.shape[1]))*gamma)
    gamma_matrix = np.triu(gamma_row.reshape(1, -1 )/ gamma_row.reshape(-1, 1))
    mc_returns = np.sum(trajectory_rewards*gamma_matrix, axis = 1)
    for d, mc in zip(trajectory, mc_returns):
        d.update({"mc_return": mc})
    return trajectory

def check_files_with_prefix(directory, prefix):
    search_pattern = os.path.join(directory, prefix + '.*')
    files = glob.glob(search_pattern)
    return len(files) > 0

def trace_process(dir_path):
    traces_paths = os.listdir(os.path.join(dir_path, 'traces'))
    os.makedirs(os.path.join(dir_path, 'fixed_traces'), exist_ok=True)
    print(len(traces_paths))
    num = 0
    for trace in tqdm(traces_paths, ascii=True):
        if trace.endswith('.jsonl') == False:
            continue
        trace_id = trace.split('.jsonl')[0]
        if check_files_with_prefix(os.path.join(dir_path, 'fixed_traces'), trace_id):
            continue
        action_path = os.path.join(dir_path, 'actions', trace_id + '.json')
        with open(action_path, 'r') as f:
            action = json.load(f)
        score = action['score']
        trace_content = read_jsonl(os.path.join(dir_path, 'traces', trace))
        flag = 0
        for i, step in enumerate(trace_content):
            response = step['response']
            step['fixed_response'] = response
            step['score'] = score
        if flag == 1:
            continue
        save_jsonl(trace_content, os.path.join(dir_path, 'fixed_traces', trace), mode='w')
        num += 1
    print(num)

def build_rm_data(dir_path, ouput_path, add_reward=False, model_path=None):
    if add_reward and model_path is None:
        raise ValueError('Please provide the model path for reward prediction.')
    
    if add_reward:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_path,padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                    return_dict=True,
                                                    device_map='auto',
                                                    low_cpu_mem_usage=True) 
        print('Loading finished...')
    
    def rm_format(content):
        if len(content) == 0:
            return None
        task = content[0]['target']
        last_html = content[-1]['html']
        instruction = 'The User Intent:\n' + task.strip() + '\n\n' + 'Action History:\n'
        for item in content:
            if 'fixed_response' not in item:
                return None
            response = item['fixed_response']
            response_elements = response.split('\n')
            new_response = '; '.join(response_elements)
            instruction += new_response.strip() + '\n'
        instruction += '\n' + 'The Current Screenshot:\n' + last_html.strip()
        return instruction
    
    files = os.listdir(os.path.join(dir_path, 'fixed_traces'))
    system = """You are an expert in evaluating the performance of a website navigation agent. The agent is designed to help a human user navigate the website to complete a task. Given the user’s intent, the agent's action history, and the final state of the screen, your goal is to decide whether the agent’s execution is successful or not. You must respond with YES or NO."""
    new_content = []
    num_YES = 0
    num_NO = 0
    for file in tqdm(files, ascii=True):
        file_path = os.path.join(dir_path, 'fixed_traces', file)
        file_content = read_jsonl(file_path)
        instruction = rm_format(file_content)
        if instruction is None:
            continue
        # print(instruction)
        score = file_content[-1]['score']
        label = 'YES' if score > 0.5 else 'NO'
        
        if label == "YES":
            num_YES += 1
        else:
            num_NO += 1
        
        new_item = {
                'conversations': [
                    {
                        "from": "human",
                        "value": instruction
                    },
                    {
                        "from": "gpt",
                        "value": label
                    }
                ],
                "system": system
            }
        
        if add_reward:
            reward = add_overall_reward(new_item, model, tokenizer)
            for file_content_item in file_content:
                file_content_item['score'] = reward
            save_jsonl(file_content, file_path)
            label = 'YES' if reward >= 0.5 else 'NO'
            new_item['conversations'][1]['value'] = label
            
        new_content.append(new_item)
    print(num_YES / (num_YES + num_NO))
    if ouput_path is not None:
        with open(ouput_path, 'w') as fp:
            json.dump(new_content, fp, indent=4)

def build_policy_data(dir_path, ouput_path):
    def format_history(contents, index):
            history = ""
            if index == 0:
                return history
            for i in range(index - 1, -1, -1):
                history = f"Round {i}\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n{contents[i]['prompt']}\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{contents[i]['fixed_response']}\n\n" + history
            return history

    def format_prompt(instruction, index, html_text, contents):
        history = format_history(contents, index)
        if len(history) + len(html_text) > (16384 - 512):
            html_text = html_text[:(16384 - 512)-len(history)]
        current_turn = f"Round {index}\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n{html_text}\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        prompt = f"Task Instruction: {instruction}\n\n{history}{current_turn}"
        return prompt

    def template(all_trajectories):
        for traj in all_trajectories:
            for i, step in enumerate(traj):
                instruction = step['task']
                index = i
                html_text = step['observation'][-1]['html']
                contents = step['observation']
                step['observation'] = format_prompt(instruction, index, html_text, contents)
                step['next_observation'] = format_prompt(instruction, index + 1, step['next_observation']['html'], contents + [step['next_observation']])
        return all_trajectories

    traces_dir = 'fixed_traces'
    traces = os.listdir(os.path.join(dir_path, traces_dir))
    data = []
    for trace in tqdm(traces):
        if trace.endswith('.jsonl') == False:
            continue
        trace_content = read_jsonl(os.path.join(dir_path, traces_dir, trace))
        try:
            target = trace_content[-1]['target']
            label = 1 if trace_content[-1]['score'] >= 0.5 else 0
        except:
            continue
        
        new_trace = []
        for i, item in enumerate(trace_content):
            if 'fixed_response' not in item:
                new_trace = []
                break
            new_item = {
                'observation': trace_content[:i + 1],
                'next_observation': trace_content[i + 1] if i != len(trace_content) - 1 else trace_content[i],
                'task': target, 
                'reward': trace_content[-1]['score'], 
                'done': False if i != len(trace_content) - 1 else True, 
                'action': item['fixed_response'], 
                'trajectory_reward': label
            }
            new_trace.append(new_item)
        if len(new_trace) == 0:
            continue
        data.append(new_trace)
    data = template(data)
    torch.save(data, ouput_path)

def add_overall_reward(item, model, tokenizer):
    yes_token_id = tokenizer.convert_tokens_to_ids("YES")
    no_token_id = tokenizer.convert_tokens_to_ids("NO")
    # print(f"Yes token ID: {yes_token_id}. No token ID: {no_token_id}")

    system = item['system']
    instruction = item['conversations'][0]['value']
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    prompt_ids = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False)
    model.eval()
    with torch.no_grad():
        input_ids = prompt_ids[:, :16384].to(model.device)
        outputs = model(input_ids)
        outlogits = outputs.logits
        logits = outlogits[:, -1, :]
        max_logits = torch.max(logits[:, [yes_token_id, no_token_id]], dim=-1, keepdim=True).values
        stabilized_logits = logits[:, [yes_token_id, no_token_id]]
        prob_yes_stabilized = stabilized_logits[:, 0]
        prob_no_stabilized = stabilized_logits[:, 1]
        scores = torch.exp(prob_yes_stabilized) / (torch.exp(prob_yes_stabilized) + torch.exp(prob_no_stabilized))
        if scores[0] >= 0.5:
            score = 1
        else:
            score = 0
    return score

import multiprocessing as mp
from tqdm import tqdm

def process_data_chunk(data_chunk, actor_path, device_ids, result_queue):
    torch.cuda.empty_cache()
    # Assign GPUs for actor 
    actor_device = f'cuda:{device_ids[0]}'
    agent = LlamaAgent(device=actor_device, policy_lm=actor_path)
    agent.model = agent.model.to(actor_device)

    success_item = []
    fail_item = []
    success_tasks = {}

    file_content = [add_mc_return(t, gamma=0.9) for t in data_chunk if len(t) != 0]
    for traj in file_content:
        trajectory_reward = traj[0]['trajectory_reward']
        task = traj[0]['task']
        if trajectory_reward >= 0.5:
            if task in success_tasks:
                continue
            for item in traj:
                success_item.append(item)
            success_tasks[task] = True
        else:
            for item in traj:
                fail_item.append(item)

    success_scores = []
    with torch.no_grad():
        for item in tqdm(success_item, ascii=True):
            action_log_p = agent.get_log_prob([item['observation']], [item['action']])
            num_tokens = action_log_p.shape[1]
            action_log_p = action_log_p.sum(dim=1)
            action_p = torch.exp(action_log_p) ** (1 / num_tokens)
            success_scores.append(action_p)
            item['rank_score'] = action_p

    # all_items = success_item
    all_items = success_item + fail_item 
    output_file = f'{device_ids[0]}.pt'
    torch.save(all_items, output_file)

def replay_data(files, actor_path, num_processes=8):
    file_contents = []
    for file in files:
        file_content = torch.load(file)
        for traj in file_content:
            if isinstance(traj, list) == True:
                file_contents.append(traj)
            else:
                print(file)
    file_chunks = [file_contents[i::num_processes] for i in range(num_processes)]
    print(len(file_chunks))
    for chunk in file_chunks:
        print(len(chunk))
    # Create a multiprocessing queue to gather results
    result_queue = mp.Queue()
    # List to hold process objects
    processes = []
    # Assign different GPU pairs to each process
    gpu_pairs = [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)]
    # Start processes
    for i in range(num_processes):
        p = mp.Process(target=process_data_chunk, args=(file_chunks[i], actor_path, gpu_pairs[i], result_queue))
        p.start()
        processes.append(p)
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    all_data = []
    for i in range(8):
        output_file = f'{i}.pt'
        data_chunk = torch.load(output_file)
        all_data.extend(data_chunk)
    success_items = []
    fail_items = []
    for item in all_data:
        if item['trajectory_reward'] >= 0.5:
            if isinstance(item['rank_score'], torch.Tensor) == True:
                item['rank_score'] = item['rank_score'].cpu().item()
            success_items.append(item)
        else:
            fail_items.append(item)
    print(len(success_items), len(fail_items))
    
    # Load current data and add MC return
    current_data = torch.load(args.output_path)
    current_data = [add_mc_return(t, gamma=0.9) for t in current_data]
    current_data_items = []
    for traj in current_data:
        for item in traj:
            current_data_items.append(item)
    
    random.shuffle(success_items)
    success_items_filer= []
    num = 2 * len(current_data_items)
    for item in success_items:
        if item['rank_score'] <= 0.95:
            if item['rank_score'] <= 0.5:
                continue
            success_items_filer.append(item)
    real_num = min(len(success_items_filer), num)
    all_item_filter = success_items_filer[:real_num]
    print(len(all_item_filter))
    
    # Combine filtered data with current data
    all_data = all_item_filter + current_data_items
    # Save the combined data
    random.shuffle(all_data)
    print(args.output_path.replace('.pt', '_filter.pt'))
    torch.save(all_data, args.output_path.replace('.pt', '_filter.pt'))

    for i in range(8):
        output_file = f'{i}.pt'
        os.remove(output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process webarena data')
    parser.add_argument('--rollout_path', type=str, default='')
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--experience_paths', type=str, nargs="+", default=None)
    parser.add_argument('--add_reward', action='store_const', const=True, default=False)
    parser.add_argument('--orm_path', type=str, default='', help='Path to the trained reward model')
    parser.add_argument('--actor_path', type=str, default='', help='Path to the actor')
    parser.add_argument('--stage', type=str, nargs="+", default=None, help='process stage: 1. trace process 2. rm data process 3. policy data process')
    args = parser.parse_args()
    print(args.stage)

    if '1' in args.stage:
        trace_process(args.rollout_path)
        build_rm_data(args.rollout_path, None, args.add_reward, args.orm_path)
        build_policy_data(args.rollout_path, args.output_path)
    if '2' in args.stage:
        replay_data(args.experience_paths, args.actor_path)