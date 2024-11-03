import json
import torch
import textwrap
from openai import OpenAI
import random
import re
from webrl.models import VLMDoubleCritic

random.seed(42)

client = OpenAI(
    base_url = "",
    api_key = ""
)

PROMPT = """You are a smart task creator for a website intelligent assistant. Your goal is to generate clear and practical tasks that the assistant can assist people with when they use {web} in their daily lives. These tasks should encompass a wide range of possible instructions and questions that may arise when using {web} website.

Your need to draw inspiration from the #Given Task# to create new tasks. These new tasks should belong to the same domain as the #Given Task# but be more diverse. The difficulty level of the #Created Task# should be similar to that of the #Given Task#. The #Created Task# must be reasonable, understandable and realistic. ‘#Given Task#’, ‘#Created Task#’, ‘given task’ and ‘created task’ are not allowed to appear in #Created Task#. 

**Guidelines:**
- **Format each task** clearly using backticks (`) for each command description.
- Use a variety of phrasing styles to avoid repetitive expressions.
- Use variable names that match those in the provided task examples, such as place names, usernames, and product names. Avoid inventing entirely new variable names.
- Maintain the same or similar difficulty level as the #Given Task#. Tasks can be slightly more or less challenging but should stay within a reasonable range.

#Given Task#
{task_examples}

#Created Task#
"""

def call_gpt(model='gpt-3.5-turbo', temperature=0, top_p=0, prompt=''):
    response = client.chat.completions.create(
                model=model,
                messages= [{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p
            )
    response=response.choices[0].message.content
    return response

def filter_by_critic(tasks, threshold=[0.05, 0.75]):
    critic_lm = 'path to initial model like llama3.1'
    critic_resume_path = 'path to critic parameters'
    
    device = 'cuda:0'
    critic = VLMDoubleCritic(device=device, accelerator=None, critic_lm=critic_lm, cache_dir=None, in_dim=4096, out_dim=2)
    torch.cuda.empty_cache()
    state_dict = torch.load(critic_resume_path, map_location='cpu')
    critic.load_state_dict(state_dict)
    critic = critic.to(device)
    softmax = torch.nn.Softmax(dim = -1)
    
    htmls = json.load(open('/path_to/htmls.json'))[0]
    
    filtered_tasks = []
    with torch.no_grad():
        for item in tasks:
            task = item['task']
            web = item['web']
            html = htmls[web]
            # html = '** Simplified html **'
            observation = f"Task Instruction: {task}\n\nRound 0\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n{html}\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            v1, v2 = critic([observation], '')
            v1 = softmax(v1)[:, 1][0]
            v2 = softmax(v2)[:, 1][0]
            if threshold[0] <= v1 and threshold[1] >= v1 or threshold[0] <= v2 and threshold[1] >= v2:
                filtered_tasks.append(item)
    return filtered_tasks

if __name__ == '__main__':
    path = "path of file that contains failed instructions"
    path_to_save = 'path of file that store generated instructions'
    
    with open(path) as fp:
        failed_tasks = [json.loads(line) for line in fp]
    
    grouped_tasks = {}
    for item in failed_tasks:
        web = item['web']
        task = item['task']
        if web not in grouped_tasks:
            grouped_tasks[web] = []
        grouped_tasks[web].append(task)
    
    webs = list(grouped_tasks.keys())
    for web in webs:
        seed_num = 10
        generation_turns = 2
        
        tasks = grouped_tasks[web]
        num = min(seed_num, len(tasks))
        for turn in range(generation_turns):
            seed_tasks = random.sample(tasks, num)
            # print(seed_tasks)
            
            task_examples = ""
            for task in seed_tasks:
                task_examples += f'- `{task.strip()}`\n'
            task_examples = task_examples.strip()
            # print(task_examples)
            
            prompt = PROMPT.format(web=web, task_examples=task_examples)
            response = call_gpt(model='gpt-4o-2024-05-13', prompt=prompt, temperature=1, top_p=1)
            extracted_content = re.findall(r'`([^`]+)`', response)
            
            generated_tasks = []
            for task in extracted_content:
                new_item = {
                    "task": task,
                    "web": web
                }
                generated_tasks.append(new_item)
                
            with open(path_to_save, 'a') as fp:
                for task in generated_tasks:
                    fp.write(json.dumps(task) + '\n')