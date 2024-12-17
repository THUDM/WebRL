<div align="center">

# WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning

</div>

![image](./assets/webrl.png)

*Technique adopted in [AutoGLM](https://xiao9905.github.io/AutoGLM/), a series of Phone Use and Web Browser Use Foundation Agents*

<p align="center">
   📃 <a href="https://arxiv.org/abs/2411.02337" target="_blank"> Paper </a> | 🤗 <a href="https://huggingface.co/THUDM/webrl-glm-4-9b" target="_blank"> WebRL-GLM-4-9B </a> | <a href="https://huggingface.co/THUDM/webrl-llama-3.1-8b" target="_blank"> WebRL-LLaMA-3.1-8B </a> | <a href="https://www.modelscope.cn/collections/WebRL-77a3e54a2dde4b" target="_blank"> ModelScope </a>
</p>

***

WebRL, a self-evolving online curriculum learning framework designed for training web agents, targeting the WebArena environment. 

## 🚀 Quick Start

### Dependencies

First, create a conda environment and install all pip package requirements.

```bash
conda create -n webrl python==3.10
conda activate webrl

cd WebRL
pip install -e .
```

### Model checkpoints

#### Actor checkpoints

The WebRL-GLM-4-9B checkpoint was released here and we use it:

- [WebRL-GLM-4-9B checkpoint](https://huggingface.co/THUDM/webrl-glm-4-9b)
- [WebRL-Llama-3.1-8B checkpoint](https://huggingface.co/THUDM/webrl-llama-3.1-8b)
- [WebRL-Llama-3.1-70B checkpoint](https://huggingface.co/THUDM/webrl-llama-3.1-70b)

#### ORM checkpoint

The checkpoint for Outcome-supervised Reward Model (ORM) is as follow:

- [ORM-Llama-3.1-8B checkpoint](https://huggingface.co/THUDM/webrl-orm-llama-3.1-8b/tree/main)



### ✈️ Train SFT model

We use LLaMA-Factory to train the SFT baseline, which is the starting model for WebRL. We release the code and data used for training. You can train the SFT baseline with the following commands:

```bash
cd LLaMA-Factory
bash run.sh examples/train_full/llama3_full_policy_web.yaml
```

### ✈️ Train WebRL

After training the SFT baseline, you should use it as the initial model of the actor and critic.  You can train WebRL with the following commands:

```bash
bash run_multinode.sh
```

This command is used to train the actor and critic in each phase.

### 💡 Generating New Instructions

You can generate new instructions with the following commands:

```bash
python scripts/gen_task.py
```

### 🛜 Interaction and Evaluation

The instruction and script for interaction with WebArena is provided in [VAB-WebArena-Lite](https://github.com/THUDM/VisualAgentBench/tree/main/VAB-WebArena-Lite).
You can implement the interaction process of WebRL according to the [``Evaluating in WebRL Setting (Text Modal)``](https://github.com/THUDM/VisualAgentBench/tree/main/VAB-WebArena-Lite#-evaluating-in-webrl-setting-text-modal) section of VAB-WebArena-Lite.


To enable interaction with WebArena, you need to configure each task in the same format as the sample test case provided in the ``test_webarena_lite.raw.json`` file in VAB-WebArena-Lite. Below is the template for a task configuration:

```python
{
  
  "sites": [
    <site> # possible choices: "shopping_admin", "map", "shopping", "reddit", "gitlab"
  ],
  "task_id": <Your task id>
  "require_login": true,
  "storage_state": "./.auth/shopping_admin_state.json",
  "start_url": <start url of site>, # possible choices: "__SHOPPING_ADMIN__", "__SHOPPING__", "__GITLAB__", "__MAP__", "__REDDIT__"
  "geolocation": null,
  "intent_template": "",
  "instantiation_dict": {},
  "intent": <Task>,
  "require_reset": false,
  "eval": {
    "eval_types": [
      "string_match"
    ],
    "reference_answers": {
      "exact_match": "N/A"
    },
    "reference_url": "",
    "program_html": [],
    "string_note": "",
    "reference_answer_raw_annotation": ""
  },
  "intent_template_id": 0
}
```

After configuring the tasks, use the script ``scripts/generate_test_data.py`` to generate the configuration files. Make sure to modify the data path in the script to point to the JSON file containing your configured interaction cases.

After interaction finished, run ``scripts/process_data.py`` to process the interaction trajectories.

```bash
python scripts/process_data.py \
  --stage 1 2 \
  --add_reward \
  --rollout_path <directory_of_interaction_trajectories> \
  --experience_paths "path1", "path2" \ 
  --orm_path <path_to_ORM_model> \
  --actor_path <path_to_actor_model_for_computing_perplexity> \
  --output_path <path_to_output_file>
```
- `stage`: Specifies the processing method for the data
  - 1: Convert rollout trajectories into the required format.
  - 2: Incorporate historical experiences filtered by perplexity.
- `add_reward`: Apply ORM to label each trajectory.
- `output_path`: The file containing processed interaction trajectories, ready for direct use in training.
  - stage 1: Processed interaction trajectories will be saved in this file. Contains data without historical experiences.
  - stage 2: An additional file, output_path + '_filter', will also be generated.
    - output_path: Contain data without historical experiences.
    - output_path + '_filter': Contain data with historical experiences.
- `rollout_path`: Path to the `traces` subfolder containing initial interaction trajectories, typically generated after running Webarena-Lite.
- `experience_paths`: List of file paths to processed interaction data (`output_path`) from previous phases.

Both output_path and output_path + '_filter' are formatted for direct use in subsequent training.

## Citation
```
@artical{qi2024webrl,
      title={WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning}, 
      author={Qi, Zehan and Liu, Xiao and Iong, Iat Long and Lai, Hanyu and Sun, Xueqiao and Yang, Xinyue and Sun, Jiadai and Yang, Yu and Yao, Shuntian and Zhang, Tianjie and others},
      journal={arXiv preprint arXiv:2411.02337},
      year={2024},
}
```
