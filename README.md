<div align="center">

# WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning

</div>

![image](./assets/webrl.png)

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

### Model checkpoint

We will release the checkpoint of ORM, Llama3.1-8B trained with WebRL, and Llama3.1-70B trained with WebRL after the rebuttal period, due to the need for anonymity.


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

Please see details in [interaction](./interaction).