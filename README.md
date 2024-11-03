<div align="center">

# WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning

</div>

![image](https://github.com/user-attachments/assets/84ed9842-ec01-4625-b2fa-25c9c2e69c84)

***

## 🚀 Quick Start

### Dependencies

First, create a conda environment and install all pip package requirements.

```bash
conda create -n webrl python==3.10
conda activate webrl

cd WebRL
pip install -e .
```

### Train SFT model

We use LLaMA-Factory to train the SFT baseline, which is the starting model for WebRL. You can train the SFT baseline with the following commands:

```bash
cd LLaMA-Factory
bash run.sh examples/train_full/llama3_full_policy_web.yaml
```

### Train ORM

We use LLaMA-Factory to train the ORM. You can train the ORM with the following commands:

```bash
cd LLaMA-Factory
bash run.sh examples/train_full/llama3_full_orm_web.yaml
```

### Train WebRL

After training the SFT baseline, you should use it as the initial model of the actor and critic.  You can train WebRL with the following commands:

```bash
bash run_multinode.sh
```


This command is used to train the actor and critic in each phase.

### Generating New Instructions

### Interaction

### Evaluation
