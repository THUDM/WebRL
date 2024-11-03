from webrl.data import ReplayBuffer
import numpy as np
from tqdm import tqdm
from webrl.algorithms.webrl import WebRLTrainer
from webrl.misc import colorful_print
from torch.utils.data import DataLoader
from webrl.data import DummyDataset
import wandb
import os
import torch
import time
import copy
from webrl.environment.env_utils import add_mc_return


def offpolicy_train_loop(agent,
                tokenizer,
                accelerator,
                batch_size: int = 2,
                capacity: int = 500000,
                grad_accum_steps: int = 1,
                critic_lr: float= 1e-5,
                lm_lr: float = 1e-5,
                gamma: float = 0.9,
                use_wandb: bool = False,
                max_grad_norm: float = 0.01,
                save_path: str = None,
                offline_data_path: str = None,
                offline_data_path_test: str = None,
                checkpointing_steps=150,
                critic_epochs:int = 1,
                actor_epochs:int = 1,
                **kwargs):

    global global_accelerator 
    global_accelerator = accelerator[0]
    accelerator_critic, accelerator_actor = accelerator[0], accelerator[1]

    trainer = WebRLTrainer(agent=agent,\
                            accelerator=accelerator,\
                            tokenizer=tokenizer,\
                            critic_lr = critic_lr,\
                            lm_lr = lm_lr,\
                            gamma = gamma,\
                            grad_accum_steps = grad_accum_steps,
                            max_grad_norm = max_grad_norm,
                            use_wandb = use_wandb,
                            checkpointing_steps = checkpointing_steps,
                            save_path = save_path,
                            actor_epochs = actor_epochs,
                            critic_epochs = critic_epochs)
    
    replay_buffer= ReplayBuffer(batch_size= batch_size, capacity=capacity)
    
    all_trajectories = torch.load(offline_data_path)[:400]
    
    if accelerator_critic.is_main_process:
        print(f"The number of offline trajectories is {len(all_trajectories)}")
    if '_filter' not in offline_data_path:
        all_trajectories = [add_mc_return(t, gamma=gamma) for t in all_trajectories]
    if offline_data_path_test is None:
        train_trajectories = all_trajectories[:int(len(all_trajectories)*0.95)]
        val_trajectories = all_trajectories[int(len(all_trajectories)*0.95):]
    else:
        train_trajectories = all_trajectories
        val_trajectories = torch.load(offline_data_path_test)
        val_trajectories = [add_mc_return(t, gamma=gamma) for t in val_trajectories]

    replay_buffer = ReplayBuffer(batch_size= batch_size, capacity=capacity)
    validation_buffer = ReplayBuffer(batch_size= batch_size, capacity=capacity)

    if '_filter' not in offline_data_path:
        data = sum(train_trajectories, [])
    else:
        data = train_trajectories
    
    if offline_data_path_test is None:
        if '_filter' not in offline_data_path:
            val_data = sum(val_trajectories, [])
        else:
            val_data = val_trajectories
    else:
        val_data = val_trajectories
    
    for d in data:
        if d['action'].endswith('<|eot_id|>') == False:
            d['action'] = d['action'] + '<|eot_id|>'
        replay_buffer.insert(**d)
    for d in val_data:
        validation_buffer.insert(**d)
       
    # offline training
    if accelerator_critic.is_main_process:
        print(">>>Training Offline")
    info = {}

    # NOTE: the training process of critic, if you want to train critic, you can use these code
    if accelerator_critic.is_main_process:
        print(">>>Training Critic")
    info.update(trainer.update_critic(replay_buffer, validation_buffer))

    # NOTE: the training process of actor, if you want to train actor, you can use these code
    if accelerator_critic.is_main_process:
        print(">>>Training Policy")
    info.update(trainer.update_policy(replay_buffer, validation_buffer))
