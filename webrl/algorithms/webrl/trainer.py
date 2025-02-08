import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from webrl.data import DummyDataset
import random
import wandb
import math
from transformers import get_scheduler
import os
import time
import deepspeed
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import shutil
from accelerate.utils import pad_across_processes

        
import subprocess

def print_simple_gpu_usage():
    current_rank = int(os.getenv('RANK', -1))
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,nounits,noheader'], 
                            stdout=subprocess.PIPE, text=True)
    for i, line in enumerate(result.stdout.strip().split('\n')):
        total, used, free = line.split(',')
        print(f"Rank {current_rank}, GPU {i}: Total Memory: {total} MB, Used Memory: {used} MB, Free Memory: {free} MB")

def dict_mean(dict_list):
    mean_dict = {}
    if len(dict_list) > 0:
        for key in dict_list[0].keys():
            if "min" in key:
                mean_dict[key] = min(d[key] for d in dict_list if key in d)
            elif "max" in key:
                mean_dict[key] = max(d[key] for d in dict_list if key in d)
            else:
                tmp_list = [d[key] for d in dict_list if key in d]
                if len(tmp_list) != 0:
                    mean_dict[key] = sum(tmp_list) / len(tmp_list)
                else:
                    mean_dict[key] = 0
    return mean_dict

class WebRLTrainer():
    def __init__(self, agent,
                    accelerator,
                    tokenizer,
                    critic_lr: float = 1e-3,
                    lm_lr: float = 1e-5,
                    grad_accum_steps: int = 8,
                    gamma: float = 0.9,
                    max_grad_norm: float=0.01,
                    use_wandb: bool = False,
                    checkpointing_steps: int = 500,
                    save_path: str = 'logs',
                    actor_epochs: int = 1,
                    critic_epochs: int = 1):
        """
        beta: coefficient for the bc loss
        """
        super().__init__()
        self.agent = agent
        self.tokenizer = tokenizer
        self.lm_optimizer = torch.optim.AdamW(agent.model.parameters(), lr = lm_lr)
        self.critic_optimizer = torch.optim.AdamW(agent.critic.parameters(), lr = critic_lr)
        self.ref_optimizer = torch.optim.AdamW(agent.ref_model.parameters(), lr = lm_lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.mse_loss = torch.nn.SmoothL1Loss(reduction='none')
        # self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.grad_accum_steps = grad_accum_steps
        self.gamma = gamma
        self.step = 0
        self.max_grad_norm = max_grad_norm
        self.accelerator = accelerator
        self.softmax = torch.nn.Softmax(dim = -1)
        self.use_wandb = use_wandb
        self.checkpointing_steps = checkpointing_steps
        self.save_path = save_path
        self.actor_epochs = actor_epochs
        self.critic_epochs = critic_epochs

    
    def critic_loss(self, observation, action, reward, next_observation, done, mc_return,
                    validation = False, **kwargs):
        reward = torch.Tensor(reward).to(self.accelerator[0].unwrap_model(self.agent.critic).device, dtype = self.accelerator[0].unwrap_model(self.agent.model).dtype).flatten()
        done = torch.Tensor(done).to(self.accelerator[0].unwrap_model(self.agent.critic).device, dtype = self.accelerator[0].unwrap_model(self.agent.model).dtype).flatten()
        mc_return = torch.Tensor(mc_return).to(self.accelerator[0].unwrap_model(self.agent.critic).device, dtype = self.accelerator[0].unwrap_model(self.agent.model).dtype).flatten()
        v1, v2 = self.agent.critic(observation, action, detach_model=False)
        nv1, nv2 = self.agent.critic(next_observation, action, detach_model=False)

        v1 = v1.reshape(-1, 2)
        v2 = v2.reshape(-1, 2)
        nv1 = nv1.reshape(-1, 2)
        nv2 = nv2.reshape(-1, 2)
        
        regression_target = (mc_return > 0).long()
        v1_loss = self.criterion(v1, regression_target)
        v1_acc = (v1.argmax(dim = 1) == regression_target).float().mean()
        v2_loss = self.criterion(v2, regression_target)
        v2_acc = (v2.argmax(dim = 1) == regression_target).float().mean()
        nv1_loss = self.criterion(nv1, regression_target)
        nv2_loss = self.criterion(nv2, regression_target)
        if not validation:
            self.accelerator[0].backward(v1_loss + v2_loss + nv1_loss + nv2_loss)
        v1_loss, v2_loss = v1_loss.detach().cpu(), v2_loss.detach().cpu()
        v1_acc, v2_acc = v1_acc.detach().cpu(), v2_acc.detach().cpu()
        
        
        info = {"v1.loss": v1_loss,\
                "v2.loss": v2_loss,\
                "v1.acc": v1_acc,\
                "v2.acc": v2_acc,\
                "v1.mean": torch.mean(v1),\
                "v1.min": torch.min(v1),\
                "v1.max": torch.max(v1),\
                "v1.std": torch.std(v1),
                "v2.mean": torch.mean(v2),
                "v2.max": torch.max(v2),
                "v2.min": torch.min(v2),
                "v2.std": torch.std(v2),
                }
        if validation:
            validation_info = {}
            for k,v in info.items():
                validation_info["validation."+k] = v
            return validation_info
        return info

    def actor_loss_sft(self, observation, action, next_observation, mc_return, pi_action, advantage, reward,
                   validation=False,**kwargs):
        log_prob, action_attention_mask = self.agent.get_log_prob(observation, action)
        pg_loss = -log_prob.sum() / action_attention_mask.sum()
        value_loss = torch.zeros_like(pg_loss)
        if not validation and pg_loss is not None:
            self.accelerator[1].backward(pg_loss+value_loss)
        info =  {"pg.loss": pg_loss.detach().cpu().item()}
        if validation:
            validation_info = {}
            for k,v in info.items():
                validation_info["validation."+k] = v
            validation_info['log_prob'] = log_prob.detach().cpu().mean()
            return validation_info
        return info
    
    def actor_loss(self, observation, action, next_observation, mc_return, pi_action, advantage, reward, add_sft_loss=False,
                   validation=False,**kwargs):
        mc_return = torch.Tensor(mc_return).to(self.accelerator[1].unwrap_model(self.agent.model).device, dtype = self.accelerator[1].unwrap_model(self.agent.model).dtype).flatten()
        reward = torch.Tensor(reward).to(self.accelerator[1].unwrap_model(self.agent.model).device, dtype = self.accelerator[1].unwrap_model(self.agent.model).dtype).flatten()
        with torch.no_grad():
            v1, v2 = self.agent.critic(observation, action, detach_model=False)
            nv1, nv2 = self.agent.critic(next_observation, action, detach_model=False)
            v1 = self.softmax(v1)[:, 1]
            v2 = self.softmax(v2)[:, 1]
            nv1 = self.softmax(nv1)[:, 1]
            nv2 = self.softmax(nv2)[:, 1]
            v = torch.minimum(v1, v2).flatten()
            nv = torch.minimum(nv1, nv2).flatten()
            advantage = nv - v - 0.05 + reward + mc_return
            advantage = torch.clamp(advantage, -1, 1).to(self.accelerator[1].unwrap_model(self.agent.model).device, dtype = self.accelerator[1].unwrap_model(self.agent.model).dtype)

        log_prob, action_attention_mask = self.agent.get_log_prob(observation, action)
        log_prob_detach = log_prob.detach()
        ref_log_prob = self.agent.get_log_prob_ref(observation, action).to(log_prob.device)
        non_zero_counts = action_attention_mask.sum(dim=1)
        
        advantages = advantage.flatten()
        advantages = torch.where(advantages >= 0, torch.tensor(1.0), torch.tensor(-0.6)).to(log_prob.device)
        
        log_prob = log_prob.sum(dim = 1).flatten()
        ref_log_prob = ref_log_prob.sum(dim = 1).flatten()
        # ref_prob = torch.exp(ref_log_prob).mean(dim = 1)
        ref_prob = torch.exp(ref_log_prob / non_zero_counts)
        
        beta = torch.ones_like(advantages).to(advantages.device)
        cond1 = (advantages >= 0) & (ref_prob >= 0.8)
        cond2 = (advantages < 0) & (ref_prob < 0.9)
        beta[cond1] = 5.0
        beta[cond2] = 5.0
        
        # loss = self.mse_loss(beta * log_prob_ratio, advantages)
        mask = (advantages > 0).to(dtype=log_prob.dtype, device=log_prob.device)
        
        non_zero_counts_mask = non_zero_counts[mask.bool()].sum()
        safe_count = torch.where(non_zero_counts_mask > 0, non_zero_counts_mask, torch.tensor(1.0, dtype=non_zero_counts_mask.dtype))
        
        log_prob_ratio = (log_prob - ref_log_prob) / safe_count
        
        ratio = torch.abs(advantages / log_prob_ratio)
        beta = torch.where(beta <= ratio, beta, ratio).detach()
        
        loss = self.mse_loss(beta * log_prob_ratio, advantages)
        loss = torch.reciprocal(beta) * loss * mask
        # loss = loss * mask
        loss = loss.mean()
        value_loss = torch.zeros_like(loss)
        if add_sft_loss == True:
            log_prob_detached = log_prob.detach()
            prob_detached = torch.exp(log_prob_detached / non_zero_counts)
            coefficient = torch.exp(1 - prob_detached)
            
            sft_loss = -(log_prob * coefficient)[mask.bool()].sum() / safe_count
            # sft_loss = -log_prob[mask.bool()].sum() / safe_count
            loss += sft_loss
            
        # if not validation and loss is not None and mask.sum() != 0:
        if not validation and loss is not None:
            self.accelerator[1].backward(loss + value_loss)

        advantages = advantages.detach().cpu()
        info =  {"pg.loss": loss.detach().cpu().item() if loss is not None else 0,
                "advantages.mean": advantages.mean(),
                "log_prob": (log_prob / safe_count).cpu().mean(),
                "ref_prob": ref_prob.cpu().mean(),
                "ref_log_prob": (ref_log_prob / safe_count).cpu().mean(),
                "log_prob_ratio": log_prob_ratio.cpu().mean(),
                "mask": mask.cpu().sum(),
                "beta": beta.cpu().mean(),
                "sft_loss": sft_loss.cpu().mean() if add_sft_loss == True else 0,
                "non_zero_counts": torch.sum(non_zero_counts).cpu(),
                "beta_log_prob_ratio": (beta * log_prob_ratio).cpu().mean()}
        if validation:
            validation_info = {}
            for k,v in info.items():
                validation_info["validation."+k] = v
            validation_info['log_prob'] = log_prob.detach().cpu().mean()
            return validation_info
        return info

    
    def update_critic(self, replay_buffer, validation_buffer = None):
        data = [replay_buffer.get(i) for i in range(len(replay_buffer))]
        if self.accelerator[0].is_main_process:
            print(f'====================The size of training data is: {len(data)}!====================')
            os.makedirs(os.path.join(self.save_path, 'critic'), exist_ok=True)
        for d in data:
            for k,v in d.items():
                d[k] = v[0]
        dataloader = DataLoader(DummyDataset(data), batch_size=replay_buffer.batch_size)
        # Scheduler and math around the number of training steps.
        print(len(dataloader), ' ', replay_buffer.batch_size)
        num_update_steps_per_epoch = math.ceil(len(dataloader) / self.accelerator[0].gradient_accumulation_steps)
        max_train_steps = self.critic_epochs * num_update_steps_per_epoch
        self.agent.critic.base_lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.agent.critic.base_lm.train()
        self.step = 0
        
        self.agent.critic, self.critic_optimizer, dataloader = self.accelerator[0].prepare(self.agent.critic, self.critic_optimizer, dataloader)
        
        self.agent.critic.base_lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.agent.critic.base_lm.train()
        
        # for name, module in self.agent.critic.base_lm.named_modules():
        #     if hasattr(module, 'gradient_checkpointing'):
        #         print(f'Module: {name}, Gradient Checkpointing: {module.gradient_checkpointing}')
        
        # Afterwards we recalculate our number of training epochs
        if self.accelerator[0].is_main_process:
            print(f'====================The size of data loader is: {len(dataloader)}!====================')
        # print_simple_gpu_usage()
        
        num_update_steps_per_epoch = math.ceil(len(dataloader) / self.accelerator[0].gradient_accumulation_steps)
        max_train_steps =  self.critic_epochs * num_update_steps_per_epoch
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
        progress_bar = tqdm(range(max_train_steps), disable=not self.accelerator[0].is_local_main_process)
        
        valid_data = [validation_buffer.get(i) for i in range(len(validation_buffer))]
        for d in valid_data:
            for k,v in d.items():
                d[k] = v[0]
        valid_dataloader = DataLoader(DummyDataset(valid_data), batch_size=replay_buffer.batch_size)
        valid_dataloader = self.accelerator[0].prepare(valid_dataloader)
        for i in range(num_train_epochs):
            info = {}
            with torch.autograd.set_detect_anomaly(True):
                info_list = []
                for batch in dataloader:
                    with self.accelerator[0].accumulate(self.agent.critic):
                        info_list.append(self.critic_loss(**batch))
                        self.accelerator[0].clip_grad_norm_(self.agent.critic.parameters(), self.max_grad_norm)
                        self.critic_optimizer.step()
                        self.critic_optimizer.zero_grad()
                    if self.accelerator[0].sync_gradients:
                        self.step += 1
                        progress_bar.update(1)
                        info.update(dict_mean(info_list))
                        info_list = []
                        if self.accelerator[0].is_main_process:
                            if self.use_wandb: 
                                wandb.log(info)
                        if self.step == max_train_steps - 1:
                            self.save_critic(os.path.join(self.save_path, 'critic'))
                            
                        if self.step % self.checkpointing_steps == 0:
                            if self.accelerator[0].is_main_process:
                                os.makedirs(os.path.join(self.save_path, 'critic', f'steps_{self.step}'), exist_ok=True)
                            self.save_critic(os.path.join(self.save_path, 'critic', f'steps_{self.step}'))
                                
                        if validation_buffer is not None and self.step % self.checkpointing_steps == 0:
                            valid_info = {}
                            valid_info_list = []
                            # torch.cuda.synchronize()
                            with torch.no_grad():
                                for batch in valid_dataloader:
                                    valid_info_list.append(self.critic_loss(validation=True, **batch))
                            valid_info.update(dict_mean(valid_info_list))
                            
                            if self.use_wandb and self.accelerator[0].is_main_process:
                                wandb.log(valid_info)
                    
        return info
        
        
    def update_policy(self, replay_buffer, validation_buffer = None, no_update_actor=False):
        self.step = 0
        info = {}
        info_list = []
        # update actor
        if not no_update_actor:
            if self.accelerator[1].is_main_process:
                print(">>>updating actor")
            data = [replay_buffer.get(i) for i in range(len(replay_buffer))]
            if self.accelerator[1].is_main_process:
                print(f'====================The size of training data is: {len(data)}!====================')
            for d in data:
                for k,v in d.items():
                    d[k] = v[0]
                    
            dataloader = DataLoader(DummyDataset(data), batch_size=replay_buffer.batch_size)
            self.agent.model.gradient_checkpointing_enable()
            self.agent.model.train()
            self.step = 0
            self.agent.model, self.lm_optimizer, dataloader = self.accelerator[1].prepare(self.agent.model, self.lm_optimizer, dataloader)
            # self.agent.ref_model = self.agent.ref_model.to(self.accelerator[1].device)
            self.agent.ref_model, self.ref_optimizer, dataloader_ = self.accelerator[2].prepare(self.agent.ref_model, self.ref_optimizer, dataloader)
            # Afterwards we recalculate our number of training epochs
            if self.accelerator[1].is_main_process:
                print(f'====================The size of data loader is: {len(dataloader)}!====================')
            num_update_steps_per_epoch = math.ceil(len(dataloader) / self.accelerator[1].gradient_accumulation_steps)
            max_train_steps =  self.actor_epochs * num_update_steps_per_epoch
            num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
            progress_bar = tqdm(range(max_train_steps), disable=not self.accelerator[1].is_local_main_process)
            
            valid_data = [validation_buffer.get(i) for i in range(len(validation_buffer))]
            for d in valid_data:
                for k,v in d.items():
                    d[k] = v[0]
            valid_dataloader = DataLoader(DummyDataset(valid_data), batch_size=replay_buffer.batch_size)

            for i in range(num_train_epochs):
                for batch in dataloader:
                    with self.accelerator[1].accumulate(self.agent.model):
                        pi_action = None
                        advantages = None
                        info = self.actor_loss(**batch, pi_action=pi_action, advantage=advantages)
                        info_list.append(info)
                        
                        self.accelerator[1].clip_grad_norm_(self.agent.model.parameters(), self.max_grad_norm)
                        
                        grad_norm = self.agent.model.get_global_grad_norm()
                        # In some cases the grad norm may not return a float
                        if hasattr(grad_norm, "item"):
                            grad_norm = grad_norm.item()
                        if self.accelerator[1].is_main_process and self.use_wandb:
                            wandb.log({"avg_grad_norm": grad_norm})
                        
                        self.lm_optimizer.step()
                        self.lm_optimizer.zero_grad()
                        
                    # print(self.accelerator[1].process_index)
                    if self.accelerator[1].sync_gradients:
                        self.step += 1
                        progress_bar.update(1)
                        if self.step == max_train_steps - 1:
                            self.save_actor(os.path.join(self.save_path, 'actor'))
                        # info.update(dict_mean(info_list))
                        # info_list = []
                        if self.accelerator[1].is_main_process:
                            if self.use_wandb: 
                                for info in info_list:
                                    wandb.log(info)
                        info.update(dict_mean(info_list))
                        info_list = []
                        if self.step % self.checkpointing_steps == 0:
                            if self.accelerator[1].is_main_process:
                                os.makedirs(os.path.join(self.save_path, 'actor', f'steps_{self.step}'), exist_ok=True)
                            self.save_actor(os.path.join(self.save_path, 'actor', f'steps_{self.step}'))   
                        
                        if validation_buffer is not None and self.step % self.checkpointing_steps == 0:
                            valid_info = {}
                            valid_info_list = []
                            # torch.cuda.synchronize()
                            with torch.no_grad():
                                for batch in tqdm(valid_dataloader, disable= not self.accelerator[1].is_local_main_process):
                                    valid_info_list.append(self.actor_loss(validation=True, pi_action=None, advantage=None, **batch))
                            valid_info.update(dict_mean(valid_info_list))
                            time.sleep(60)
                            if self.use_wandb and self.accelerator[1].is_main_process:
                                wandb.log(valid_info)
                            
        return info

    def save_critic(self, path):
        self.accelerator[0].wait_for_everyone()
        # self.accelerator.save_state(os.path.join(path, 'training_state'), safe_serialization=False)
        self.agent.critic.save_checkpoint(path)
        # state_dict = self.accelerator[0].get_state_dict(self.agent.critic)
        if self.accelerator[0].is_main_process:
            state_dict = get_fp32_state_dict_from_zero_checkpoint(path)
            for state in state_dict:
                print(state, state_dict[state].shape)
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            # self.accelerator[0].save(new_state_dict, os.path.join(path, 'pytorch_critic.bin'))
            torch.save(new_state_dict, os.path.join(path, 'pytorch_critic.bin'))
            
            latest_path = os.path.join(path, 'latest')
            if os.path.isfile(latest_path):
                with open(latest_path, 'r') as fd:
                    tag = fd.read().strip()
            else:
                raise ValueError(f"Unable to find 'latest' file at {latest_path}")
            ds_checkpoint_dir = os.path.join(path, tag)
            shutil.rmtree(ds_checkpoint_dir)
        
    def save_actor(self, path):
        torch.cuda.empty_cache()
        self.accelerator[1].wait_for_everyone()
        self.agent.model.save_checkpoint(path)
        if self.accelerator[1].is_main_process:
            # state_dict = self.agent.model.state_dict()
            # new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            # self.accelerator[1].save(new_state_dict, os.path.join(path, 'pytorch_actor.bin'))
            state_dict = get_fp32_state_dict_from_zero_checkpoint(path)
            for state in state_dict:
                print(state, state_dict[state].shape)
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            # self.accelerator[0].save(new_state_dict, os.path.join(path, 'pytorch_critic.bin'))
            torch.save(new_state_dict, os.path.join(path, 'pytorch_actor.bin'))
            
            latest_path = os.path.join(path, 'latest')
            if os.path.isfile(latest_path):
                with open(latest_path, 'r') as fd:
                    tag = fd.read().strip()
            else:
                raise ValueError(f"Unable to find 'latest' file at {latest_path}")
            ds_checkpoint_dir = os.path.join(path, tag)
            shutil.rmtree(ds_checkpoint_dir)
        self.tokenizer.save_pretrained(path)
