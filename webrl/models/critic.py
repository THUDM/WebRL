import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch.nn as nn
from accelerate import Accelerator

class VLMDoubleCritic(nn.Module):
    def __init__(self, device, accelerator, critic_lm, cache_dir, in_dim, out_dim):
        super(VLMDoubleCritic, self).__init__()
        out_dim = 2
        self.device = device
        self.accelerator = accelerator
        self.base_lm = AutoModelForCausalLM.from_pretrained(critic_lm, use_cache=False, trust_remote_code=True, low_cpu_mem_usage=True).model.to('cpu')
        # self.base_lm = AutoModelForCausalLM.from_pretrained(critic_lm, use_cache=False, trust_remote_code=True, low_cpu_mem_usage=True).transformer.to('cpu')
        self.config = self.base_lm.config
        self.base_tokenizer = AutoTokenizer.from_pretrained(critic_lm, use_cache=False, trust_remote_code=True)
        self.base_tokenizer.truncation_side = 'right'
        self.base_tokenizer.padding_side = 'right'
        self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        self.critic1 = nn.Sequential(nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, out_dim))
        self.critic2 = nn.Sequential(nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, out_dim))
        initializer_range = 0.2
        self.critic1[0].weight.data.normal_(mean=0.0, std=initializer_range)
        self.critic1[0].bias.data.zero_()
        self.critic1[2].weight.data.normal_(mean=0.0, std=initializer_range)
        self.critic1[2].bias.data.zero_()
        self.critic2[0].weight.data.normal_(mean=0.0, std=initializer_range)
        self.critic2[0].bias.data.zero_()
        self.critic2[2].weight.data.normal_(mean=0.0, std=initializer_range)
        self.critic2[2].bias.data.zero_()
    
    def gradient_checkpointing_enable(self):
        self.base_lm.gradient_checkpointing_enable()
        
    def gradient_checkpointing_disable(self):
        self.base_lm.gradient_checkpointing_disable()
    
    def forward(self, observation, action, detach_model=False):
        # detach_model = True
        obs_ids = self.base_tokenizer(observation, padding = True, return_tensors='pt', max_length=16384, truncation = True).to(self.device)
        
        # modify the parameter for the forward function
        obs_ids['return_dict'] = True
        obs_ids['output_hidden_states'] = True
        
        if detach_model:
            with torch.no_grad():
                lm_states = self.base_lm(**obs_ids)[0][:, -1, :]
        else:
            lm_states = self.base_lm(**obs_ids)[0][:, -1, :]
        v_states = lm_states
        return self.critic1(v_states), self.critic2(v_states)

