import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from webrl.models.critic import VLMDoubleCritic
import signal
import time
import requests

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)


class LlamaAgent():
    def __init__(self, device, accelerator, policy_lm, critic_lm, ref_lm = None,
                cache_dir = '~/.cache', dropout = 0.5, TEMPLATE = None, use_lora=False,
                do_sample = True, temperature = 1.0, max_new_tokens = 32, use_bfloat16 = False, eos_str = None, critic = None):
        if use_bfloat16:
            self.model = AutoModelForCausalLM.from_pretrained(policy_lm, use_cache=False,
                                                              torch_dtype = torch.bfloat16, trust_remote_code=True)
        else:
            print(policy_lm)
            self.model = AutoModelForCausalLM.from_pretrained(policy_lm, use_cache=False, trust_remote_code=True).to('cpu')

        critic_config = AutoConfig.from_pretrained(critic_lm, trust_remote_code=True)
        self.template = TEMPLATE
        self.policy_lm = policy_lm
        if critic is None:
            self.critic = VLMDoubleCritic(device[0], accelerator[0], critic_lm = critic_lm, cache_dir = cache_dir, in_dim = critic_config.hidden_size, out_dim = 1)  
        else:
            self.critic = critic
            self.critic.device = device[0]
            self.critic.accelerator = accelerator[0]
            
        self.target_critic = None
        self.tokenizer = AutoTokenizer.from_pretrained(policy_lm, trust_remote_code=True, use_cache=False)
        self.tokenizer.truncation_side = 'right'
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
        
        if ref_lm is None:
            ref_lm = policy_lm
        self.ref_model = AutoModelForCausalLM.from_pretrained(policy_lm, use_cache=False, trust_remote_code=True).to('cpu')
    
    def post_init(self, device, accelerator):
        self.device = device
        self.accelerator = accelerator
        self.critic.device = device[0]
        self.critic.accelerator = accelerator[0]
        
    def update_critic(self, critic_resume_path):
        print(f'Resuming of Critic from {critic_resume_path}')
        torch.cuda.empty_cache()
        state_dict = torch.load(critic_resume_path, map_location='cpu')
        self.critic.load_state_dict(state_dict)
    
    def get_log_prob(self, observation, action, device=None):
        if device is None:
            device = self.accelerator[1].device
        input_texts = [obs + act for obs, act in zip(observation, action)]
        input_ids = self.tokenizer(input_texts, return_tensors='pt', padding=True, max_length=16384, truncation = True).to(device)
        
        obs_lengths = [len(self.tokenizer(obs, return_tensors='pt')['input_ids'][0]) for obs in observation]
        
        outputs = self.model(labels = input_ids["input_ids"].to(device),
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

        return action_log_probs * action_attention_mask, action_attention_mask

    def get_log_prob_ref(self, observation, action, device=None):
        if device is None:
            device = self.ref_model.device
        with torch.no_grad():
            input_texts = [obs + act for obs, act in zip(observation, action)]
            input_ids = self.tokenizer(input_texts, return_tensors='pt', padding=True, max_length=16384, truncation = True).to(device)
            obs_lengths = [len(self.tokenizer(obs, return_tensors='pt')['input_ids'][0]) for obs in observation]
            outputs = self.ref_model(labels = input_ids["input_ids"],
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
            return action_log_probs * action_attention_mask