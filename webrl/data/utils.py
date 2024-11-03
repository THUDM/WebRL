from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
class DummyDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]


class ReplayBuffer:
    def __init__(self, batch_size=2, capacity=10000):
        self.max_size = capacity
        self.size = 0
        self.observations = None
        self.rewards = None
        self.next_observations = None
        self.dones = None
        self.batch_size = batch_size
        self.actions = None
        self.mc_returns = None

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        rand_indices = np.random.randint(0, self.size, size=(batch_size,)) % self.max_size
        return {
            "observation": self.observations[rand_indices],
            "action": self.actions[rand_indices],
            "reward": self.rewards[rand_indices],
            "next_observation": self.next_observations[rand_indices],
            "done": self.dones[rand_indices],
            "mc_return": self.mc_returns[rand_indices],
        }
        
    def get(self, idx):
        idx = [idx]
        return {
            "observation": self.observations[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "next_observation": self.next_observations[idx],
            "done": self.dones[idx],
            "mc_return": self.mc_returns[idx],
        }

    def __len__(self):
        return self.size

    def insert(
        self,
        /,
        observation,
        action,
        reward: np.ndarray,
        next_observation,
        done: np.ndarray,
        mc_return,
        **kwargs
    ):
        """
        Insert a single transition into the replay buffer.

        Use like:
            replay_buffer.insert(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )
        """
        if isinstance(reward, (float, int)):
            reward = np.array(reward)
        if isinstance(mc_return, (float, int)):
            mc_return = np.array(mc_return)
        if isinstance(done, bool):
            done = np.array(done)
        if isinstance(reward, torch.Tensor):
            reward = reward.numpy()
        if isinstance(mc_return, torch.Tensor):
            mc_return = mc_return.numpy()

        if self.observations is None:
            self.observations = np.array(['']*self.max_size, dtype = 'object')
            self.actions = np.array(['']*self.max_size, dtype = 'object')
            self.rewards = np.empty((self.max_size, *reward.shape), dtype=reward.dtype)
            self.next_observations = np.array(['']*self.max_size, dtype = 'object')
            self.dones = np.empty((self.max_size, *done.shape), dtype=done.dtype)
            self.mc_returns = np.empty((self.max_size, *mc_return.shape), dtype=mc_return.dtype)

        assert reward.shape == ()
        assert done.shape == ()

        self.observations[self.size % self.max_size] = observation
        self.actions[self.size % self.max_size] = action
        self.rewards[self.size % self.max_size] = reward
        self.next_observations[self.size % self.max_size] = next_observation
        self.dones[self.size % self.max_size] = done
        self.mc_returns[self.size % self.max_size] = mc_return

        self.size += 1
        
    def reset(self):
        self.max_size = capacity
        self.size = 0
        self.observations = None
        self.rewards = None
        self.next_observations = None
        self.dones = None
        self.batch_size = batch_size
        self.actions = None
        self.mc_returns = None