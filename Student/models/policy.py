
import torch
import torch.nn as nn
from torch.distributions import Normal

# Reference: https://github.com/DSSC-projects/soft-actor-critic/blob/f6599dde8d284e33969c57443748c7d00e06f1d3/src/neuralnets.py#L92
class PolicyNetwork(nn.Module):
   
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=84):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action

    def forward(self, state):
        x = self.shared(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -5, 2)
        return mean, log_std

    def get_action(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        if deterministic:
            action = torch.tanh(mean) * self.max_action
            return action, None

        dist = Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action) * self.max_action

        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.log(1 - action.pow(2) / (self.max_action ** 2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def evaluate_actions(self, states, actions):
        mean, log_std = self.forward(states)
        std = log_std.exp()
        dist = Normal(mean, std)

        clipped = actions / self.max_action
        clipped = torch.clamp(clipped, -0.999, 0.999)
        raw_actions = torch.atanh(clipped)

        log_prob = dist.log_prob(raw_actions)
        log_prob -= torch.log(1 - clipped.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return log_prob, entropy

# Reference: https://github.com/DSSC-projects/soft-actor-critic/blob/f6599dde8d284e33969c57443748c7d00e06f1d3/src/neuralnets.py#L183
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=84):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state)
