# %% [markdown]
# ### Imports

# %%
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from collections import deque
import random
import mlflow

# %% [markdown]
# ### Hyperparameters

# %%
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
REPLAY_BUFFER_SIZE = int(1e6)
BATCH_SIZE = 128
GAMMA = 0.99
MAX_EPISODES = 1000
MAX_STEPS = int(1e3)
TARGET_REWARD = 200
MIN_MEMORY = 1e4
TARGET_POLICY_CLIP = 0.5

TAU = 0.005 
POLICY_DELAY = 3 
TARGET_POLICY_NOISE = 0.3

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("TARGET_POLICY_NOISE_")

# %% [markdown]
# ### Environment

# %%
env = gym.make('LunarLanderContinuous-v2')
state_dim = env.observation_space.shape[0] # state representation
action_dim = env.action_space.shape[0] # possible actions
max_action = float(env.action_space.high[0]) # max of one throttle 

print('state_dim:', state_dim)
print('action_dim:', action_dim)
print('max_action:', max_action)

# %% [markdown]
# #### Neural Networks

# %%
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 512)
        # self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return self.max_action * x

# %%
actor = Actor(state_dim, action_dim, max_action)
actor

# %%
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 512)
        # self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.q = nn.Linear(256, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(state_action))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        q_value = self.q(x)
        return q_value

# %%
class TwinCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # First Q-Network
        self.fc1_1 = nn.Linear(state_dim + action_dim, 512)
        self.fc2_1 = nn.Linear(512, 512)
        self.fc3_1 = nn.Linear(512, 256)
        self.q1 = nn.Linear(256, 1)

        # Second Q-Network
        self.fc1_2 = nn.Linear(state_dim + action_dim, 512)
        self.fc2_2 = nn.Linear(512, 512)
        self.fc3_2 = nn.Linear(512, 256)
        self.q2 = nn.Linear(256, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)

        # Compute Q1
        x1 = torch.relu(self.fc1_1(state_action))
        x1 = torch.relu(self.fc2_1(x1))
        x1 = torch.relu(self.fc3_1(x1))
        q1 = self.q1(x1)

        # Compute Q2
        x2 = torch.relu(self.fc1_2(state_action))
        x2 = torch.relu(self.fc2_2(x2))
        x2 = torch.relu(self.fc3_2(x2))
        q2 = self.q2(x2)

        return q1, q2 


# %%
# critic = Critic(state_dim, action_dim)
critic = TwinCritic(state_dim, action_dim)
critic

# %% [markdown]
# ### Agents

# %%
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mean, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        self.mean = mean
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mean)

    def __call__(self):
        noise = self.theta * (self.mean - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(*self.mean.shape)
        self.x_prev += noise
        return self.x_prev

class DDPG_Agent():
    
    def __init__(self, state_dim, action_dim, max_action):
        # Actor
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        actor_target_state_dict = self.actor_target.state_dict()
        self.actor_target.load_state_dict(actor_target_state_dict)

        # Critic
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        critic_target_state_dict = self.critic_target.state_dict()
        self.critic_target.load_state_dict(critic_target_state_dict)

        # Optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # Other
        self.memory = deque(maxlen=REPLAY_BUFFER_SIZE)

        # Max Action
        self.max_action = max_action
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)  

        # OU Noise for exploration
        self.noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(action_dim), sigma=0.2) 

    def select_action(self, state, noise=0.1):
        # Convert state to tensor
        state = torch.FloatTensor(state).to(self.device)

        # Get action from actor network)
        action = self.actor(state).cpu().detach().numpy()

        # Add exploration noise
        action += self.noise()

        # Clip action to return an action within valid range of possible actions
        return np.clip(action, -self.max_action, self.max_action)
    
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < MIN_MEMORY:
            return # We need to wait until enough experience is collected
        
        # Sample from replay buffer
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute targets without tracking gradients
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = rewards + GAMMA * (1-dones) * self.critic_target(next_states, next_actions)

        # Critic Step
        q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q_values, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Actor Step
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Soft Update Target Networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            # Update target parameters in place
            target_param.data.copy_(TAU * param.data + (1-TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            # Update target params in place
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)


class TD3_Agent():
    def __init__(self, state_dim, action_dim, max_action):
        # Actor
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Twin Critics
        self.critic = TwinCritic(state_dim, action_dim)
        self.critic_target = TwinCritic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # Exploration Noise
        self.noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(action_dim), sigma=0.2) 

        # Replay Buffer
        self.memory = deque(maxlen=REPLAY_BUFFER_SIZE)

        # Max Action
        self.max_action = max_action

        # Training step counter
        self.train_step = 0 

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)


    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state).cpu().detach().numpy()

        # Add Gaussian noise
        action += np.random.normal(0, noise, size=action.shape)
        return np.clip(action, -self.max_action, self.max_action)

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def train(self):
        if len(self.memory) < MIN_MEMORY:
            return  # Wait until enough experience is collected
        
        # Sample batch from replay buffer
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Target Policy Smoothing: Add noise to target actions
        with torch.no_grad():
            noise = torch.clamp(
                torch.normal(0, TARGET_POLICY_CLIP, size=actions.shape).to(self.device), -0.5, 0.5
            )
            next_actions = self.actor_target(next_states) + noise
            next_actions = torch.clamp(next_actions, -self.max_action, self.max_action)

            # Compute both target Q-values
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            target_q = rewards + GAMMA * (1 - dones) * torch.min(q1_next, q2_next)

        # Compute current Q-values from twin critics
        q1, q2 = self.critic(states, actions)
        mlflow.log_metric('Q1_Value', q1.mean().item(), step=self.train_step)
        mlflow.log_metric('Q2_values', q2.mean().item(), step=self.train_step)
        
        # Critic Loss (Huber loss instead of MSE for stability)
        critic_loss = F.smooth_l1_loss(q1, target_q) + F.smooth_l1_loss(q2, target_q)
        mlflow.log_metric('Critic_Loss', critic_loss.item(), self.train_step)

        # Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # TD Error (Bellman Residual)
        td_error = (target_q - q1).mean().abs().item()
        mlflow.log_metric('TD_Error', td_error, self.train_step)

        # Entropy Action
        action_variance = actions.var().item() # Variance of selected actions
        mlflow.log_metric('Entropy_Actions', action_variance, self.train_step)

        # Delayed Actor Updates
        if self.train_step % POLICY_DELAY == 0:
            # Compute actor loss (use Q1 only)
            actor_loss = -self.critic(states, self.actor(states))[0].mean()
            mlflow.log_metric('Policy_Loss', actor_loss.item(), self.train_step)

            # Update Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft Update Target Networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        self.train_step += 1        

# %% [markdown]
# ### Train Loop

# %%
import datetime

date = datetime.datetime.now().strftime("%Y-%m-%d")
NAME = "DELETE"
TRAIN_RUN_NAME = f"{NAME}__train"

# %%
env = gym.make('LunarLanderContinuous-v2')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = TD3_Agent(state_dim, action_dim, max_action)
reward_history = []

try:
    if mlflow.active_run():
        run_id = mlflow.active_run().info.run_id
        print(f"Attempting to end active run: {run_id}")
        mlflow.end_run()
except Exception as e:
    print(f"Warning: Tried to end an inactive run. Skipping. Error: {e}")

mlflow.set_experiment('TARGET_POLICY_NOISE_')

mlflow.set_tracking_uri("file:./mlruns") 
mlflow.autolog(disable=True)

with mlflow.start_run(run_name=TRAIN_RUN_NAME, nested=True):
    mlflow.log_param('algorithm', 'DDPG')
    mlflow.log_param('lr_actor', LR_ACTOR)
    mlflow.log_param('lr_critic', LR_CRITIC)
    mlflow.log_param('replay_buffer_size', REPLAY_BUFFER_SIZE)
    mlflow.log_param('batch_size', BATCH_SIZE)
    mlflow.log_param('gamma',GAMMA)
    mlflow.log_param('tau_polyak_averaging_factor',TAU)
    mlflow.log_param('max_episodes', MAX_EPISODES)
    mlflow.log_param('max_steps', MAX_STEPS)
    mlflow.log_param('target_reward', TARGET_REWARD)

    for episode in range(MAX_EPISODES):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(MAX_STEPS):
            action = agent.select_action(state, noise=0.1)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_experience(state, action, reward, next_state, done)
            agent.train()
            episode_reward += reward
            state = next_state
            if done:
                break

        reward_history.append(episode_reward)
        avg_last_10_rewards = np.mean(reward_history[-10:])
        
        print(f'Episode rewards: {episode + 1}, Reward: {episode_reward}')

        mlflow.log_metric('train_episode_reward', episode_reward, step=episode)
        mlflow.log_metric('train_avg_last_10_rewards', avg_last_10_rewards, step=episode)

        if len(reward_history) >= 100:
            if np.mean(reward_history[-100:]) >= TARGET_REWARD:
                print('TARGET_REWARD Achieved!')
                break

    env.close()

    actor_path = f'./models/{TRAIN_RUN_NAME}.pth'
    torch.save(agent.actor.state_dict(), actor_path)
    mlflow.log_artifact(actor_path)
    

# %% [markdown]
# ### TEST

# %%
import gymnasium as gym
import torch
import numpy as np
import mlflow
import matplotlib.pyplot as plt

# Load environment
env = gym.make('LunarLanderContinuous-v2', render_mode='human')

# Get environment details
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Load trained agent
# agent = DDPG_Agent(state_dim, action_dim, max_action)
agent = TD3_Agent(state_dim, action_dim, max_action)
actor_path = f'./models/{TRAIN_RUN_NAME}.pth'
agent.actor.load_state_dict(torch.load(actor_path))
agent.actor.eval() 

# Set MLflow tracking
mlflow.set_tracking_uri("file:./mlruns")

try:
    if mlflow.active_run():
        run_id = mlflow.active_run().info.run_id
        print(f"Attempting to end active run: {run_id}")
        mlflow.end_run()
except Exception as e:
    print(f"Warning: Tried to end an inactive run. Skipping. Error: {e}")

TEST_RUN_NAME = f"{NAME}__test"

with mlflow.start_run(run_name=TEST_RUN_NAME, nested=True):
    mlflow.log_param("test_episodes", 100)
    
    test_rewards = []
    
    for episode in range(100):
        state, _ = env.reset()
        episode_reward = 0

        while True:
            action = agent.select_action(state, noise=0)  # No exploration noise during testing
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
            if done:
                break

        test_rewards.append(episode_reward)
        avg_last_10_rewards = np.mean(reward_history[-10:])
        

        mlflow.log_metric('test_episode_reward', episode_reward, step=episode)
        # mlflow.log_metric('test_avg_last_10_rewards', avg_last_10_rewards, step=episode)

        print(f"Test Episode {episode+1}: Reward = {episode_reward}")

    env.close()


