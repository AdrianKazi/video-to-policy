# test/test.py

import os
import torch
import numpy as np

from gymnasium.wrappers import RecordVideo

from env.env import make_env, get_env_dims
from agents.td3 import TD3_Agent
from config.config import *
from tracking.mlflow_utils import setup_mlflow, start_run
import mlflow


def test():

    VIDEO_PATH = "videos"
    MODEL_PATH = "models_saved/actor.pth"
    POST_LANDING_FRAMES = 100   # 2 sec extra

    os.makedirs(VIDEO_PATH, exist_ok=True)

    env = make_env(render_mode="rgb_array")
    env = RecordVideo(env, VIDEO_PATH, episode_trigger=lambda e: True)

    state_dim, action_dim, max_action = get_env_dims(env)
    agent = TD3_Agent(state_dim, action_dim, max_action)

    agent.actor.load_state_dict(torch.load(MODEL_PATH))
    agent.actor.eval()

    setup_mlflow()

    with start_run("test"):

        for episode in range(MAX_VIDEOS):

            state, _ = env.reset()
            ep_reward = 0
            done = False
            terminated = False
            truncated = False

            while not done:

                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = agent.actor(state_tensor).detach().cpu().numpy()[0]

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                state = next_state
                ep_reward += reward
           
            if terminated and not truncated:
                noop = np.zeros(env.action_space.shape, dtype=np.float32)
                for _ in range(POST_LANDING_FRAMES):
                    _, _, _, truncated, _ = env.step(noop)
                    if truncated:
                        break

            print(f"Test episode {episode} | reward: {ep_reward}")
            mlflow.log_metric("test_reward", ep_reward, step=episode)

    env.close()