import envs
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
import logging
import agents
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from tqdm import trange

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

# Assuming you have an MLP policy
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.network = nn.Sequential(
                nn.Linear(env.observation_space.shape[0], 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
        )
        # Creating separate output for each dimension of the MultiDiscrete action space
        self.action_heads = nn.ModuleList([
            nn.Linear(64, n) for n in env.action_space.nvec
        ])
        
    def forward(self, x):
        x = self.network(x)
        action_probs = [head(x) for head in self.action_heads]
        return action_probs


class BehavioralCloning:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.model = Policy()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, total_timesteps):
        for i in trange(total_timesteps):
            obs, _ = self.env.reset()
            done = False
            while not done:
                # Let the agent perform an action
                word = self.agent.act(obs)
                action = self.env._translate_word(word)
                assert len(action) == len(self.model.action_heads), "Mismatch between action dimensions!"
                # Train the model to predict each dimension of the agent's action
                self.optimizer.zero_grad()
                predicted_actions = self.model(torch.tensor(obs).float().unsqueeze(0))
                
                loss = sum(self.criterion(predicted_actions[i], torch.tensor([action[i]])) 
                           for i in range(len(action)))
                loss.backward()
                self.optimizer.step()

                # Step the environment
                obs, _, done, _, _ = self.env.step(action)

env = envs.WordleEnv2.create_default()
check_env(env, warn=True)
agent = agents.WordleAgent1(list(env.valid_words))


logging.info("cloning")
cloner = BehavioralCloning(env, agent)
cloner.train(total_timesteps=10000)

# Then we can continue with the RL training
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    gamma=0.95,
    tensorboard_log="./ppo_wordle_tensorboard/",
)
model.policy.set_parameters(list(cloner.model.parameters()), exact_match=False)
logging.info("learning")
model.learn(
    total_timesteps=100000,
    tb_log_name="tutored",
    eval_freq=1000,
    n_eval_episodes=10,
)
model.save("ppo_trained")