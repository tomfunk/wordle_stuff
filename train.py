import envs 
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
import torch as th

check_env(envs.WordleEnv2.create_default(), warn=True)

n_envs = 16
many_env = make_vec_env(envs.WordleEnv2.create_default, n_envs=n_envs)

policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[512, 512, 512, 512], vf=[512, 512, 512, 512])])

model = PPO(
    'MlpPolicy', many_env, policy_kwargs=policy_kwargs, verbose=1, n_steps=n_envs*100, batch_size=64,
    gamma=0.95,
    tensorboard_log="./ppo_wordle_tensorboard/"
)
model.learn(total_timesteps=5000000)
model.save('ppo')