import envs 
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

check_env(envs.WordleEnv2.create_default(), warn=True)

n_envs = 16
many_env = make_vec_env(envs.WordleEnv2.create_default, n_envs=n_envs)

model = PPO.load('ppo', env=many_env, tensorboard_log="./ppo_wordle_tensorboard/")

model.learn(total_timesteps=5000000, reset_num_timesteps=False)
model.save('ppo')