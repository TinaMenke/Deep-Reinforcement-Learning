import os
import carla_rllib
import numpy as np
import gym
import argparse
import json
from carla_rllib.environments.carla_envs.config import parse_json
from stable_baselines.ddpg.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG


argparser = argparse.ArgumentParser(
    description='CARLA RLLIB ENV')
package_path, _ = os.path.split(os.path.abspath(carla_rllib.__file__))
argparser.add_argument(
    '-c', '--config',
    metavar='CONFIG',
    default=os.path.join(package_path +
                         "/config.json"),
    type=str,
    help='Path to configuration file (default: root of the package -> carla_rllib)')
args = argparser.parse_args()
config_json = json.load(open(args.config))
configs = parse_json(config_json)
print("-----Configuration-----")
print(configs[0])

env = gym.make("CarlaBaseEnv-v0", config=configs[0])

try:
    env = DummyVecEnv([lambda: env])

    # the noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
    model = DDPG(CnnPolicy, env, verbose=1, render=True, param_noise=param_noise,
                 action_noise=action_noise, random_exploration=0.8)
    model.learn(total_timesteps=10000)

finally:
    env.close()
