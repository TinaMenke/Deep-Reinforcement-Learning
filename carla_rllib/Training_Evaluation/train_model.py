import os
import carla_rllib
import numpy as np
import pandas as pd
import gym
import argparse
import json
from carla_rllib.environments.carla_envs.config import parse_json
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG, PPO2
from carla_rllib.Training_Evaluation.custom_policy import CustomPolicyCNNsmall, CustomPolicyCNNpaper, CustomPolicyCNNNvidia


# define the training CNN here:
policy="CnnPolicy"

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

#----------------------Define training configuration--------------------------------------------------------
    model = PPO2(policy=policy,env=env, verbose=1, tensorboard_log="./training/")
    model.learn(total_timesteps=100000)
    model.save("./training/model1")

#------------------------------enjoy the model, no need for training-------------------------------------------------------
    # model = PPO2.load(".pkl")
    # y=0
    # obs=env.reset()
    # while True:
    #     action, _states=model.predict(obs)
    #     obs,rewards,dones,info=env.step(action)
    #     y+=1
    #     if y%10==0:
    #         #print('Reward every tenth action', rewards)
    #         y=0
    #     env.render()
#-----------------------------------------------------------------------------------------------------
finally:
    env.close()



