import argparse
import os
import carla_rllib
import json
import gym
import numpy as np
from carla_rllib.environments.carla_envs.config import BaseConfig, parse_json

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

try:
    env = gym.make("CarlaBaseEnv-v0", config=configs[0])
    obs = env.reset()

    t = 0
    print("-----Carla Environment is running-----")
    while True:

        # Calculate/Predict Actions
        t += 0.15
        s = 0.3 * np.sin(t)
        a = 0.8

        if env._num_agents == 1:
            action = [s, a]  # Single agent (with continuous control)
        else:
            action = dict(Agent_1=[s, a],  # Two agents (with continuous control)
                          Agent_2=[s, a])

        # Make step in environment
        obs, reward, done, info = env.step(action)

        print(env._agents[0].state)

        # Reset if done
        if env._num_agents == 1 and done:
            env.reset()
            t = 0
        elif env._num_agents > 1 and any(d for d in done.values()):
            env.reset()
            t = 0

finally:
    env.close()
    print("-----Carla Environment is closed-----")
