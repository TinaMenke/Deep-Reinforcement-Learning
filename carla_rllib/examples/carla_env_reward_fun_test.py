import argparse
import os
import carla_rllib
import json
import gym
import numpy as np
from carla_rllib.environments.carla_envs.config import BaseConfig, parse_json
import pygame
from pygame.locals import K_UP
from pygame.locals import K_DOWN
from pygame.locals import K_LEFT
from pygame.locals import K_RIGHT

if __name__ == "__main__": 
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
    clock = pygame.time.Clock()

    try:
        obs = env.reset()
        a = 0
        steer_cache  = 0
        print("-----Carla Environment is running-----")
        y = 0
        # import time
        while True:
            milliseconds = clock.get_time()
            for event in pygame.event.get():
                keys = pygame.key.get_pressed()
                if keys[K_UP]:
                    a = 1.0
                elif keys[K_DOWN]:
                    a = -1.0
                else:
                    a = 0     

                steer_increment = 9e-4 * milliseconds
                if  keys[K_LEFT]:   
                    steer_cache -= steer_increment
                elif keys[K_RIGHT]:
                    steer_cache += steer_increment
                else:
                    steer_cache = 0.0
            
            s = min(0.7, max(-0.7, steer_cache))
            
            action = [s, a]
            obs, reward, done, info = env.step(action)
            clock.tick(5)

            y +=1
            if y%10 == 0:
                print("reward",reward)
                y = 0

            if done:
                env.reset()

    finally:
        env.close()
        print("-----Carla Environment is closed-----")
