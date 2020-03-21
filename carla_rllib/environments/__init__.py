from gym.envs.registration import register

register(id='CarlaBaseEnv-v0',
         entry_point='carla_rllib.environments.carla_envs:BaseEnv',
         kwargs={'config': None}
         )
