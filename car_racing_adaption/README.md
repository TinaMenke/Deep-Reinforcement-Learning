# Adaption in CarRacing-v0

In order to try our adaptions in CarRacing-v0, firstly you need to clone or fork the [OpenAI Gym](<https://github.com/openai/gym>) and  [OpenAI Baselines](<https://github.com/openai/baselines>), to get the environment and implementation of RL algorithms. Please follow the steps in the two repositories in order to install it successfully on your computer.

## Implement preprocessing in CarRacing-v0

gym --> envs --> box2d --> add [car_racing_2.py](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/car_racing_adaption/car_racing_2.py>)

gym --> envs --> box2d --> replace "init.py" file with [init.py](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/car_racing_adaption/__init__.py>)

## Try different CNN structure in CarRacing-v0

baselines --> common --> replace models.py by [models.py](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/car_racing_adaption/models.py>)

## Training, Evaluatation

For training and evaluation please follow the step described in the original repository.

