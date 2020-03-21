# Deep Reinforcement Learning with Continuous Control in CARLA

Welcome to Projektpraktikum Maschinelles Lernen in SS2019. We are group3: Patrick Hemmer, Tina Menke and Mingyuan Zhou. 

This project is supervised by Karam Daaboul, Karl Kurzer, Svetlana Pavlitskaya and Abhishek Vivekanandan. 

Regarding the [CarRacing-v0](<https://gym.openai.com/envs/CarRacing-v0/>) enviroment in [OpenAI Gym](<https://github.com/openai/gym>), we conducted preprocessing of the input image in the environment, used different CNN network structures and conducted hyperparameter optimization in training. As for the reinforcement learning agent we used PPO2. The training and evaluation results can be found in the seminar paper. 
Instructions about the implementation can be founded [here](<https://ids-git.fzi.de/mzhou/carla_rllib/tree/develop-group3-test/car_racing_adaption>).

Regarding CARLA, we used PPO2. This is the main part of this Repository. Based on @svmuelle's work, we implemented preprocessing of the input images, using different CNN network structures and conducted hyperparameter optimization. 
Before trying out the things, it's highly recommended to get familiar with the [CARLA wrapper](<https://ids-git.fzi.de/svmuelle/carla_rllib>).

Our work is based on the CARLA0.9.6. You can read the [paper](https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3/paper.pdf) for more information and details.

## Installation

* Clone this repository to your local computer:

```konsole
$ git clone https://ids-git.fzi.de/mzhou/carla_rllib.git
```
* Install the required python packages (into your virtual environment):


```console
$ cd carla_rllib
$ pip install -r requirements.txt
```

* Add carla_rllib to your PYTHONPATH:

```console
$ export CARLA_RLLIB=your/path/to/carla_rllib
$ export PYTHONPATH=$CARLA_RLLIB
```

## What we did

Carla configuration: 
* Set the frame skipping to 0,2,4 in [config.json](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/carla_rllib/environments/carla_envs/config.py>)

CARLA wrapper: [carla_wrapper.py](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/carla_rllib/wrappers/carla_wrapper.py>)

* change camera's parameters in `_start()`
* set terminal condition of one episode in `_is_terminal()`

Carla environment: [carla_env.py](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/carla_rllib/environments/carla_envs/carla_env.py>)

* Define 6 spawn positions in Town05 in `get_random_spawn_point()`
* Adjust steering angle in [Line 217 & Line 221](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/carla_rllib/environments/carla_envs/carla_env.py#L217>)
* Add preprocessing of images from wrapper in `_get_obs()`,  which includes resizing, cropping and grey scaling
* Define the state information and retrieve them in `step()`
* Add autonomous driving cars in the environment in [Line109](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/carla_rllib/environments/carla_envs/carla_env.py#L109>)
* Add static obstacles in the environment in [Line173](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/carla_rllib/environments/carla_envs/carla_env.py#L173>)

Training&Evaluation&Visulization:

* Design different reward functions in [reward_functions.py](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/carla_rllib/utils/reward_functions.py>)
* Start train process in [train_model.py](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/carla_rllib/Training_Evaluation/train_model.py>)
* Create evaluation method in [evaluate.py](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/carla_rllib/Training_Evaluation/evaluate.py>)
* Plot a single curve from training data in `.csv ` file in [plot_data.py](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/carla_rllib/Plot/plot_data.py>)
* Plot multiple curves from tensorboard logs in [plot_tensorboard.py](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/carla_rllib/Plot/plot_tensorboard.py>)
* Define CNN, CNN small and CNN paper in [custom_policy.py](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/carla_rllib/Training_Evaluation/custom_policy.py>)


## Start your first training

It is possible to start the training in a quickest way, 

```console
$ cd carla_rillb/Training_Evaluation/
$ python train_model.py
```

This training is configurated as following:

high resolution segmented image(200*100) + reward function2 + CNN + 1e5timesteps + PPO2(default)

## Customize your own training

To customize your own agent's states and training configurations is also possible. You can design the RL experiment according to the following steps.

### Configurating the environment and agent

Configuration:  [config.json](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/carla_rllib/environments/carla_envs/config.py>)

* Configures the environment setup such as single- or multi-agent, frame skipping, map and more

Environment:  [carla_env.py](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/carla_rllib/environments/carla_envs/carla_env.py>)

* Customize spawn positions in Town05 in `get_random_spawn_point()`: Default: 6 positions in Town05
* Choose the preprocessing method and the size of input image in `_get_obs()`
  ***Note:*** !!! The image size after preprocessing should be compatible with observation_space in [Line230](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/carla_rllib/environments/carla_envs/carla_env.py#L230>)
* Calculate the reward  in `_calculate_reward()`, choose one or design one in [reward_functions.py](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/carla_rllib/utils/reward_functions.py>)
* Adjust reset information in `_get_reset()`
* Adjust the action/observation space to support stable baseline learning if necessary in `__init__()`
* From [Line109](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/carla_rllib/environments/carla_envs/carla_env.py#L109>): Add autonomous driving cars in the environment, default: disabled, just uncomment if you need
* From [Line173](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/carla_rllib/environments/carla_envs/carla_env.py#L173>): Add static obstacles in the environment, default: disabled, just uncomment if you need

Carla Wrapper:  [carla_wrapper.py](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/carla_rllib/wrappers/carla_wrapper.py>)

- Use either RGB camera or segmentation camera in `_start()` (BaseWrapper)
- Choose your terminal conditions in `_is_terminal()` (BaseWrapper)
- Adjust the controls if necessary in `step()` (ContinuousWrapper/DiscreteWrapper)
- Adjust the reset if necessary in `reset()` (ContinuousWrapper/DiscreteWrapper)

States:  [states.py](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/carla_rllib/wrappers/states.py>)

- Define a custom state that fits your learning goals
- Always adjust `_get_sensor_data()` or `_get_non_sensor_data` of the BaseWrapper if you change the state

### Training the model

Training:  [train_model.py](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/carla_rllib/Training_Evaluation/train_model.py>)

* Choose the CNN structure in Line17: policy, more detail to CNN in [custom_policy.py](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/carla_rllib/Training_Evaluation/custom_policy.py>)
* Configurate the PPO2 algorithm from Line48
* Loading the model is possible from Line53, default: commented

### Evaluating and visualizing results

Evaluation:  [evaluate_model.py](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/carla_rllib/Training_Evaluation/evaluate_model.py>)

* Define the timesteps and rounds you want to evaluate in Line49
  ***Note:*** For evaluation define respective spawn position in [carla_env.py](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/carla_rllib/environments/carla_envs/carla_env.py>). The evaluation function generates .csv file for the evaluation process at current path. Adaption or disabling is possible in [evaluate.py](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/carla_rllib/Training_Evaluation/evaluate.py>)

Plot data:

* The tensorboard log can be downloaded as `.csv` and plotted using [plot_data.py](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/carla_rllib/Plot/plot_data.py>) , adaption needed
* For comparison of more than one tensorboard log, follow the ways in [plot_tensorboard.py](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/carla_rllib/Plot/plot_tensorboard.py>), adaption needed
  ***Note:*** ! For plotting multiple tensorboard logs a special order of these logs is required. More details and how to do it can be found in [plot_tensorboard.py](<https://ids-git.fzi.de/mzhou/carla_rllib/blob/develop-group3-test/carla_rllib/Plot/plot_tensorboard.py>)

  



