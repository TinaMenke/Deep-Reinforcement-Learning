import numpy as np
import pandas as pd

##create a helper function to evaluate the model/agent
def evaluate(env, model, num_timesteps=1000, evaluate_rounds=1):
    """
    :param env: the carla_env
    :param model: the trained model
    :param num_steps: number of timesteps to evaluate it
    :param evaluate_rounds: how many rounds do you want evaluate

     !!! mean_reward_episode calculates mean reward for the last 500 episodes, usually it is enough,
     if you are aware that there are more than 500 episodes per round, just change it in mean_reward_episode
    """

    total_mean_reward_episode=0
    total_mean_reward_timestep=0

    for k in range(evaluate_rounds):

        episode_rewards=[0.0] #store the reward per episode
        timestep_rewards = [] #store the reward per action
        episodes=[1]
        timesteps=[]
        episode=1
        obs=env.reset()

        for i in range(num_timesteps):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            # reward update:
            timestep_rewards = np.append(timestep_rewards, rewards[0])
            timesteps=np.append(timesteps,i)
            episode_rewards[-1] += rewards

            if dones:
                episode+=1
                obs=env.reset()
                episode_rewards.append(0.0)
                episodes=np.append(episodes,episode)

        #compute mean reward for episode and action:
        mean_reward_episode=round(np.mean(episode_rewards[-500:]),1)
        mean_reward_timestep=round(np.mean(timestep_rewards),1)
        print("Evaluation Round %s:"%(k+1))
        print("Mean reward per episode:", mean_reward_episode, "Number of episodes:", len(episode_rewards))
        print("Mean reward per timestep:", mean_reward_timestep, "Number of timesteps:", num_steps)

        # !!!write into csv at current path
        timestep_rewards={"timesteps":timesteps,
                          "timestep_reward": timestep_rewards}
        episode_rewards={"episodes":episodes,
                         "episode_reward": episode_rewards}
        timestep_rewards = pd.DataFrame(data=timestep_rewards)
        timestep_rewards.to_csv('./reward_per_timestep_%s.csv'%(k+1))
        episode_rewards= pd.DataFrame(data=episode_rewards)
        episode_rewards.to_csv('./reward_per_episode_%s.csv'%(k+1))

        # update total average in defined rounds:
        total_mean_reward_episode +=mean_reward_episode
        total_mean_reward_timestep +=mean_reward_timestep
        env.reset()

    # calculate total average in several rounds:
    total_mean_reward_episode=total_mean_reward_episode/evaluate_rounds
    total_mean_reward_timestep=total_mean_reward_timestep/evaluate_rounds
    print("For the %s evaluation rounds:\n"%evaluate_rounds)
    print("Mean reward per episode:", total_mean_reward_episode)
    print("Mean reward per timestep:", total_mean_reward_timestep)
