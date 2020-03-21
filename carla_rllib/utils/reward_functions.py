"""Reward functions

This script provides examples of reward functions for reinforcement learning.

Functions:
    * reward_1 - (tbd)
    * reward_2 - (tbd)
    ...
"""
import numpy as np


def reward_1(distance_to_center_line, delta_heading, current_speed, target_speed,
             max_reward=1.0, min_reward=-1.0, weights=[0.6, 0.4, 0.5]):
    """Returns the reward based on distance to center line and delta heading"""
    # Distance reward 1
    distance_reward = max(
        max_reward - (distance_to_center_line / 0.6)**3, min_reward)

    # Distance reward 2 (DQN)
    # distance_reward = max(2.0 - (distance_to_center_line + 1.0, 8)**2, -2.0)

    # Distance reward 3
    # a maximal reward
    # b penalty clipping factor
    # c zeros
    # max(-(a / c**2) * (distance_to_center_line / 0.7 * 3.5)**2 + a, b);

    # Heading reward
    heading_reward = max(1 - delta_heading / (1/3 * 90), min_reward)

    # # Speed reward
    # if current_speed <= target_speed:
    #     speed_reward = max(-1, 1 + 10 * np.log(current_speed / target_speed))
    # else:
    #     speed_reward = max(2.0 - (current_speed / target_speed)**4, -1)

    # Combination
    reward = weights[0] * distance_reward + \
        weights[1] * heading_reward
    # weights[2] * speed_reward

    return reward


def reward_2(agent): # design idea from CARLA paper + adaption
    
    reward = 5 * agent.state.traveled_distance + 0.05 * agent.state.velocity_diff - 2 * agent.state.collision_binary_diff - \
             5 * agent.state.lane_invasion_diff - 2 * agent.state.distance_to_center_line
    return reward

def reward_2_1(agent): #add 'delta_heading' into the reward_2
    
    reward = 5 * agent.state.traveled_distance + 0.05 * agent.state.velocity_diff - 2 * agent.state.collision_binary_diff - \
             5 * agent.state.lane_invasion_diff - 2 * agent.state.distance_to_center_line - 0.3*agent.state.delta_heading
    return reward


def reward_3(agent): # change factors based on reward_2
    reward = 3 * agent.state.traveled_distance + 0.02 * agent.state.velocity_diff - 2 * agent.state.collision_binary_diff - \
             5 * agent.state.lane_invasion_diff - 2 * agent.state.distance_to_center_line
    return reward

def reward_5(agent): # adding the information about distance to nearst object
    reward = 5 * agent.state.traveled_distance + 0.05 * agent.state.velocity_diff - 2 * agent.state.collision_binary_diff - \
             5 * agent.state.lane_invasion_diff - 2 * agent.state.distance_to_center_line-5 * agent.state.close_vehicle_binary
    return reward

def reward_6(agent): # adding the speed limit
    reward = 3 * agent.state.traveled_distance + 0.03 * agent.state.velocity_diff - 2 * agent.state.collision_binary_diff - \
             5 * agent.state.lane_invasion_diff - 2 * agent.state.distance_to_center_line-7 * agent.state.close_vehicle_binary - \
             2*(agent.state.velocity-agent.state.speed_limit)
    return reward

def reward_7(agent): # reward normalization version-1
    w_velocity=2.0
    w_distance_center_line=1.0
    w_collision=-5.0
    w_lane_invasion = -10.0
    reward=w_velocity*agent.state.reward_velocity+\
           w_distance_center_line*(1.8-agent.state.distance_to_center_line)/1.8+\
           w_collision*agent.state.collision_binary_diff+\
           w_lane_invasion*agent.state.lane_invasion_diff
    return reward

def reward_8(agent): # reward normalization version-2
    #---------------------------------Information of the weight-------------------------
    w_velocity=3.0
    w_distance_center_line= 5.0
    w_collision= -1.0
    w_lane_invasion = -0.5
    #w_survive=5.0
    #w_traveled_distance=2.0
    w_delta_heading=3.0
    #---------------------------Precalculation for the reward---------------------

    #aim_traveled_distance=300
    #terminal_elapsed_ticks=2000
    maximum_delta_heading=10
    #reward_survive=(agent.state.elapsed_ticks**2)*((terminal_elapsed_ticks)**(-2))
    #reward_traveled_distance=(1/aim_traveled_distance)*agent.state.traveled_distance
    if agent.state.terminal==True:
        reward_terminal=-200
    else:
        reward_terminal= 1
    if agent.state.velocity==0:
        reward_heading= -0.5
        reward_distance_to_centerline= -0.5
        reward_terminal=0
    else: 
        if agent.state.delta_heading<=20:
            reward_heading= -(1/maximum_delta_heading)*(agent.state.delta_heading) + 1
        else:
            reward_heading = -1
        reward_distance_to_centerline = (1.8-agent.state.distance_to_center_line)/1.8

    #---------------------Print and check error----------------------------------
    # print('velocity')
    # print(agent.state.velocity)
    # print(' ')
    # print('agent.state.reward_velocity')
    # print(agent.state.reward_velocity)
    # print(' ')
    #print('reward disitance to center line')
    #print(reward_distance_to_centerline)
    #print('delta_heading')
    #print(agent.state.delta_heading)
    # print(' ')
    # print('reward heading')
    # print(reward_heading)
    # print(' ')
    #print('reward_lane_invasion')
    #print(agent.state.lane_invasion_diff)
    #print(agent.state.lane_invasion)
    #print('traveled distance')
    #print(reward_traveled_distance)

    #-------------------reward calculation---------------------------------------------
    reward=w_velocity * agent.state.reward_velocity + \
           w_distance_center_line * reward_distance_to_centerline + \
           w_collision * agent.state.collision_binary_diff + \
           w_lane_invasion * agent.state.lane_invasion_diff + reward_terminal +\
           w_delta_heading * reward_heading 
           #w_traveled_distance * reward_traveled_distance
                 
    #print('REWARD')
    #print(reward)
    return reward


