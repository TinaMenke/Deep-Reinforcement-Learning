"""CARLA Gym Environment

This script provides single- and multi-agent environments for Reinforcement Learning in the Carla Simulator.

Class:
    * BaseEnv - environment base class
    * SAEnv - environment with one agent
    * MAEnv - environment with multiple agent
"""
import sys
import os
import glob
import random
import logging
import math
try:
    sys.path.append(glob.glob('%s/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        os.environ["CARLA_ROOT"],
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
import gym
import pygame
import numpy as np
from pygame.locals import K_ESCAPE
from gym.spaces import Box, Dict
from carla_rllib.wrappers.carla_wrapper import DiscreteWrapper
from carla_rllib.wrappers.carla_wrapper import ContinuousWrapper
from carla_rllib.utils import reward_functions
import cv2
from random import sample

class BaseEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels']}

    def __init__(self, config):

        print("-----Starting Environment-----")
        # Read config
        self._stable_baselines = config.stable_baselines
        self._agent_type = config.agent_type
        self._sync_mode = config.sync_mode
        self._delta_sec = config.delta_sec
        self._render_enabled = config.render
        self._host = config.host
        self._port = config.port
        self._map = config.map
        self._num_agents = config.num_agents

        # Declare remaining variables
        self.frame = None
        self.timeout = 2.0
        ###
        self.agents_list = []

        # Initialize client and get/load map
        try:
            client = carla.Client(self._host, self._port)
            self.world = client.get_world()
            if (self._map and self.world.get_map().name != self._map):
                client.set_timeout(100.0)
                print('Load map: %r.' % self._map)
                self.world = client.load_world(self._map)
            client.set_timeout(2.0)
            print("Connected to Carla Server")
        except:
            raise ConnectionError("Cannot connect to Carla Server!")

        # Enable/Disable Synchronous Mode
        self._settings = self.world.get_settings()
        if self._sync_mode:
            if not self._settings.synchronous_mode:
                _ = self.world.apply_settings(carla.WorldSettings(
                    no_rendering_mode=False,
                    synchronous_mode=True,
                    fixed_delta_seconds=self._delta_sec))
            print("Synchronous Mode enabled")
        else:
            if self._settings.synchronous_mode:
                _ = self.world.apply_settings(carla.WorldSettings(
                    no_rendering_mode=False,
                    synchronous_mode=False,
                    fixed_delta_seconds=None))
            print("Synchronous Mode disabled")

        # Create Agent(s)
        self._agents = []
        spawn_points = self.world.get_map().get_spawn_points()[
            :self._num_agents]
        if self._agent_type == "continuous":
            for n in range(self._num_agents):
                self._agents.append(ContinuousWrapper(self.world,
                                                      spawn_points[n],
                                                      self._render_enabled))
        elif self._agent_type == "discrete":
            for n in range(self._num_agents):
                self._agents.append(DiscreteWrapper(self.world,
                                                    spawn_points[n],
                                                    self._render_enabled))
            else:
                raise ValueError(
                    "Agent type not available. Adjust config and choose one from: ['continuous', 'discrete']")

        #-----------------------------Adding 100 autonomously driving cars:
        # vehicles_list = []
        # try:
        #
        #     self.world = client.get_world()
        #     blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        #     # blueprintsWalkers = world.get_blueprint_library().filter(args.filterw)
        #
        #     # if args.safe:
        #     #   blueprints = [x for x in blueprints if int(
        #     #        x.get_attribute('number_of_wheels')) == 4]
        #     #    blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
        #     #    blueprints = [
        #     #        x for x in blueprints if not x.id.endswith('carlacola')]
        #
        #     spawn_points_auto = self.world.get_map().get_spawn_points()
        #     # spawn_points_auto = [-205.7, 84.3, 0.2, 179]
        #     number_of_spawn_points = len(spawn_points_auto)
        #     number_of_vehicles = 100
        #     # number_of_vehicles = 300
        #     if number_of_vehicles < number_of_spawn_points:
        #         random.shuffle(spawn_points_auto)
        #     elif number_of_vehicles > number_of_spawn_points:
        #         msg = 'requested %d vehicles, but could only find %d spawn points'
        #         logging.warning(msg, number_of_vehicles,
        #                         number_of_spawn_points)
        #         number_of_vehicles = number_of_spawn_points
        #
        #     # @todo cannot import these directly.
        #     SpawnActor = carla.command.SpawnActor
        #     SetAutopilot = carla.command.SetAutopilot
        #     FutureActor = carla.command.FutureActor
        #
        #     # --------------
        #     # Spawn vehicles
        #     # --------------
        #     batch = []
        #     for n, transform in enumerate(spawn_points_auto):
        #         if n >= number_of_vehicles:
        #             break
        #         blueprint = random.choice(blueprints)
        #         if blueprint.has_attribute('color'):
        #             color = random.choice(
        #                 blueprint.get_attribute('color').recommended_values)
        #             blueprint.set_attribute('color', color)
        #         if blueprint.has_attribute('driver_id'):
        #             driver_id = random.choice(
        #                 blueprint.get_attribute('driver_id').recommended_values)
        #             blueprint.set_attribute('driver_id', driver_id)
        #         blueprint.set_attribute('role_name', 'autopilot')
        #         batch.append(SpawnActor(blueprint, transform).then(
        #             SetAutopilot(FutureActor, True)))
        #
        #     for response in client.apply_batch_sync(batch):
        #         if response.error:
        #             logging.error(response.error)
        #         else:
        #             vehicles_list.append(response.actor_id)
        #         self.agents_list = vehicles_list
        #
        # finally:
        #     print("100 autonomously driving cars were added to the environment")
#-------------------------------------------------------------------------------------------------------

#---------------------------------------Adding bins:------------------------------------------------------
            # Adding bins
        # try:
        #
        #     self.world = client.get_world()
        #     static_blueprints = self.world.get_blueprint_library().filter('static.prop.bin')
        #
        #     # static_spawn_points = [[188.1, -87.1, 0.2, 0], [83.2, 145.3, 0.2, 0], [123.4, -133.8, 0.2, 3], [-76.3, -34.0, 0.2, -178], [-264.9, 35.9, 0.2, 179]]
        #     static_spawn_points = [[51.63, 208.45, 0.2, 0], [-219.63, 84.16, 0.2, 0], [-216, -95, 0.2, 0],
        #                            [75, -185, 0.2, 3], [62, 144, 0.2, 0]]
        #     # static_spawn_points = self.world.get_map().get_spawn_points()
        #
        #     static_batch = []
        #
        #     for spawn_point in static_spawn_points:
        #         # if spawn_point > 100000:
        #         # break
        #         static_blueprint = random.choice(static_blueprints)
        #         if static_blueprint.has_attribute('role_name'):
        #             role_name = random.choice(
        #                 static_blueprint.get_attribute('role_name').recommended_values)
        #             static_blueprint.set_attribute('role_name', role_name)
        #         # blueprint.set_attribute('role_name', 'autopilot')
        #         transform = carla.Transform(carla.Location(x=spawn_point[0], y=spawn_point[1], z=spawn_point[2]),
        #                                     carla.Rotation(yaw=spawn_point[3]))
        #         static_batch.append(self.world.spawn_actor(static_blueprint, transform))
        #
        #     # for response in client.apply_batch_sync(static_batch):
        #     #    if response.error:
        #     #        logging.error(response.error)
        #     #    else:
        #     #        static_object_list.append(response.id)
        #     # self.agents_list.append(static_object_list)
        #
        # finally:
        #     print('Ziel ist es hier Mülleimer zu spawnen')
#----------------------------------------------------------------------------------------------------------------------

        # Baseline support
        self.action_space = None
        self.observation_space = None
        if (self._stable_baselines and
            self._agent_type == "continuous" and
                self._num_agents == 1):
            #change the steer angle ,  values before: ([-1.0, -1.0])
            low = np.array([-0.2, -1.0])
            #low = np.array([-0.5, -1.0])
            #low = np.array([-1.0, -1.0])
            #change the steer angle, values before:([1.0, 1.0])
            #high = np.array([0.5, 1.0])
            high = np.array([0.2, 1.0])
            #high = np.array([1.0, 1.0])
            self.action_space = Box(low, high, dtype=np.float32)
            print("observation_space")
            self.observation_space = Box(low=0, high=255,
                                         #shape=(84, 84, 3), #Angepasst von defaul 84x84 auf 256x256
                                         #shape=(120, 200, 3), #from Patrick's version
                                         shape=(100,200,3), #test
                                         #shape=(90,200,3),# test
                                         dtype=np.uint8)
            print("Baseline support enabled")
        else:
            print("Baseline support disabled\n" +
                  "(Note: Baselines are only supported for single agent with continuous control)")

        # Frame skipping
        if self._agent_type == "continuous":
            self._frame_skip = config.frame_skip
            #self._frame_skip = 4
            print("Frame skipping enabled")
        else:
            self._frame_skip = 0
            print("Frame skipping disabled")

        # Spawn agents

        # Hacky workaround to solve waiting time when spawned:
        # Unreal Engine simulates starting the car and shifting gears,
        # so you are not able to apply controls for ~2s when an agent is spawned
        for agent in self._agents:
            agent._vehicle.apply_control(
                carla.VehicleControl(manual_gear_shift=True, gear=1))
        self.start_frame = self.world.tick()
        for agent in self._agents:
            agent._vehicle.apply_control(
                carla.VehicleControl(manual_gear_shift=False))
        if self._render_enabled:
            self.render()
        print("Agent(s) spawned")

    def step(self, action):
        """Run one timestep of the environment's dynamics

        Single-Agent Environment:
        Accept a list of actions and return a tuple (observations, reward, terminal and info)

        Multi-Agent Environment:
        Accept a list of actions stored in a dictionary for each agent and
        return dictionaries accordingly (observations, rewards, terminals and infos)

        Parameters:
        ----------
            action: list (dict)
                action(s) provided by the agent(s)

        Returns:
        ----------
            obs_dict: list (dict)
                observation(s) of the current environment
            reward_dict: float (dict)
                reward(s) returned after previous actions
            done_dict: bool (dict)
                whether the episode of the agent(s) is(are) done
            info_dict: dict
                contains auxiliary diagnostic information
        """

        # Set step and initialize reward_dict
        reward_dict = dict()
        for agent in self._agents:
            if self._num_agents == 1:
                agent.step(action)
            else:
                agent.step(action[agent.id])
            reward_dict[agent.id] = 0

        # Run step, update state, calculate reward and check for dead agent
        done = 0
        for _ in range(self._frame_skip + 1):

        # -----------------Adapted by us ------------------------------------

            #agent.state.velocity_limit=5.0
            agent.state.traveled_distance=0.0
            #print("Velocity limit: ")
            #print(agent.state.speed_limit)
            #print('DISTANCE TO CENTERLINE')
            #print(agent.state.distance_to_center_line)

            agent.state.previous_position = agent.state.position
            #print('PREVIOUS POSITION')
            #print(agent.state.previous_position)

            #print('PREVIOUS LANE INVASION')
            agent.state.previous_lane_invasion = agent.state.lane_invasion
            #print(agent.state.previous_lane_invasion)

            agent.state.previous_velocity = agent.state.velocity
            #print('PREVIOUS VELOCITY')
            #print(agent.state.previous_velocity)
            
            agent.state.previous_collision = agent.state.collision
            if (agent.state.previous_collision):
                agent.state.previous_collision_binary = 1
            else:
                agent.state.previous_collision_binary = 0
                #print('PREVIOUS COLLISION')
                #print(agent.state.previous_collision_binary)

        #--------------------------------------------------------------------
            self.frame = self.world.tick()
            if self._render_enabled:
                self.render()
            for agent in self._agents:
                done += agent.update_state(self.frame,
                                           self.start_frame,
                                           self.timeout)
        #-----------------Adapted by us ------------------------------------
                #print('CURRENT LANE INVASION')
                #print(agent.state.lane_invasion)
                #print('CURRENT VELOCITY')
                #print(agent.state.velocity)
                agent.state.velocity_diff = agent.state.velocity - agent.state.previous_velocity
                #print('VELOCITY DIFFERENCE')
                #print(agent.state.velocity_diff)
                if (agent.state.collision):
                    agent.state.collision_binary = 1
                    #assign lane type
                    #self._agent.state.collision.l
                else:
                    agent.state.collision_binary = 0
                    #print('CURRENT COLLISION')
                    #print(agent.state.collision_binary)
                    #print(agent.sta)
                agent.state.collision_binary_diff = agent.state.collision_binary - agent.state.previous_collision_binary
                #print('COLLISION DIFFERENCE')
                #print(agent.state.collision_binary_diff)
                #Has a lane invasion occured?
                agent.state.lane_invasion_diff = agent.state.lane_invasion - agent.state.previous_lane_invasion
                #print('TRAVELLED DISTANCE')
                agent.state.position_diff_x=agent.state.position[0]-agent.state.previous_position[0]
                agent.state.position_diff_y = agent.state.position[1] - agent.state.previous_position[1]
                agent.state.traveled_distance = (agent.state.position_diff_x**2+agent.state.position_diff_y**2)**0.5
#-----------------------------------------------------------Printing life information-------------------------------
                #print('traveled distance:')
                #print(agent.state.traveled_distance)
                #print('run time')
                #print(agent.state.elapsed_ticks)
#---------------------------------------------------------------------------------------------------------------

#-------------------------------calculate distance to nearest vehicles----------------------------------------------
                # min_distance = 1000
                # if len(self.agents_list) > 1:
                #     for _id in self.agents_list:
                #         other_actor = self.world.get_actor(_id)
                #         location = other_actor.get_location()
                #         distance = math.sqrt((location.x - agent.state.position[0]) ** 2 + (
                #                     location.y - agent.state.position[
                #                 1]) ** 2)  # + (location.z - agent.state.position[2])**2)
                #         distance_x = location.x - agent.state.position[0]
                #         distance_y = location.y - agent.state.position[1]
                #         if distance < min_distance:
                #             min_distance = distance
                #             # min_distance_x = distance_x
                #             # min_distance_y = distance_y
                #             agent.state.closest_vehicle = other_actor
                #             agent.state.closest_vehicle_distance = distance
                #             # print(min_distance)
                #     if (min_distance < 2):
                #         agent.state.close_vehicle_binary = 1
                #     else:
                #         agent.state.close_vehicle_binary = 0
                #     #print(agent.state.close_vehicle_binary)
                #     # print("distance x: " + str(min_distance_x))
                #     # print("distance y: " + str(min_distance_y))

#----------------calculate the velocity part of reward function for reward_function 8 ---------------------------------------------
                # agent.state.reward_velocity=0.0
                # if agent.state.velocity<=agent.state.velocity_limit:
                #     agent.state.reward_velocity=agent.state.velocity/agent.state.velocity_limit
                # else:
                #     agent.state.reward_velocity=3-0.4*agent.state.velocity
                #     if agent.state.reward_velocity<=-1:
                #         agent.state.reward_velocity=-1
#-----------------------------------------------------------------------------------------------------------------------------
                #print('DICT NACH UPDATE')
                #print(reward_dict[agent.id])
                reward_dict[agent.id] += self._calculate_reward(agent)
            if done:
                break

        # Retrieve observations, terminal and info
        obs_dict = self._get_obs()
        done_dict = self._is_done()
        info_dict = self._get_info()

        if self._num_agents == 1:
            return obs_dict["Agent_1"], reward_dict["Agent_1"], done_dict["Agent_1"], info_dict["Agent_1"]
        else:
            return obs_dict, reward_dict, done_dict, info_dict

    def reset(self):
        """Reset the state of the environment and return initial observations

        Returns:
        ---------- 
            obs_dict: dict
                the initial observations
        """
        # Set reset
        for agent in self._agents:
            reset = self._get_reset(agent)
            agent.reset(reset)

        # Run reset and update state
        self.start_frame = self.world.tick()
        if self._render_enabled:
            self.render()
        for agent in self._agents:
            agent.update_state(self.start_frame,
                               self.start_frame,
                               self.timeout)

        # Retrieve observations
        obs_dict = self._get_obs()

        if self._num_agents == 1:
            return obs_dict["Agent_1"]
        else:
            return obs_dict

    def render(self, mode='human'):
        """Render display"""
        for agent in self._agents:
            agent.render()

    def _get_obs(self):
        """Return current observations

        ---Note---
        Pull information out of state
        """

        # Extract observations for agents
        obs_dict = dict()
        for agent in self._agents:
            obs_dict[agent.id] = agent.state.image

#------------------------------------------------------------- IMAGE PREPROCESSING --------------------------------------------------------------
            seg_image = obs_dict[agent.id]
            #cv2.imwrite('original_input_image.png',seg_image)
            seg_image = cv2.resize(seg_image, (200,150)) #(width, height)
            #cv2.imwrite('resize_input_image.png', seg_image)
            seg_image_crop = seg_image[30:150, 0:0+200] #[height, width], original one(from Patrick)

            seg_image_crop_new=seg_image[50:150,0:200] # test1, cut off more trees
            seg_image_crop_new_2=seg_image[70:150,0:200] # test2

            #-------changing to gray image:---------
            # seg_image_crop=cv2.cvtColor(seg_image_crop, cv2.COLOR_BGR2GRAY)
            # seg_image_crop=seg_image_crop[:,:,np.newaxis]
            #print((seg_image_crop.shape))
            #------------------------------------------------------------

            #-----------Tesing: store the image and test the processing------------------------------------------
            #cv2.imwrite('.png', agent.state.image)
            #-----------------------------------------------------------------------------------------------------------
            obs_dict[agent.id] = seg_image_crop_new
#------------------------------------------------------------- IMAGE PREPROCESSING --------------------------------------------------------------
        return obs_dict

    def _calculate_reward(self, agent):
        """Return the current reward"""
        # TODO: Calculate reward based on agent's state
        #10*(agent.state.traveled_distance) + 
        #reward = 5 * agent.state.traveled_distance + 0.05 * agent.state.velocity_diff - 2 * agent.state.collision_binary_diff - 5 * agent.state.lane_invasion_diff
        reward = reward_functions.reward_2(agent)

        return reward

    def _is_done(self):
        """Return the current terminal condition"""
        done_dict = dict()
        for agent in self._agents:
            done_dict[agent.id] = agent.state.terminal
        return done_dict

    def _get_info(self):
        """Return current information"""
        # TODO: add something to print out
        info_dict = dict()
        for agent in self._agents:
            info_dict[agent.id] = dict(Info="Store whatever you want")
        return info_dict

# -------------------------------------------------------- Methode für Random Spawn Points -------------------------------------------------------
    def get_random_spawn_point(self):

        liste_spawn_points = [[66, -186.8, 0.2, 0],
                             [46.0, 208.7, 0.2, 0],
                             [44.0, 145.4, 0.2, 0],
                             [47.3, -146.2, 0.2, 3],
                             [-205.2, -95.1, 0.2, -178],
                             [-205.7, 84.3, 0.2, 179]
                                ]
        return sample(liste_spawn_points, 1)
# -------------------------------------------------------- Methode für Random Spawn Points -------------------------------------------------------

    def _get_reset(self, agent):
        """Return reset information for an agent

        ---Note---
        Implement your reset information here and
        adjust wrapper reset function if necessary
        """
        random_spawn_point = self.get_random_spawn_point()
        #print(random_spawn_point[0])
        #agent.state.traveled_distance=0.0

        if self._agent_type == "continuous":
            reset = dict(
                Agent_1=dict(position=(random_spawn_point[0][0], random_spawn_point[0][1], random_spawn_point[0][2]), #Hier random choice aus Spawn Point Liste [default: (46.1, 208.9)]
                             yaw=random_spawn_point[0][3],
                             steer=0,
                             acceleration=-1.0),
                Agent_2=dict(position=(56.1, 208.9),
                             yaw=0,
                             steer=0,
                             acceleration=-1.0)
            )
        else:
            reset = dict(
                Agent_1=dict(position=(0, 0),
                             yaw=0,
                             velocity=(1, 0),
                             acceleration=(0, 0)),
                Agent_2=dict(position=(-10, 0),
                             yaw=0,
                             velocity=(1, 0),
                             acceleration=(0, 0))
            )
        return reset[agent.id]

    def set_phase(self):
        """Set up curriculum phase"""
        raise NotImplementedError

    def close(self):
        """Destroy agent and reset world settings"""
        for agent in self._agents:
            agent.destroy()
        self.world.apply_settings(self._settings)
        pygame.quit()
