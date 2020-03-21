#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a RL agent to control the ego vehicle
"""
import time
from threading import Thread
import cv2
import numpy as np
import json

try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')

import carla

from srunner.challenge.autoagents.autonomous_agent import AutonomousAgent, Track
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
#
from carla_rllib.examples.custom_policy import CustomPolicyCNNsmall, CustomPolicyCNNpaper, CustomPolicyCNNNvidia
from stable_baselines import PPO2


class PrakAgent(AutonomousAgent):

    """
    RL agent to control the ego vehicle
    """
    RENDER_WIDTH = 1280
    RENDER_HEIGHT = 720
    current_control = None
    agent_engaged = False

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        print(path_to_conf_file)
        self.track = Track.ALL_SENSORS_HDMAP_WAYPOINTS

        # Sensor setup
        config = json.load(open(path_to_conf_file))
        self._cam_config = config["camera"]
        self._debug = config["debug"]["enabled"]
        self._policy_checkpoint = config["policy_checkpoint"]

        # Agent setup
        if not self._debug:
            hero_actor = None
            for actor in CarlaDataProvider.get_world().get_actors():
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                    hero_actor = actor
                    break
            if hero_actor:
                pass
            self._agent =PPO2.load(self._policy_checkpoint)

            self._surface = None
            self._start_render()
        else:
            self._input_cam = config["debug"]["input_camera"]
            self._resize = config["debug"]["resize"]

            self.agent_engaged = False
            self.current_control = carla.VehicleControl()
            self.current_control.steer = 0.0
            self.current_control.throttle = 0.0
            self.current_control.brake = 1.0
            self.current_control.hand_brake = False
            self._hic = HumanInterface(self, self._cam_config["width"], self._cam_config["height"],
                                       self._input_cam, self._resize)
            self._thread = Thread(target=self._hic.run)
            self._thread.start()

    def _start_render(self):
        """ """
        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode(
            (PrakAgent.RENDER_WIDTH, PrakAgent.RENDER_HEIGHT),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Prak Agent")
        
    def sensors(self):
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        """
        sensors = [
            {'type': 'sensor.camera.' + str(self._cam_config["type"]), 'id': 'InputCamera',
             'x': self._cam_config["x"], 'y': self._cam_config["y"], 'z': self._cam_config["z"],
             'roll': self._cam_config["roll"], 'pitch': self._cam_config["pitch"], 'yaw': self._cam_config["yaw"],
             'width': self._cam_config["width"], 'height': self._cam_config["height"], 'fov': self._cam_config["fov"]
            },
            {'type': 'sensor.camera.rgb', 'id': 'RenderCamera',
             'x': -8.0, 'y': 0.0, 'z': 6.0,
             'roll': 0.0, 'pitch': -30.0, 'yaw': 0.0,
             'width': 1280, 'height': 720, 'fov': 100
             }
        ]

        return sensors

    def run_step(self, camera_data, timestamp):
        """
        Execute one step of navigation.
        """
        if self._debug:
            self.agent_engaged = True
            return self.current_control
        else:
            input_data = camera_data['InputCamera'][1]
            input_data = input_data[:, :, :3]
            input_data = input_data[:, :, ::-1]

            seg_image = cv2.resize(input_data, (200,150)) #(width, height)
            #cv2.imwrite('resize_input_image.png', seg_image)
            seg_image_crop_new=seg_image[50:150,0:200] #test, cut off more trees


            #steer, acceleration = self._policy.predict(input_data) # TODO: use your policy to predict actions
            action, _ = self._agent.predict(seg_image_crop_new)
            steer=action[0]
            acceleration=action[1]
            control = carla.VehicleControl()
            control.steer = float(steer)
            if acceleration >= 0.0:
                print("acceleration")
                control.throttle = float(acceleration)
                control.brake = 0.0
            else:
                print("brake")
                control.throttle = 0.0
                control.brake = float(acceleration)
            
            render_data = camera_data['RenderCamera'][1]
            render_data = render_data[:, :, :3]
            render_data = render_data[:, :, ::-1]
            self._surface = pygame.surfarray.make_surface(
                render_data.swapaxes(0, 1))
            if self._surface is not None:
                self._display.blit(self._surface, (0, 0))
            pygame.display.flip()

            print("[Timestamp: {}]".format(timestamp))

            return control

    def destroy(self):
        """
        Cleanup
        """
        if self._debug:
            self._hic.quit = True
            self._thread.join()
        else:
            pygame.quit()


class HumanInterface(object):

    """
    Class to control a vehicle manually for debugging purposes
    """

    def __init__(self, parent, width, height, input_cam=False, resize=[0, 0]):
        self.quit = False
        self._parent = parent
        self._input_cam = input_cam
        self._resize = resize
        if not self._input_cam:
            self._width = 1280
            self._height = 720
        elif self._resize[0] and self._resize[1] and self._input_cam:
            self._width = self._resize[0]
            self._height = self._resize[1]
        else:
           self._width = width
           self._height = height

        self._throttle_delta = 0.05
        self._steering_delta = 0.01
        self._surface = None

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode(
            (self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Prak Agent")

    def run(self):
        """
        Run the GUI
        """
        while not self._parent.agent_engaged and not self.quit:
            time.sleep(0.5)

        controller = KeyboardControl()
        while not self.quit:
            self._clock.tick_busy_loop(20)
            controller.parse_events(self._parent.current_control, self._clock)
            # Process events
            pygame.event.pump()

            # process sensor data
            input_data = self._parent.sensor_interface.get_data()
            if self._input_cam:
                image = input_data['InputCamera'][1]
            else:
                image = input_data['RenderCamera'][1]
            image = image[:, :, :3]
            image = image[:, :, ::-1]
            if self._resize:
                image = cv2.resize(image,
                                   dsize=(self._width, self._height),
                                   interpolation=cv2.INTER_CUBIC)

            # display image
            self._surface = pygame.surfarray.make_surface(
                image.swapaxes(0, 1))
            if self._surface is not None:
                self._display.blit(self._surface, (0, 0))
            pygame.display.flip()

        pygame.quit()


class KeyboardControl(object):

    """
    Manual keyboard control for debugging purposes
    """

    def __init__(self):
        """
        Init
        """
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0

    def parse_events(self, control, clock):
        """
        Parse the keyboard events and set the vehicle controls accordingly
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

            self._parse_vehicle_keys(
                pygame.key.get_pressed(), clock.get_time())
            control.steer = self._control.steer
            control.throttle = self._control.throttle
            control.brake = self._control.brake
            control.hand_brake = self._control.hand_brake

    def _parse_vehicle_keys(self, keys, milliseconds):
        """
        Calculate new vehicle controls based on input keys
        """
        self._control.throttle = 0.6 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 15.0 * 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0

        self._steer_cache = min(0.95, max(-0.95, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]
