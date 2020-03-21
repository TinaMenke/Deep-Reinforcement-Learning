#!/usr/bin/env python

"""
This module provides an Agent class to evaluate a policy
"""

from __future__ import print_function

import carla
import json
from srunner.challenge.autoagents.AutonomousAgent import AutonomousAgent, Track
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


class PrakAgent(AutonomousAgent):

    """
    Autonomous agent to evaluate \'Projektpraktika\'
    """

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self.track = Track.ALL_SENSORS_HDMAP_WAYPOINTS

        # Setup sensors
        config = json.load(open(path_to_conf_file))
        self._cam_config = config["camera"]
        self._policy_path = config["policy"]

        # Setup actor
        self._actor = None
        for actor in CarlaDataProvider.get_world().get_actors():
            if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                self._actor = actor
                break

        # Setup policy
        # self._policy = load_policy(self._policy_path)

    def sensors(self):
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:
        """

        sensors = [
            {'type': 'sensor.camera.semantic_segmentation', 'id': 'MainCamera',
             'x': self._cam_config['x'], 'y': self._cam_config['y'], 'z': self._cam_config['z'],
             'roll': self._cam_config['roll'], 'pitch': self._cam_config['pitch'], 'yaw': self._cam_config['yaw'],
             'width': self._cam_config['width'], 'height': self._cam_config['height'], 'fov': self._cam_config['fov']},
        ]

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """

        steer, acceleration = self._policy.predict(input_data)

        control = carla.VehicleControl()
        control.steer = steer
        control.hand_brake = False
        if acceleration >= 0.0:
            control.throttle = acceleration
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = -acceleration

        print("[Timestamp: {}]".format(timestamp))

        return control
