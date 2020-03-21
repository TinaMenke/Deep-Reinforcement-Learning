"""CARLA Wrapper

This script provides actor wrappers with continuous and discrete action control for the CARLA simulator.

Classes:
    * BaseWrapper - wrapper base class
    * ContinuousWrapper - actor with continuous action control
    * DiscreteWrapper - actor with discrete action control
"""
import sys
import os
import glob
try:
    sys.path.append(glob.glob('%s/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        os.environ["CARLA_ROOT"],
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
from carla import ColorConverter as cc
import pygame
import queue
import numpy as np
import argparse
from carla_rllib.wrappers.sensors import SegmentationSensor, CollisionSensor, LaneInvasionSensor, RenderCamera, RgbSensor
from carla_rllib.wrappers.states import BaseState


VEHICLE_MODELS = ['vehicle.audi.a2',
                  'vehicle.audi.tt',
                  'vehicle.carlamotors.carlacola',
                  'vehicle.citroen.c3',
                  'vehicle.dodge_charger.police',
                  'vehicle.jeep.wrangler_rubicon',
                  'vehicle.yamaha.yzf',
                  'vehicle.nissan.patrol',
                  'vehicle.gazelle.omafiets',
                  'vehicle.ford.mustang',
                  'vehicle.bmw.isetta',
                  'vehicle.audi.etron',
                  'vehicle.bmw.grandtourer',
                  'vehicle.mercedes-benz.coupe',
                  'vehicle.toyota.prius',
                  'vehicle.diamondback.century',
                  'vehicle.tesla.model3',
                  'vehicle.seat.leon',
                  'vehicle.lincoln.mkz2017',
                  'vehicle.kawasaki.ninja',
                  'vehicle.volkswagen.t2',
                  'vehicle.nissan.micra',
                  'vehicle.chevrolet.impala',
                  'vehicle.mini.cooperst']

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720


class BaseWrapper(object):

    ID = 1

    def __init__(self, world, spawn_point, render=False):

        self.id = "Agent_" + str(BaseWrapper.ID)
        self._world = world
        self._map = self._world.get_map()
        self._carla_id = None
        self._vehicle = None
        self._sensors = []
        self._queues = []
        self._render_enabled = render
        self.state = BaseState()
        self._simulate_physics = True

        self._start(spawn_point, VEHICLE_MODELS[1], self.id)
        if self._render_enabled:
            self._start_render()

        BaseWrapper.ID += 1
        print(self.id + " was spawned in " + str(self._map.name) +
              " with CARLA_ID " + str(self._carla_id))

    def _start(self, spawn_point, actor_model=None, actor_name=None):
        """Spawn actor and initialize sensors"""
        # Get (random) blueprint
        if actor_model:
            blueprint = self._world.get_blueprint_library().find(actor_model)
        else:
            blueprint = np.random.choice(
                self._world.get_blueprint_library().filter("vehicle.*"))
        if actor_name:
            blueprint.set_attribute('role_name', actor_name)
        if blueprint.has_attribute('color'):
            color = np.random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')

        # Spawn vehicle
        self._vehicle = self._world.spawn_actor(blueprint, spawn_point)
        self._carla_id = self._vehicle.id

        #Set up sensors
        self._sensors.append(SegmentationSensor(self._vehicle,
                                                width=1024, height=720,# new adption
                                                #width=84, height=84,
                                                orientation=[1.5, 1.6, -30, 0]) # new adption
                                                #orientation=[1.5,1.6,-10,0] )# adjust camera's angle
                                                #orientation=[1.5, 1.4, -30, 0])
                             )
#####-----------------------add the rbg sensor:---------------------------------
        # self._sensors.append(RgbSensor(self._vehicle,
        #                                #width=1024, height=720,
        #                                width=84, height=84,
        #                                orientation=[1.5, 1.4, -30, 0]))
#-------------------------------------------------------------------------------------------------
        self._sensors.append(CollisionSensor(self._vehicle))
        self._sensors.append(LaneInvasionSensor(self._vehicle))




    def step(self, action):
        """Set one actor action

        Parameters:
        ----------
        action: object
            an action provided by the agent

        """
        raise NotImplementedError

    def reset(self, reset):
        """Set one actor reset

        Parameters:
        ----------
        reset: dict
            contains reset information specific to the learning goals

        """
        raise NotImplementedError

    def render(self):
        """Render the carla simulator frame"""
        self._render_camera.render(self.display)
        pygame.display.flip()

    def update_state(self, frame, start_frame, timeout):
        """Update the agent's current state

        ---Note---
        Implement your terminal conditions
        """

        # Retrieve sensor data
        self._get_sensor_data(frame, timeout)

        # Calculate non-sensor data
        self._get_non_sensor_data(frame, start_frame)

        # Check terminal conditions
        if self._is_terminal():
            self.state.terminal = True

        # Disable simulation physics if terminal
        if self.state.terminal:
            self._togglePhysics()
            return True
        else:
            return False

    def _get_sensor_data(self, frame, timeout):
        """Retrieve sensor data"""
        data = [s.retrieve_data(frame, timeout)
                for s in self._sensors]
        self.state.image = data[0]
        self.state.collision = data[1]
        self.state.lane_invasion = data[2]

    def _get_non_sensor_data(self, frame, start_frame):
        """Calculate non-sensor data"""

        # Position
        transformation = self._vehicle.get_transform()
        location = transformation.location
        self.state.position = [np.around(location.x, 2),
                               np.around(location.y, 2)]

        # Velocity
        velocity = self._vehicle.get_velocity()
        self.state.velocity = np.around(np.sqrt(velocity.x**2 +
                                                velocity.y**2 +
                                                velocity.z**2), 2)

        # Acceleration
        acceleration = self._vehicle.get_acceleration()
        self.state.acceleration = np.around(np.sqrt(acceleration.x**2 +
                                                    acceleration.y**2 +
                                                    acceleration.z**2), 2)

        # Heading wrt lane direction
        nearest_wp = self._map.get_waypoint(location,
                                            project_to_road=True)
        vehicle_heading = transformation.rotation.yaw
        wp_heading = nearest_wp.transform.rotation.yaw
        #----------------------------------------
        # print('vehicle_headings')
        # print(vehicle_heading)
        # print('wp_heading')
        # print(wp_heading)
        #-----------------------------------
        delta_heading = np.abs(vehicle_heading - wp_heading)
        if delta_heading < 180:
            self.state.delta_heading = delta_heading
        elif delta_heading > 180 and delta_heading <= 360:
            self.state.delta_heading = 360 - delta_heading
        else:
            self.state.delta_heading = delta_heading - 360
#-----------------------------------------------
        #print('delta_heading')
        #print(self.state.delta_heading)
#--------------------------------------------------------------
        # Opposite lane check and
        # Distance to center line of nearest (permitted) lane
        distance = np.sqrt(
            (location.x - nearest_wp.transform.location.x) ** 2 +
            (location.y - nearest_wp.transform.location.y) ** 2
        )

       # self.state.distance_to_center_line = np.around(nearest_wp.lane_width - distance, 2)

        if self.state.delta_heading > 90:
            self.state.opposite_lane = True
            self.state.distance_to_center_line = np.around(
                nearest_wp.lane_width - distance, 2)
        else:
            self.state.opposite_lane = False
            self.state.distance_to_center_line = np.around(distance, 2) #original
            #self.state.distance_to_center_line = np.around(nearest_wp.lane_width - distance, 2)

        # Lane type check
        wp = self._map.get_waypoint(location)
        self.state.lane_type = wp.lane_type.name

        # Lane change check
        self.state.lane_change = wp.lane_change.name

        # Junction check
        self.junction = wp.is_junction

        # Elapsed ticks
        self.state.elapsed_ticks = frame - start_frame

        # Speed limit
        speed_limit = self._vehicle.get_speed_limit()
        if speed_limit:
            self.state.speed_limit = speed_limit
        else:
            self.state.speed_limit = None

    def _is_terminal(self):
        """Check terminal conditions"""
        # TODO: Adjust terminal conditions
        if (self.state.collision
            or self.state.distance_to_center_line > 1.8 # original:1.8
                #or self.state.traveled_distance>300
                or self.state.elapsed_ticks >= 3000):# original: 1000
            return True
        else:
            return False

    def _start_render(self):
        """Start rendering camera"""
        self.display = pygame.display.set_mode(
            (SCREEN_WIDTH, SCREEN_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        self._render_camera = RenderCamera(self._vehicle)

    def _togglePhysics(self):
        self._simulate_physics = not self._simulate_physics
        self._vehicle.set_simulate_physics(self._simulate_physics)

    def destroy(self):
        """Destroy agent and sensors"""
        actors = [s.sensor for s in self._sensors] + [self._vehicle]
        for actor in actors:
            if actor is not None:
                actor.destroy()
        if self._render_enabled:
            self._render_camera.sensor.destroy()


class ContinuousWrapper(BaseWrapper):

    def __init__(self, world, spawn_point, render=False):
        super(ContinuousWrapper, self).__init__(world, spawn_point, render)

    def step(self, action):
        """Apply steering and throttle/brake control

        action = [steer, acceleration]

        """
        control = self._vehicle.get_control()
        control.manual_gear_shift = False
        control.reverse = False
        control.hand_brake = False
        control.steer = float(action[0])

        if action[1] >= 0:
            control.brake = 0
            control.throttle = float(action[1])
        else:
            control.throttle = 0
            control.brake = -float(action[1])
        self._vehicle.apply_control(control)

    def reset(self, reset):
        """Reset position and controls as well as sensors and state

        reset = dict(
            position=[x, y],
            yaw=rotation,
            steer=steer,
            acceleration=acceleration
        )

        """
        # position
        transform = carla.Transform(
            carla.Location(reset["position"][0], reset["position"][1]),
            carla.Rotation(yaw=reset["yaw"])
        )
        self._vehicle.set_transform(transform)

        # controls
        self._vehicle.set_velocity(carla.Vector3D(0, 0, 0))
        self._vehicle.set_angular_velocity(carla.Vector3D(0, 0, 0))
        control = self._vehicle.get_control()
        control.steer = reset["steer"]
        if reset["acceleration"] >= 0:
            control.brake = 0
            control.throttle = reset["acceleration"]
        else:
            control.throttle = 0
            control.brake = -reset["acceleration"]
        self._vehicle.apply_control(control)

        # sensors and state
        self._sensors[1].reset()
        self._sensors[2].reset()
        self.state.terminal = False
        self.state.position = (reset["position"][0],
                               reset["position"][1])

        # Enable simulation physics if disabled
        if not self._simulate_physics:
            self._togglePhysics()


class DiscreteWrapper(BaseWrapper):

    def __init__(self, world, spawn_point, render=False):
        super(DiscreteWrapper, self).__init__(world, spawn_point, render)

    def step(self, action):
        """Apply discrete transformation/teleportation

        action = [x, y, rotation]

        """
        print("step - discrete")
        transform = carla.Transform(
            carla.Location(action[0], action[1]),
            carla.Rotation(yaw=action[2])
        )
        self._vehicle.set_transform(transform)

    def reset(self, reset):
        """Reset position as well as sensors and state

        reset = dict(
            position=[x, y],
            yaw=rotation
        )

        """

        # Reset position
        transform = carla.Transform(
            carla.Location(reset["position"][0], reset["position"][1]),
            carla.Rotation(yaw=reset["yaw"])
        )
        self._vehicle.set_transform(transform)
        self._vehicle.set_velocity(carla.Vector3D(0, 0, 0))
        self._vehicle.set_angular_velocity(carla.Vector3D(0, 0, 0))

        # sensors and state
        self._sensors[1].reset()
        self._sensors[2].reset()
        self.state.terminal = False
        self.state.position = (reset["position"][0],
                               reset["position"][1])

        # Enable simulation physics if disabled
        if not self._simulate_physics:
            self._togglePhysics()
