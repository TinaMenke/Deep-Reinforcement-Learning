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
import queue
import numpy as np
import time
import math
import collections
import pygame
import weakref
from carla import ColorConverter as cc


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


class RgbSensor(object):
    def __init__(self, parent_actor, width=84, height=84,
                 orientation=[-5.5, 2.8, -15, 0]):
        self.sensor = None
        self._parent = parent_actor
        self._width = width
        self._height = height
        self._queue = queue.Queue()

        # Initialize Sensor and start to listen
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self._width))
        bp.set_attribute('image_size_y', str(self._height))
        self.sensor = world.spawn_actor(bp, carla.Transform(
            carla.Location(x=orientation[0], z=orientation[1]),
            carla.Rotation(pitch=orientation[2], yaw=orientation[3])),
            attach_to=self._parent)
        self.sensor.listen(self._queue.put)
        # change view angle:
        #bp.set_attribute('fov', str(120.0))

    def retrieve_data(self, frame, timeout):
        while True:
            image = self._queue.get(timeout=timeout)
            if image.frame == frame:
                image = self._preprocess_data(image)
                return image

    def _preprocess_data(self, image):
        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array


class SegmentationSensor(object):
    def __init__(self, parent_actor, width=84, height=84,
                 orientation=[-5.5, 2.8, -15, 0], palette='citypalette'):
        self.sensor = None
        self._parent = parent_actor
        self._width = width
        self._height = height
        self._queue = queue.Queue()

        if palette == 'citypalette':
            self.spec = cc.CityScapesPalette
        else:
            self.spec = cc.Raw

        # Initialize Sensor and start to listen
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        bp.set_attribute('image_size_x', str(self._width))
        bp.set_attribute('image_size_y', str(self._height))

        # change the angle of camera(part of preproessing)
        bp.set_attribute('fov',str(120.0))

        self.sensor = world.spawn_actor(bp, carla.Transform(
            carla.Location(x=orientation[0], z=orientation[1]),
            carla.Rotation(pitch=orientation[2], yaw=orientation[3])),
            attach_to=self._parent)
        self.sensor.listen(self._queue.put)

    def retrieve_data(self, frame, timeout):
        while True:
            image = self._queue.get(timeout=timeout)
            if image.frame == frame:
                image = self._preprocess_data(image)
                return image

    def _preprocess_data(self, image):
        image.convert(self.spec)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array


class LidarSensor(object):
    def __init__(self, parent_actor, width=84, height=84,
                 orientation=[-5.5, 2.8, -15, 0]):
        self.sensor = None
        self._parent = parent_actor
        self._width = width
        self._height = height
        self._queue = queue.Queue()

        # Initialize Sensor and start to listen
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        bp.set_attribute('range', '5000')
        self.sensor = world.spawn_actor(bp, carla.Transform(
            carla.Location(x=orientation[0], z=orientation[1]),
            carla.Rotation(pitch=orientation[2], yaw=orientation[3])),
            attach_to=self._parent)
        self.sensor.listen(self._queue.put)

    def retrieve_data(self, frame, timeout):
        while True:
            image = self._queue.get(timeout=timeout)
            if image.frame == frame:
                image = self._preprocess_data(image)
                return image

    def _preprocess_data(self, image):
        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 3), 3))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(self._width, self._height) / 100.0
        lidar_data += (0.5 * self._width, 0.5 * self._height)
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (self._width, self._height, 3)
        lidar_img = np.zeros((lidar_img_size), dtype=int)
        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
        return lidar_img


class DepthSensor(object):
    def __init__(self, parent_actor, width=84, height=84,
                 orientation=[-5.5, 2.8, -15, 0], kind=None):
        self.sensor = None
        self._parent = parent_actor
        self._width = width
        self._height = height
        self._queue = queue.Queue()

        if kind == 'grayscale':
            self.spec = cc.Depth
        else:
            self.spec = cc.LogarithmicDepth

        # Initialize Sensor and start to listen
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.camera.depth')
        bp.set_attribute('image_size_x', str(self._width))
        bp.set_attribute('image_size_y', str(self._height))
        self.sensor = world.spawn_actor(bp, carla.Transform(
            carla.Location(x=orientation[0], z=orientation[1]),
            carla.Rotation(pitch=orientation[2], yaw=orientation[3])),
            attach_to=self._parent)
        self.sensor.listen(self._queue.put)

    def retrieve_data(self, frame, timeout):
        while True:
            image = self._queue.get(timeout=timeout)
            if image.frame == frame:
                image = self._preprocess_data(image)
                return image

    def _preprocess_data(self, image):
        image.convert(self.spec)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array


class GnssSensor(object):
    def __init__(self, parent_actor, orientation=[1.0, 2.8]):
        self.sensor = None
        self._parent = parent_actor
        self._queue = queue.Queue()

        # Initialize Sensor and start to listen
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(
            carla.Location(x=orientation[0], z=orientation[1])),
            attach_to=self._parent)
        self.sensor.listen(self._queue.put)

    def retrieve_data(self, frame, timeout):
        while True:
            data = self._queue.get(timeout=timeout)
            if data.frame == frame:
                data = self._preprocess_data(data)
                return data

    def _preprocess_data(self, data):
        return (data.latitude, data.longitude)


class LaneInvasionSensor(object):
    def __init__(self, parent_actor):

        # This sensor is a work in progress, currently very limited.

        self.sensor = None
        self._parent = parent_actor
        self.history = []
        self._last_event_frame = 0
        self._crossed_lane_counter = 0

        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    def retrieve_data(self, frame, timeout):
        return self._crossed_lane_counter

    def reset(self):
        self._crossed_lane_counter = 0
        self._last_event_frame = 0

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        if event.crossed_lane_markings:
            # Current workaround to count and store crossed lane markings:
            # Prevent crossed lane markings from being counted several times
            # by ignoring all subsequent events within a period of 20 frames.
            if (event.frame - self._last_event_frame > 20):
                self._last_event_frame = event.frame
                self._crossed_lane_counter += 1
                self.history.append(event.crossed_lane_markings[0].type.name)


class CollisionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self._collision = None
        self.other_actor = None

        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))

    def retrieve_data(self, frame, timeout):
        return self._collision

    def reset(self):
        self._collision = False
        self.other_actor = None

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.other_actor = get_actor_display_name(event.other_actor)
        self._collision = False if self.other_actor == "Sidewalk" else True


class RenderCamera(object):
    def __init__(self, parent_actor, width=1280, height=720,
                 orientation=[-5.5, 2.8, -15, 0]):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self._width = width
        self._height = height

        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self._width))
        bp.set_attribute('image_size_y', str(self._height))
        self.sensor = world.spawn_actor(bp, carla.Transform(
            carla.Location(x=orientation[0], z=orientation[1]),
            carla.Rotation(pitch=orientation[2], yaw=orientation[3])),
            attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda image: RenderCamera._on_rgb_image(weak_self, image))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _on_rgb_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))


class CameraManager(object):
    def __init__(self, parent_actor, width=84, height=84,
                 orientation=[-5.5, 2.8, -15, 0]):
        self.sensor = None
        self._index = None
        self._parent = parent_actor
        self._width = width
        self._height = height
        self.out = None

        self._camera_position = carla.Transform(carla.Location(x=orientation[0], z=orientation[1]),
                                                carla.Rotation(pitch=orientation[2], yaw=orientation[3]))

        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'rgb'],
            # ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'depth'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'segmentation'],
            ['sensor.lidar.ray_cast', None, 'lidar']]

        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(self._width))
                bp.set_attribute('image_size_y', str(self._height))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '5000')
            item.append(bp)

        self.set_sensor(0)

    def set_sensor(self, sensor_idx):
        self._index = sensor_idx
        if self.sensor:
            self.sensor.destroy()
        self.sensor = self._parent.get_world().spawn_actor(
            self.sensors[sensor_idx][-1],
            self._camera_position,
            attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid
        # circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: self._parse_image(weak_self, image))

    def get_images(self):
        images = {}
        for s in range(len(self.sensors)):
            self.set_sensor(s)
            images[self.sensors[s][-2]] = self.out
        return images

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self._index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self._width, self._height) / 100.0
            lidar_data += (0.5 * self._width, 0.5 * self._height)
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self._width, self._height, 3)
            lidar_img = np.zeros((lidar_img_size), dtype=int)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.out = lidar_img
        else:
            image.convert(self.sensors[self._index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.out = array
