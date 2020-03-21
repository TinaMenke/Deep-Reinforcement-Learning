import numpy as np
import random


def parse_config(config):
    raise NotImplementedError


class Rectangle:
    def __init__(self, bottom_left, top_right):
        self.bottom_left = bottom_left
        self.top_right = top_right

    def intersects(self, other):
        """Check if two rectangles intersect"""
        return not (self.top_right[0] < other.bottom_left[0] or
                    self.bottom_left[0] > other.top_right[0] or
                    self.top_right[1] < other.bottom_left[1] or
                    self.bottom_left[1] > other.top_right[1])


def set_global_seeds(i):
    try:
        import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    myseed = i + 1000 * rank if i is not None else None
    try:
        import tensorflow as tf
        tf.set_random_seed(myseed)
    except ImportError:
        pass
    np.random.seed(myseed)
    random.seed(myseed)
