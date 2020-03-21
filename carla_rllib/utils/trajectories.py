"""Trajectory Planer

This script allows the user to
(i) calculate trajectories with the jerk minimization algorithm and
(ii) run trajectories in the CARLA simulator.

Functions:
    * angle - calculates the angle between two vectors
    * check_action_feasibility - checks the feasibility of a trajectory
    * transform_coordinates - rotates trajectory by given angle
    * test_trajectory - plans and runs a trajectory in carla
    * calculate_coefficients - calculates the coefficients for jerk minimization
    * jerk_minimization - the main function of the script: calculates trajectory
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
import numpy as np
import math
import time


def angle(v1, v2):
    """Calculate angle between two vectors"""
    return np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def rotate_vector(v, theta):
    """Rotate 2D-vector"""
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return R.dot(v)


def check_action_feasibility():
    raise NotImplementedError


def transform_coordinates(trajectory, theta):
    """Rotate trajectory coordinates to vehicle heading

    Parameters
    ----------
    trajectory: list of 3x2 numpy-arrays
        trajectory including position, velocity and acceleration vectors
    theta: float
        rotation angle (radian) of agent wrt the global coordinate system

    Returns
    ----------
    trajectory: list of 3x2 numpy-arrays
        trajectory including position, velocity and acceleration vectors
    """
    origin = trajectory[0][0]

    for p in trajectory:
        p[0] = rotate_vector(p[0] - origin, theta) + origin

    return trajectory


def test_trajectory(actor, velocity, acceleration, deltaV, deltaD,
                    T=1.4, t=0.02, delay=0.05, setback=False):
    """Compute and follow a trajectory in CARLA

    Parameters
    ----------
    actor: carla.Actor
        Agent in CARLA simulation
    velocity: {list, tuple, numpy-array} of floats (vx, vy)
    acceleration: {list, tuple, numpy-array} of floats (ax, ay)
    deltaV: float
        Change of velocity
    deltaD: float
        Change of lateral position
    T: float
        Timeframe for trajectory planning
    t: float
        Controllers update rate
    delay: float
        delay of action for visual control when following trajectory
    setback: bool
        True if agent is set back to starting position after following trajectory
    """

    if not isinstance(actor, carla.Vehicle):
        raise TypeError("Actor needs to be a carla.Actor.")
    # Get position and orientation of agent
    wp = actor.get_transform()
    location = wp.location
    rotation = wp.rotation.yaw

    # Calculate trajectory
    traj = jerk_minimization(
        (location.x, location.y), velocity, acceleration, deltaV, deltaD, T, t
    )

    # Transform trajectory based on agents orientation
    theta = math.radians(rotation)
    trans_traj = transform_coordinates(traj, theta)

    reference_heading = np.array([1, 0])
    # Follow the trajectory
    for p in trans_traj:
        rad = angle(reference_heading, p[1])
        yaw = np.rad2deg(rad) * np.sign(deltaD)
        actor.set_transform(carla.Transform(
            carla.Location(p[0][0], p[0][1]),
            carla.Rotation(yaw=rotation + yaw)
        ))
        if delay:
            time.sleep(delay)
    if setback:
        rad = angle(reference_heading, trans_traj[0][1])
        yaw = np.rad2deg(rad) * np.sign(deltaD)
        actor.set_transform(carla.Transform(
            carla.Location(trans_traj[0][0][0], trans_traj[0][0][1]),
            carla.Rotation(yaw=rotation + yaw)
        ))


def calculate_coefficients(start, end, T):
    """Calculate coefficients for jerk minimization

    Parameters
    ----------
    start: array-like (list, triple, numpy-array)
        start conditions consisting of position, velocity and acceleration (one dimensional)
    end: array-like (list, triple, numpy-array)
        end conditions consisting of position, velocity and acceleration (one dimensional)
    T: int
        Timeframe the trajectory is planned for

    Returns
    ----------
    coeff: numpy-array
        coefficients (a0-a5) of the jerk minimization trajectory    
    """
    A = np.array([
        [T**3,   T**4,    T**5],
        [3*T**2, 4*T**3,  5*T**4],
        [6*T,   12*T**2, 20*T**3],
    ])

    a_0, a_1, a_2 = start[0], start[1], start[2] / 2.0
    c_0 = a_0 + a_1 * T + a_2 * T**2
    c_1 = a_1 + 2 * a_2 * T
    c_2 = 2 * a_2

    B = np.array([
        end[0] - c_0,
        end[1] - c_1,
        end[2] - c_2
    ])

    a_3_4_5 = np.linalg.solve(A, B)
    coeff = np.concatenate((np.array([a_0, a_1, a_2]), a_3_4_5))

    return coeff


def jerk_minimization(position, velocity, acceleration,
                      deltaV, deltaD, T=1.4, t=0.02):
    """Calculate trajectory based on jerk minimization

    Parameters
    ----------
    position: {list, tuple, numpy-array} of floats (x, y)
    velocity: {list, tuple, numpy-array} of floats (vx, vy)
    acceleration: {list, tuple, numpy-array} of floats (ax, ay)
    deltaV: float
        Change of velocity
    deltaD: float
        Change of lateral position
    T: float
        Timeframe for trajectory planning
    t: float
        Controllers update rate

    Returns
    ----------
    trajectory: list of 3x2 numpy-arrays
        trajectory including position, velocity and acceleration vectors
    """
    # Initial/End conditions
    initS = (position[0], velocity[0], acceleration[0])
    initD = (position[1], velocity[1], acceleration[1])

    # TODO: find proper boundary condition for dS
    dS = ((2 * initS[1] + deltaV) / 2) * T

    endS = (initS[0] + dS, initS[1] + deltaV, 0)
    endD = (initD[0] + deltaD, 0, 0)

    # Calculate coefficients
    coeff_S = calculate_coefficients(initS, endS, T)
    coeff_D = calculate_coefficients(initD, endD, T)

    # Calculate trajectory
    coeff = np.array([coeff_S, coeff_D])

    trajectory = [np.array([[1, i, i**2, i**3, i**4, i**5],
                            [0, 1, 2 * i, 3 * i**2, 4 * i**3, 5 * i**4],
                            [0, 0, 2, 6 * i, 12 * i**2, 20 * i**3]]).dot(coeff.T)
                  for i in np.arange(0, T + t, t)]

    return trajectory


if __name__ == "__main__":

    trajectory = jerk_minimization(
        (0, 0), (15, 0), (0, 0), 20, 2.75, T=1.4, t=0.02)

    import matplotlib.pyplot as plt

    # s
    x1 = [n[0][0] for n in trajectory]
    y1 = [n[0][1] for n in trajectory]
    plt.subplot(3, 1, 1)
    plt.plot(x1, y1, '-')
    plt.xlabel('s')
    plt.ylabel('d')

    # v
    x2 = [n[1][0] for n in trajectory]
    y2 = [n[1][1] for n in trajectory]
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(0, 1.42, 0.02), x2, '-')

    plt.subplot(3, 1, 2)
    plt.plot(np.arange(0, 1.42, 0.02), y2, '-')
    plt.xlabel('time')
    plt.ylabel('v')

    # a
    x3 = [n[2][0] for n in trajectory]
    y3 = [n[2][1] for n in trajectory]
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(0, 1.42, 0.02), x3, '-')

    plt.subplot(3, 1, 3)
    plt.plot(np.arange(0, 1.42, 0.02), y3, '-')
    plt.xlabel('time')
    plt.ylabel('a')

    plt.show()
