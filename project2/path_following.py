#!/usr/bin/env python
import numpy as np
from utils import decompose, distance, wrap_to_pi


class PathFollower(object):
    """Implements path following using feedback control.
    """
    def __init__(self, pos, vel, time, goal_threshold=0.2):
        """Initialize the PathFollower.

        Args:
            pos (ndarray): Positions size (n,2)
            vel (ndarray): Velocities size (n,2)
            time (ndarray): Time since start size (n,)
            goal_threshold (float): Threshold in meters for when the goal is
                deemed reached. Default 0.2.
        """
        super(PathFollower, self).__init__()
        self._goal_reached = False
        self._pos = pos
        self._vel = vel
        self._time = time
        self._goal_threshold = goal_threshold
        self._curvature = map(self._get_curvature, range(len(time)))
        # Start indexing from 1, because velocity at index 0 is 0
        self._idx = 1

    def _get_curvature(self, idx):
        """Get the curvature of the path at the given index.

        Args:
            idx (int): Index of point in path at which to get curvature

        Returns:
            float: Curvature at the given index
        """
        try:
            x_vel, y_vel = self._vel[idx]
            x_vel_n, y_vel_n = self._vel[idx+1]
            dt = self._time[idx+1] - self._time[idx]
            a = np.sqrt(x_vel**2 + y_vel**2)
            an = np.sqrt(x_vel_n**2 + y_vel_n**2)
            d = np.array([x_vel/a, y_vel/a])
            n = np.array([-y_vel/a, x_vel/a])
            d_n = np.array([x_vel_n/an, y_vel_n/an])
            d_dot = (d_n-d)/dt
            curvature = (d_dot.dot(n))/a
            return curvature
        except IndexError:
            return 0.

    def _get_nearest(self, pose, window=2):
        """Get the index of the path point closest to the given pose. Updates
        the current running index and whether the goal has been reached.

        Args:
            pose (ndarray): Pose with size (3,) or a position with size (2,)
            window (int, optional): Window size for how many indeces are
                considered ahead the current running index. Default 2.

        Returns:
            int: The nearest index
        """
        min_idx = self._idx
        max_idx = np.minimum(self._pos.shape[0], self._idx + window)
        idx = range(min_idx, max_idx)
        nearest = idx[np.argmin(distance(pose[:2], self._pos[idx]))]
        self._idx = nearest

        # Check if goal has been reached
        if distance(pose[:2], self._pos[-1]) <= self._goal_threshold:
            self._goal_reached = True

        return nearest

    def _get_desired_pose(self, idx):
        """Get the desired pose at the given path index. The desired heading
        is calculated from the velocity.

        Args:
            idx (int): Index of point in path

        Returns:
            ndarray: The desired pose at the given index
        """
        x, y = self._pos[idx]
        _, direction = decompose(self._vel[idx])
        theta = wrap_to_pi(np.arctan2(direction[1], direction[0]))
        return x, y, theta

    def _get_transform(self, idx):
        """Get a matrix to transform the error into the coordinate system of
        the desired pose.

        Args:
            idx (int): The index of the desired point in path

        Returns:
            ndarray: The transformation matrix, size (3,3)
        """
        heading = np.arctan2(self._vel[idx, 1], self._vel[idx, 0])
        sin_h = np.sin(heading)
        cos_h = np.cos(heading)
        return np.array([[cos_h, sin_h, 0],
                         [-sin_h, cos_h, 0],
                         [0, 0, 1]])

    @property
    def goal_reached(self):
        """Determine whether the goal has been reached.

        Returns:
            bool: Returns True if the goal has been reached
        """
        return self._goal_reached

    def get_control(self, pose):
        """Calculate the values for the feedback control.

        Args:
            pose (ndarray): The current pose of the mobile base

        Returns:
            tuple: Linear and angular velocity
        """
        # Get the nearest point's index on the path
        nearest = self._get_nearest(pose)
        # Calculate the desired pose on the nearest point on the path
        d_pose = self._get_desired_pose(nearest)

        # Calculate the difference between x, y and theta
        transformation = self._get_transform(nearest)
        e = transformation.dot(pose-d_pose)

        # Limit the difference of the orientation between -pi and pi
        e[2] = wrap_to_pi(e[2])

        # Get the linear velocity control from the nearest point on the path
        linear, _ = decompose(self._vel[nearest])

        # Pole-placement
        a = -2
        b = -3
        # Gains
        k2 = a*b
        k3 = -(a+b)

        # Angular velocity control calculation
        omega1 = -k2*e[1]-k3*e[2]
        s_dot = (linear*np.cos(e[2]))/(e[1]*self._curvature[nearest]-1)
        angular = omega1 + self._curvature[nearest]*s_dot

        return linear, angular
