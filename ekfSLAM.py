#!/usr/bin/env python
import numpy as np
from numpy import sin, cos, sqrt, pi
import time


class EKFSLAM:
    def __init__(self, state_vector_size, control_size, measurement_size):
        self.state_vector = np.zeros((state_vector_size, 1))
        self.cov_matrix = 1000. * np.identity(state_vector_size)
        self.q = np.diag(np.array([0.1, 0.01]))
        self.R = np.diag(np.array([0.1, 0.01]))
        self.motion_j_state = np.identity(state_vector_size)
        self.motion_j_noise = np.zeros((state_vector_size, control_size))
        self.obs_j_state = np.zeros((measurement_size, 5))
        self.Q = np.zeros((state_vector_size, state_vector_size))
        self.time_stamp = 0
        self.gt = np.zeros((state_vector_size, 1))

    def initialize_position(self, x, y, theta):
        print("Robot position initialized")
        self.state_vector[0, 0] = x
        self.state_vector[1, 0] = y
        self.state_vector[2, 0] = theta
        self.state_vector[3, 0] = None
        self.state_vector[4, 0] = None
        self.state_vector[5, 0] = None
        self.state_vector[6, 0] = None
        self.state_vector[7, 0] = None
        self.state_vector[8, 0] = None
        self.state_vector[9, 0] = None
        self.state_vector[10, 0] = None
        self.state_vector[11, 0] = None
        self.state_vector[12, 0] = None


    def update_gt(self, x, y, theta):
        self.gt[0, 0] = x
        self.gt[1, 0] = y
        self.gt[2, 0] = theta

    def predict(self, msg):
        self.propagate_state(msg)
        self.calculate_cov()
        return self.state_vector, self.cov_matrix

    def update(self, msg):
        """
        Update the ekf based on the range and bearing measurements

        Args:
            msg: marker message

        """
        state_v = self.state_vector
        cov_matrix = self.cov_matrix
        btheta = None
        angle = None
        if len(msg.markers[:]) != 0:
            for i in range(len(msg.markers[:])):
                x = state_v[0, 0]
                y = state_v[1, 0]
                theta = state_v[2, 0]

                # Position referred to the beacons
                bx = msg.markers[i].pose.position.x
                by = msg.markers[i].pose.position.y

                # quaternions to radians
                quat1 = 2 * (msg.markers[i].pose.orientation.w * msg.markers[i].pose.orientation.z +
                             msg.markers[i].pose.orientation.x * msg.markers[i].pose.orientation.y)
                quat2 = 1 - (2 * (msg.markers[i].pose.orientation.y * msg.markers[i].pose.orientation.y +
                                  msg.markers[i].pose.orientation.z * msg.markers[i].pose.orientation.z))
                btheta = np.arctan2(quat1, quat2)

                # Id of the current beacon
                ide = msg.markers[i].ids[0]

                # Calculate the position of the detected beacon
                r = np.sqrt((bx**2) + (by**2))
                angle = np.arctan2(by, bx)

                # Limit angle
                if angle < -pi:
                    angle += 2*pi
                elif angle > pi:
                    angle -= 2*pi

                # Check if the beacon is observed before
                if np.isnan(state_v[2+ide+i, 0]):
                    Lx = x + r*cos(angle)
                    state_v[2+ide+i, 0] = Lx
                    self.state_vector[2+ide+i, 0] = Lx
                else:
                    Lx = state_v[2+ide+i, 0]

                # Check if the beacon is observed before
                if np.isnan(state_v[3+ide+i, 0]):
                    Ly = y + r*sin(angle)
                    state_v[3+ide+i, 0] = Ly
                    self.state_vector[3+ide+i, 0] = Ly
                else:
                    Ly = state_v[3+ide+i, 0]

                # Calculate distance
                delta = np.array([[(Lx - x)],
                                  [(Ly - y)]])

                q = (delta.T).dot(delta)

                # Build matrix h(Xi)=zi
                z = np.array([[np.sqrt((by)**2 + (bx)**2)],
                              [np.arctan2(by, bx)]])

                h = np.array([[np.sqrt(q[0, 0])],
                              [np.arctan2(delta[1, 0], delta[0, 0]) - theta]])

                # Calculate Jacobian H
                self.observation_jacobian_state_vector(delta, q[0, 0])  # , Lx, Ly, x, y

                # Calculate observation Jacobian Ha
                F = np.zeros((5, 13))
                F[0, 0] = 1
                F[1, 1] = 1
                F[2, 2] = 1
                F[3, 3+2*ide-2] = 1
                F[4, 4+2*ide-2] = 1
                Ha = self.obs_j_state.dot(F)
                p = self.cov_matrix
                R = self.R

                # Calculate Kalman gain
                invertible = ((Ha.dot(p)).dot(Ha.transpose()) + R)

                denominator = np.linalg.inv(np.float64(invertible))

                K = (p.dot(Ha.transpose())).dot(denominator)

                # Update position
                state_vector = state_v + K.dot(z - h)
                self.state_vector = state_vector

                # Update covariance
                cov_matrix = (np.identity(13) - K.dot(Ha)).dot(self.cov_matrix)
                self.cov_matrix = cov_matrix

            return state_vector, cov_matrix, btheta, angle
        else:
            return state_v, cov_matrix, btheta, angle

    def propagate_state(self, msg):
        """
        Predict state based on motion model

        Args:
            msg: odometry messages

        """
        # Previous values
        x = self.state_vector[0, 0]
        y = self.state_vector[1, 0]
        theta = self.state_vector[2, 0]
        if theta < -pi:
            theta += 2 * pi
        elif theta > pi:
            theta -= 2 * pi

        # Current values
        vel = msg.twist.twist.linear.x
        ang = msg.twist.twist.angular.z
        nvel = 0
        nang = 0
        dt = msg.header.stamp.secs + msg.header.stamp.nsecs * 10 ** -9 - self.time_stamp
        self.time_stamp = msg.header.stamp.secs + msg.header.stamp.nsecs * 10 ** -9

        # Calculate Jacobians F and G
        self.motion_jacobian_state_vector(vel, ang, theta, 0, 0, dt)
        self.motion_jacobian_noise_components(vel, ang, theta, 0, 0, dt)

        # Choose motion model
        if ang == 0:
            # Propagate
            self.state_vector[0, 0] = x + (vel + nvel) * dt * cos(theta)
            self.state_vector[1, 0] = y + (vel + nvel) * dt * sin(theta)
            self.state_vector[2, 0] = theta
        else:
            # Propagate
            self.state_vector[0, 0] = x - ((vel + nvel) / (ang + nang)) * sin(theta) + (
                        (vel + nvel) / (ang + nang)) * sin(theta + (ang + nang) * dt)
            self.state_vector[1, 0] = y + ((vel + nvel) / (ang + nang)) * cos(theta) - (
                        (vel + nvel) / (ang + nang)) * cos(theta + (ang + nang) * dt)
            self.state_vector[2, 0] = theta + (ang + nang) * dt

    def calculate_cov(self):
        self.Q = (self.motion_j_noise.dot(self.q)).dot(self.motion_j_noise.transpose())
        self.cov_matrix = (self.motion_j_state.dot(self.cov_matrix)).dot(self.motion_j_state.transpose()) + self.Q

    def motion_jacobian_state_vector(self, vel, ang, theta, nvel, nang, dt):
        """
        Calculate motion Jacobian

        Args:
            vel: linear velocity
            ang: angular velocity
            theta: robot orientation
            nvel: linear velocity noise
            nang: angular velocity noise
            dt: time step

        """
        if ang == 0:
            self.motion_j_state[0, 0] = 1
            self.motion_j_state[0, 1] = 0
            self.motion_j_state[0, 2] = -(vel + nvel) * dt * sin(theta)
            self.motion_j_state[1, 0] = 0
            self.motion_j_state[1, 1] = 1
            self.motion_j_state[1, 2] = (vel + nvel) * dt * cos(theta)
            self.motion_j_state[2, 0] = 0
            self.motion_j_state[2, 1] = 0
            self.motion_j_state[2, 2] = 1
        else:
            self.motion_j_state[0, 0] = 1
            self.motion_j_state[0, 1] = 0
            self.motion_j_state[0, 2] = -(vel + nvel) * cos(theta) / (ang + nang) + (vel + nvel) * \
                                        cos(theta + dt * (ang + nang)) / (ang + nang)
            self.motion_j_state[1, 0] = 0
            self.motion_j_state[1, 1] = 1
            self.motion_j_state[1, 2] = (((-(vel + nvel)) * sin(theta)) / (ang + nang)) + (((vel + nvel) * \
                                        (sin(theta + dt * (ang + nang)))) / (ang + nang))
            self.motion_j_state[2, 0] = 0
            self.motion_j_state[2, 1] = 0
            self.motion_j_state[2, 2] = 1

    def motion_jacobian_noise_components(self, vel, ang, theta, nvel, nang, dt):
        """
        Calculate noise Jacobian

        Args:
            vel: linear velocity
            ang: angular velocity
            theta: robot orientation
            nvel: linear velocity noise
            nang: angular velocity noise
            dt: time step

        """
        if ang == 0:
            self.motion_j_noise[0, 0] = dt * cos(theta)
            self.motion_j_noise[0, 1] = 0
            self.motion_j_noise[1, 0] = dt * sin(theta)
            self.motion_j_noise[1, 1] = 0
            self.motion_j_noise[2, 0] = 0
            self.motion_j_noise[2, 1] = 0
        else:
            self.motion_j_noise[0, 0] = ((-np.sin(theta)) / (ang + nang)) + ((np.sin(theta + dt * (ang + nang))) / (ang + nang))
            self.motion_j_noise[0, 1] = ((dt * (vel + nvel) * np.cos(theta + dt * (ang + nang))) / (ang + nang)) + \
                                        (((vel + nvel) * np.sin(theta)) / ((ang + nang)**2)) - (((vel + nvel) *
                                        np.sin(theta + dt * (ang + nang))) / ((ang + nang)**2))
            self.motion_j_noise[1, 0] = (np.cos(theta) / (ang + nang)) - ((np.cos(theta + (dt * (ang + nang))) )/ (ang + nang))
            self.motion_j_noise[1, 1] = ((dt * (vel + nvel) * (np.sin(theta + dt * (ang + nang)))) / (ang + nang)) - \
                                        (((vel + nvel) * np.cos(theta)) / ((ang + nang)**2)) + (((vel + nvel) * \
                                        (np.cos(theta + dt * (ang + nang)))) / ((ang + nang)**2))
            self.motion_j_noise[2, 0] = 0
            self.motion_j_noise[2, 1] = dt

    def observation_jacobian_state_vector(self, delta, q):
        """
        Calculate observation Jacobian

        Args:
            delta: distance components
            q: distance

        Returns:

        """
        self.obs_j_state[0, 0] = -sqrt(q)*delta[0, 0]
        self.obs_j_state[0, 1] = -sqrt(q)*delta[1, 0]
        self.obs_j_state[0, 2] = 0
        self.obs_j_state[0, 3] = sqrt(q)*delta[0, 0]
        self.obs_j_state[0, 4] = sqrt(q)*delta[1, 0]
        self.obs_j_state[1, 0] = delta[1, 0]
        self.obs_j_state[1, 1] = -delta[0, 0]
        self.obs_j_state[1, 2] = -q
        self.obs_j_state[1, 3] = -delta[1, 0]
        self.obs_j_state[1, 4] = delta[0, 0]
        self.obs_j_state *= 1/q

    def print_initials(self):
        print("Printing some values")
        print("The initial stated is {}").format(self.state_vector)
        print("The initial cov. matrix is {}").format(self.cov_matrix)
