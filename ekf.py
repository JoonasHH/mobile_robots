#!/usr/bin/env python
import numpy as np
from numpy import sin, cos, sqrt, pi


class EKF:
    def __init__(self, state_vector_size, control_size, measurement_size):
        self.state_vector = np.zeros((state_vector_size, 1))
        self.state_vector[0, 0] = 0
        self.state_vector[1, 0] = 0
        self.cov_matrix = 1000. * np.identity(state_vector_size)
        self.q = np.diag(np.array([0.1, 0.01]))  # np.zeros((control_size, control_size))
        self.R = np.diag(np.array([0.1, 0.01]))  # np.zeros((measurement_size, measurement_size))
        self.motion_j_state = np.zeros((state_vector_size, state_vector_size))
        self.motion_j_noise = np.zeros((state_vector_size, control_size))
        self.obs_j_state = np.zeros((measurement_size, state_vector_size))
        self.Q = np.zeros((state_vector_size, state_vector_size))
        self.time_stamp = 0
        # Exact positions of the beacons
        self.beacons_x = np.array([7.3, 1, 9, 1, 5.8])
        self.beacons_y = np.array([3.0, 1, 9, 8, 8])
        self.gt = np.zeros((state_vector_size, 1))

    def initialize_position(self, x, y, theta):
        print("Robot position initialized")
        self.state_vector[0, 0] = x
        self.state_vector[1, 0] = y
        self.state_vector[2, 0] = theta

    def update_gt(self, x, y, theta):
        self.gt[0, 0] = x
        self.gt[1, 0] = y
        self.gt[2, 0] = theta

    def predict(self, msg):
        self.propagate_state(msg)
        self.calculate_cov()
        return self.state_vector, self.cov_matrix

    def update(self, msg):
        state_v = self.state_vector
        cov_matrix = self.cov_matrix

        btheta = None
        angle = None
        if len(msg.markers[:]) != 0:
            for i in range(len(msg.markers[:])):
                x = state_v[0, 0] - (5 - 4.87929284015)
                y = state_v[1, 0] - (5 - 4.97496205976)
                theta = state_v[2, 0] - 0.68430241841

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
                id = msg.markers[i].ids[0]


                # Exact position of the beacon corresponding to i
                Lx = self.beacons_x[id - 1]
                Ly = self.beacons_y[id - 1]

                # Build matrix h(Xi)=zi
                z = np.array([[np.sqrt((bx**2) + (by**2))],
                              [btheta]])

                marker_angle = np.arctan2((Ly - y), abs(Lx - x))
                angle = marker_angle - theta
                if angle < -pi:
                    angle += 2 * pi
                elif angle > pi:
                    angle -= 2 * pi

                h = np.array([[np.sqrt((Lx - x)**2 + (Ly - y)**2)],
                              [angle]])

                # Calculate Jacobian H
                self.observation_jacobian_state_vector(x, y, Lx, Ly)

                # KF coeff
                Ha = self.obs_j_state
                p = self.cov_matrix
                R = self.R

                # Calculate Kalman gain K
                invertible = ((Ha.dot(p)).dot(Ha.transpose()) + R)

                denominator = np.linalg.inv(np.float64(invertible))

                K = (p.dot(Ha.transpose())).dot(denominator)

                # Update position
                state_vector = state_v + K.dot(z - h)
                state_v = state_vector
                self.state_vector = state_vector

                # Update covariance
                cov_matrix = (np.identity(3) - K.dot(self.obs_j_state)).dot(self.cov_matrix)
                self.cov_matrix = cov_matrix

            return state_vector, cov_matrix, btheta, angle
        else:
            return state_v, cov_matrix, btheta, angle

    def propagate_state(self, msg):
        # Previous values
        x = self.state_vector[0, 0]
        y = self.state_vector[1, 0]
        theta = self.state_vector[2, 0]

        # Limit theta
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

        # Choose motion model based on angular velocity
        if ang == 0:
            # Propagate linear movement
            self.state_vector[0, 0] = x + (vel + nvel) * dt * cos(theta)
            self.state_vector[1, 0] = y + (vel + nvel) * dt * sin(theta)
            self.state_vector[2, 0] = theta
        else:
            # Propagate angular movement
            self.state_vector[0, 0] = x - ((vel + nvel) / (ang + nang)) * sin(theta) + (
                        (vel + nvel) / (ang + nang)) * sin(theta + (ang + nang) * dt)
            self.state_vector[1, 0] = y + ((vel + nvel) / (ang + nang)) * cos(theta) - (
                        (vel + nvel) / (ang + nang)) * cos(theta + (ang + nang) * dt)
            self.state_vector[2, 0] = theta + (ang + nang) * dt

    def calculate_cov(self):
        self.Q = (self.motion_j_noise.dot(self.q)).dot(self.motion_j_noise.transpose())
        self.cov_matrix = (self.motion_j_state.dot(self.cov_matrix)).dot(self.motion_j_state.transpose()) + self.Q

    def motion_jacobian_state_vector(self, vel, ang, theta, nvel, nang, dt):
        # Choose motion model based on angular velocity
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
        # Choose motion model based on angular velocity
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

    def observation_jacobian_state_vector(self, x, y, Lx, Ly):
        self.obs_j_state[0, 0] = (-Lx + x) / sqrt(((Lx - x) ** 2) + ((Ly - y) ** 2))
        self.obs_j_state[0, 1] = (-Ly + y) / sqrt(((Lx - x) ** 2) + ((Ly - y) ** 2))
        self.obs_j_state[0, 2] = 0
        self.obs_j_state[1, 0] = (Ly - y) / (((Lx - x) ** 2) + ((Ly - y) ** 2))
        self.obs_j_state[1, 1] = (-Lx + x) / (((Lx - x) ** 2) + ((Ly - y) ** 2))
        self.obs_j_state[1, 2] = -1

    def print_initials(self):
        print("Printing some values")
        print("The initial stated is {}").format(self.state_vector)
        print("The initial cov. matrix is {}").format(self.cov_matrix)
