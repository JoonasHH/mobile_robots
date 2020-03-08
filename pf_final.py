import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform, randn, random
import scipy
from scipy.stats import norm


class PF:
    def __init__(self, state_vector_size):
        self.N = 7000  # number of particles
        self.particles = None
        self.create_uniform_particles([0, 10], [0, 10], [0, 2 * np.pi])
        self.weights = np.zeros(self.N)

        self.state_vector = np.zeros((state_vector_size, 1))
        self.cov_matrix = 1000. * np.identity(state_vector_size)
        self.q = np.diag(np.array([0.1, 0.01]))  # np.zeros((control_size, control_size))
        self.R = np.diag(np.array([0.1, 0.01]))  # np.zeros((measurement_size, measurement_size))
        self.time_stamp = 0
        # Exact positions of the beacons
        self.beacons_x = np.array([7.3, 1, 9, 1, 5.8])
        self.beacons_y = np.array([3.0, 1, 9, 8, 8])
        self.gt = np.zeros((state_vector_size, 1))
        self.mean = None
        self.variance = None

    def initialize_position(self, x, y):
        self.state_vector[0] = x
        self.state_vector[1] = y

    def update_gt(self, x, y):
        self.gt[0] = x
        self.gt[1] = y

    def create_uniform_particles(self, x_range, y_range, hdg_range):
        # creating an uniform distribution of particles
        self.particles = np.empty((self.N, 3))
        self.particles[:, 0] = uniform(x_range[0], x_range[1], size=self.N)
        self.particles[:, 1] = uniform(y_range[0], y_range[1], size=self.N)
        self.particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=self.N)

        self.particles[:, 2] %= 2 * np.pi

        return

    def predict(self, vel, ang, dt):
        # move the particles according to the velocity and angular velocity of the robot (with noise q)

        # update heading
        self.particles[:, 2] += (ang * dt) + (randn(self.N) * self.q[1, 1])
        self.particles[:, 2] %= 2 * np.pi

        # move in the (noisy) commanded direction
        dist = (vel * dt) + (randn(self.N) * self.q[0, 0])
        self.particles[:, 0] += np.cos(self.particles[:, 2]) * dist
        self.particles[:, 1] += np.sin(self.particles[:, 2]) * dist

        return

    def update(self, z, beacons):
        self.weights.fill(1.)

        for i, beacon in enumerate(beacons):
            distance = np.linalg.norm(self.particles[:, 0:2] - beacon, axis=1)
            self.weights *= scipy.stats.norm(distance, self.R[0, 0]).pdf(z[i])

        self.weights += 1.e-300  # avoid round-off to zero
        self.weights /= sum(self.weights)  # normalize

        return

    def estimate(self):
        # returns mean and variance of the weighted particles
        pos = self.particles[:, 0:2]
        mean = np.average(pos, weights=self.weights, axis=0)
        var = np.average((pos - mean) ** 2, weights=self.weights, axis=0)

        return mean, var

    def simple_resample(self):

        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.  # avoid round-off error
        indexes = np.searchsorted(cumulative_sum, random(self.N))

        # resample according to indexes
        self.particles[:] = self.particles[indexes]
        self.weights[:] = self.weights[indexes]
        self.weights /= np.sum(self.weights)  # normalize
        return

    def neff(self):
        return 1. / np.sum(np.square(self.weights))

    def run_pf_1(self, msg):
        # values from the odometry of the robot
        vel = msg.twist.twist.linear.x
        ang = msg.twist.twist.angular.z

        # time of the step
        dt = msg.header.stamp.secs + msg.header.stamp.nsecs * 10 ** -9 - self.time_stamp
        self.time_stamp = msg.header.stamp.secs + msg.header.stamp.nsecs * 10 ** -9

        # prediction of the particles
        self.predict(vel, ang, dt)

        return self.state_vector, self.particles

    def run_pf_2(self, msg):

        NL = len(msg.markers[:])
        if NL != 0:
            beacons = np.zeros((NL, 2))
            b_pos = np.zeros((NL, 2))
            for i in range(len(msg.markers[:])):
                id = msg.markers[i].ids[0]
                beacons[i, :] = np.array([self.beacons_x[id - 1], self.beacons_y[id - 1]])
                b_pos[i] = [msg.markers[i].pose.position.x, msg.markers[i].pose.position.y]

            z = np.linalg.norm(b_pos, axis=1) + (randn(NL) * self.R[0, 0])
            self.update(z, beacons)

            [mean, variance] = self.estimate()

            if self.neff() < self.N / 2:
                self.simple_resample()

            self.mean = mean
            self.variance = variance

            return mean, variance
        else:
            return self.mean, self.variance

