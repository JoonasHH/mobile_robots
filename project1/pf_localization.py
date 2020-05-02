#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from marker_msgs.msg import MarkerDetection
from filtering_utils.pf import PF
import numpy as np
import matplotlib.pyplot as plt


pf = PF(3)

pub = rospy.Publisher("ekf_state", Pose, queue_size=10)
pub2 = rospy.Publisher("cov_matrix", Pose, queue_size=10)
pub3 = rospy.Publisher("gt", Pose, queue_size=10)
pub4 = rospy.Publisher("marker", Pose, queue_size=10)
pub5 = rospy.Publisher("conv_odom", Pose, queue_size=10)


class PFloc:

    def __init__(self):
        self.cov_matrix = None
        self.state = None
        self.initialized = False
        self.ground_truth = None
        self.particles = None
        self.particle_list = []

    def odom_callback(self, msg):
        if self.initialized:
            self.state, self.particles = pf.run_pf_1(msg)
            odo = Pose()
            odo.position.x = msg.pose.pose.position.x + 5
            odo.position.y = msg.pose.pose.position.y + 5
            quat1 = 2 * (msg.pose.pose.orientation.w * msg.pose.pose.orientation.z +
                         msg.pose.pose.orientation.x * msg.pose.pose.orientation.y)
            quat2 = 1 - 2 * (msg.pose.pose.orientation.y * msg.pose.pose.orientation.y +
                             msg.pose.pose.orientation.z * msg.pose.pose.orientation.z)
            theta = np.arctan2(quat1, quat2)
            odo.orientation.z = theta
            pub5.publish(odo)

    def marker_callback(self, msg):
        if self.initialized:
            self.state, self.cov_matrix = pf.run_pf_2(msg)
            pose = Pose()
            # if self.state is not None:
            pose.position.x = self.state[0]
            pose.position.y = self.state[1]
            # pose.orientation.z = self.state[2]
            pub.publish(pose)

            cov = Pose()
            cov.position.x = np.sqrt(self.cov_matrix[0])
            cov.position.y = np.sqrt(self.cov_matrix[1])
            cov.orientation.x = self.cov_matrix[0]
            cov.orientation.y = self.cov_matrix[1]
            pub2.publish(cov)

    def ground_truth_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        quat1 = 2 * (msg.pose.pose.orientation.w * msg.pose.pose.orientation.z +
                     msg.pose.pose.orientation.x * msg.pose.pose.orientation.y)
        quat2 = 1 - 2 * (msg.pose.pose.orientation.y * msg.pose.pose.orientation.y +
                         msg.pose.pose.orientation.z * msg.pose.pose.orientation.z)
        theta = np.arctan2(quat1, quat2)
        pf.update_gt(x, y)
        if not self.initialized:
            pf.initialize_position(x, y)
            self.initialized = True

        self.ground_truth = np.array([x, y])

        gt = Pose()
        gt.position.x = x
        gt.position.y = y
        gt.orientation.z = theta

        pub3.publish(gt)


    def pf_loc(self):
        rospy.init_node('pf_localization', anonymous=True)
        rospy.Subscriber("base_pose_ground_truth", Odometry, self.ground_truth_callback)
        rospy.Subscriber("odom", Odometry, self.odom_callback)
        rospy.Subscriber("base_marker_detection", MarkerDetection, self.marker_callback)
        rospy.spin()


if __name__ == '__main__':

    PFloc().pf_loc()
