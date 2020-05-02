#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from marker_msgs.msg import MarkerDetection
from filtering_utils.ekf import EKF
import matplotlib.pyplot as plt
import numpy as np
from geometry_msgs.msg import Pose

ekf = EKF(3, 2, 2)

pub = rospy.Publisher("ekf_state", Pose, queue_size=10)
pub2 = rospy.Publisher("cov_matrix", Pose, queue_size=10)
pub3 = rospy.Publisher("gt", Pose, queue_size=10)
pub4 = rospy.Publisher("marker", Pose, queue_size=10)
pub5 = rospy.Publisher("conv_odom", Pose, queue_size=10)


class EKFlocalizer:

    def __init__(self):
        self.state = None
        self.ground_truth = None
        self.cov_matrix = None
        self.initialized = False

    def odom_callback(self, msg):
        if self.initialized:
            self.state, self.cov_matrix = ekf.predict(msg)

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
            self.state, self.cov_matrix, btheta, angle = ekf.update(msg)

            marker = Pose()
            if btheta is not None:
                marker.orientation.z = btheta
                marker.orientation.y = angle

            diags = np.diagonal(self.cov_matrix)

            cov = Pose()
            cov.position.x = np.sqrt(diags[0])
            cov.position.y = np.sqrt(diags[1])
            cov.position.z = np.sqrt(diags[2])
            cov.orientation.w = diags[0]
            cov.orientation.x = diags[1]
            cov.orientation.y = diags[2]
            pub2.publish(cov)

            pose = Pose()
            pose.position.x = self.state[0]
            pose.position.y = self.state[1]
            pose.orientation.z = self.state[2]
            pub.publish(pose)

            pub4.publish(marker)

    def ground_truth_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        quat1 = 2 * (msg.pose.pose.orientation.w * msg.pose.pose.orientation.z +
                     msg.pose.pose.orientation.x * msg.pose.pose.orientation.y)
        quat2 = 1 - 2 * (msg.pose.pose.orientation.y * msg.pose.pose.orientation.y +
                         msg.pose.pose.orientation.z * msg.pose.pose.orientation.z)
        theta = np.arctan2(quat1, quat2)
        ekf.update_gt(x, y, theta)
        if not self.initialized:
            ekf.initialize_position(x, y, theta)
            self.initialized = True

        self.ground_truth = np.array([x, y, theta])

        gt = Pose()
        gt.position.x = x
        gt.position.y = y
        gt.orientation.z = theta

        pub3.publish(gt)

    def ekf_loc(self):
        rospy.init_node('ekf_localization', anonymous=True)
        rospy.Subscriber("base_pose_ground_truth", Odometry, self.ground_truth_callback)
        rospy.Subscriber("odom", Odometry, self.odom_callback)
        rospy.Subscriber("base_marker_detection", MarkerDetection, self.marker_callback)
        rospy.spin()


if __name__ == '__main__':
    try:
        EKFlocalizer().ekf_loc()
    except rospy.ROSInterruptException:
        pass
