#! /usr/bin/env python
import sys
import time
import os
import numpy as np
import rospy
from std_msgs.msg import String, Float32MultiArray
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, PoseStamped, Transform, TransformStamped, Quaternion
import tf
import pickle
from autolab_core import RigidTransform

def trans2pose(trans):
    pose = Pose()
    pose.orientation = trans.rotation
    pose.position.x = trans.translation.x
    pose.position.y = trans.translation.y
    pose.position.z = trans.translation.z
    return pose

def pose2trans(pose):
    trans = Transform()
    trans.rotation = pose.orientation
    trans.translation =pose.position
    return trans

def trans2Rigidtrans(trans, from_frame, to_frame):
    rotation_quaternion = np.asarray([trans.rotation.w, trans.rotation.x, trans.rotation.y, trans.rotation.z])
    T_trans = np.asarray([trans.translation.x, trans.translation.y, trans.translation.z])
    T_qua2rota = RigidTransform(rotation_quaternion, T_trans, from_frame = from_frame, to_frame=to_frame)
    return T_qua2rota

def pose2Rigidtrans(pose, from_frame, to_frame):
    trans = pose2trans(pose)
    T_qua2rota = trans2Rigidtrans(trans, from_frame, to_frame)
    return T_qua2rota

def transstamp2Rigidtrans(trans):
    from_frame = trans.child_frame_id
    # to_frame = trans.child_frame_id
    to_frame = trans.header.frame_id
    T_qua2rota = trans2Rigidtrans(trans.transform, from_frame, to_frame)
    return T_qua2rota

def Rigidtrans2transstamped(Rigidtrans):
    trans = TransformStamped()
    trans.header.stamp = rospy.Time.now()
    trans.header.frame_id = Rigidtrans.to_frame
    trans.child_frame_id = Rigidtrans.from_frame
    trans.transform = pose2trans(Rigidtrans.pose_msg)
    return trans

def tf2Rigidtrans(trans, from_frame, to_frame):
    rotation_quaternion = np.asarray([trans[1][3], trans[1][0], trans[1][1], trans[1][2]])
    T_trans = np.asarray([trans[0][0], trans[0][1], trans[0][2]])
    T_qua2rota = RigidTransform(rotation_quaternion, T_trans, from_frame = from_frame, to_frame=to_frame)
    return T_qua2rota

if __name__ == '__main__':
    rospy.init_node('test_publish_tf', anonymous=True)
    robot1_pub = rospy.Publisher('robot1/robot_pos', Float32MultiArray, queue_size=10)
    robot2_pub = rospy.Publisher('robot2/robot_pos', Float32MultiArray, queue_size=10)
    br = tf.TransformBroadcaster()
    tf_listener = tf.TransformListener()
    rate = rospy.Rate(10)
    # from, source: child_id  to, target: frame_id
    while not rospy.is_shutdown():
        try:
            #now = rospy.Time.now()
            #tf_listener.waitForTransform('robot1/map','robot1/odom', rospy.Time.now(), rospy.Duration(10.0))
            robot1_map_to_odom = tf_listener.lookupTransform('robot1/map', 'robot1/odom', rospy.Time(0))
            robot2_map_to_odom = tf_listener.lookupTransform('robot2/map', 'robot2/odom', rospy.Time(0))
        except:
            print('no tf')
            rate.sleep()
            continue
        robot1_map_to_odom_Rigidtrans = tf2Rigidtrans(robot1_map_to_odom, 'robot1/odom', 'robot1/map')
        robot2_map_to_odom_Rigidtrans = tf2Rigidtrans(robot2_map_to_odom, 'robot2/odom', 'robot2/map')
        robot1_odom_to_robot2_odom_Rigidtrans = tf2Rigidtrans(([0,0,0],[0,0,0,1]), 'robot2/odom', 'robot1/odom')
        robot1_map_to_robot2_map_Rigidtrans = robot1_map_to_odom_Rigidtrans*robot1_odom_to_robot2_odom_Rigidtrans*robot2_map_to_odom_Rigidtrans.inverse()
        
        tf = Rigidtrans2transstamped(robot1_map_to_robot2_map_Rigidtrans)
        # print("publish tf: " + tf.header.frame_id + " to " + tf.child_frame_id)
        br.sendTransformMessage(tf)
        # send robot pos
        try:
            #now = rospy.Time.now()
            #tf_listener.waitForTransform('robot1/map','robot1/odom', rospy.Time.now(), rospy.Duration(10.0))
            robot1_map_to_base = tf_listener.lookupTransform('robot1/map', 'robot1/base_footprint', rospy.Time(0))
            robot2_map_to_base = tf_listener.lookupTransform('robot2/map', 'robot2/base_footprint', rospy.Time(0))
        except:
            print('no tf')
            rate.sleep()
            continue
        robot1_pos_msg = Float32MultiArray()
        robot1_pos_msg.data.append(robot1_map_to_base[0][0])
        robot1_pos_msg.data.append(robot1_map_to_base[0][1])
        robot2_pos_msg = Float32MultiArray()
        robot2_pos_msg.data.append(robot2_map_to_base[0][0])
        robot2_pos_msg.data.append(robot2_map_to_base[0][1])
        robot1_pub.publish(robot1_pos_msg)
        robot2_pub.publish(robot2_pos_msg)
        time.sleep(1)