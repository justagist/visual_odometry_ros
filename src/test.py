#!/usr/bin/env python
import rospy
from visual_odometry_stam.msg import TransmatMsg

def mytopic_callback(msg):
    # print "Here are some integers:", str(msg.some_integers)
    print "Here are some floats:", str(msg.vals)

mysub = rospy.Subscriber('/trans_mat',TransmatMsg, mytopic_callback)