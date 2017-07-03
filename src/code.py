#!/usr/bin/env python
import sys
# from stam import STAM
# from cam_odo import vis_odom
import rospy
from vis_odo_ros import STAM

if __name__ == "__main__":
    if len(sys.argv) == 1:
        scene_no = 1
        print "Using default Scene:%d"%scene_no
    elif len(sys.argv) == 2:
       if int(sys.argv[1]) in (1,2,3):
           scene_no = int(sys.argv[1])
           print "Using Scene %d"%scene_no
       else:
           sys.exit("Error: Invalid Scene no. provided. Scene no. should be 1,2 or 3")
    else:
        sys.exit("Error: Invalid no of arguments given. len(sys.argv)=%d"%len(sys.argv))
    
    #init_image = read_images(scene_path+"0000.png")
    # s = STAM(scene_no)
    # s.start()

    rospy.init_node('image_from_ardrone', anonymous=True, disable_signals=True)
    cam = STAM(scene_no)
    # cam = vis_odom()
    cam.start()