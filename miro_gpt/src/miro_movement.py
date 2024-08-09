#!/usr/bin/env python3
import os
import time
import rospy
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, UInt32MultiArray
from geometry_msgs.msg import TwistStamped
import math
import miro2 as miro
from multiprocessing import Process
import sys
from train import PredictResponse


class Movement(object):

    def __init__(self, message):
        # rospy.init_node("movement")
        self.position = None
        self.start_time = time.time()
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")

        self.kinematic_pub = rospy.Publisher(
            topic_base_name + "/control/kinematic_joints", JointState, queue_size=0
        )

        self.cosmetic_pub = rospy.Publisher(
            topic_base_name + "/control/cosmetic_joints", Float32MultiArray, queue_size=0
        )

        self.vel_pub = rospy.Publisher(
            topic_base_name + "/control/cmd_vel", TwistStamped, queue_size=0
        )

        self.pub_illumination = rospy.Publisher(
            topic_base_name + "/control/illum", UInt32MultiArray, queue_size=0
        )

        # Colour change
        self.color_change = UInt32MultiArray()

        # User Prompt from process_audio
        self.user_prompt = message

        
        rospy.on_shutdown(self.shutdown_hook)
    

    def set_move_kinematic(self,
                           tilt = miro.constants.TILT_RAD_CALIB, # not a real DOF
                           lift = miro.constants.LIFT_RAD_CALIB,
                           yaw = miro.constants.YAW_RAD_CALIB,
                           pitch = miro.constants.PITCH_RAD_CALIB
                           ):
        joint_cmd = JointState()
        joint_cmd.position = [tilt, lift, yaw, pitch]
        self.kinematic_pub.publish(joint_cmd)

    # movement for the tail, eye lid, ears
    # default values are calibration (except eyes)
    def set_move_cosmetic(self,
                          tail_pitch = 0,
                          tail_yaw = 0.5,
                          left_eye = 0, # No ptosis
                          right_eye = 0,
                          left_ear = 1.0/3.0,
                          right_ear = 1.0/3.0
                          ):
        joint_cmd = Float32MultiArray()
        joint_cmd.data = [tail_pitch,tail_yaw,left_eye,right_eye,left_ear,right_ear]
        self.cosmetic_pub.publish(joint_cmd)

    def set_move_cmd(self, linear = 0.0, angular = 0.0):
        vel_cmd = TwistStamped()
        # explanation of the messages in the document
        # message variable to move forward is done by linear.x
        vel_cmd.twist.linear.x = linear
        # message variable to turn is done by angular.z
        vel_cmd.twist.angular.z = angular
        print("HERE IT IS: ", vel_cmd.twist.linear.x)
        print("HERE IT IS2: ", vel_cmd.twist.angular.z)

        self.vel_pub.publish(vel_cmd)

    def shutdown_hook(self):
        # Move joints to default positions
        self.set_move_kinematic()
        self.set_move_cosmetic()
        self.set_move_cmd()

    def execute_movements(self,lin,ang):
        self.set_move_cmd(lin,ang)
        
    def wag_tail(do, self):
        while(do):
            self.set_move_cosmetic(tail_yaw=1)
            rospy.sleep(0.5)
            self.set_move_cosmetic(tail_yaw=-1)
            rospy.sleep(0.5)
    def left_ear_waggle(self,do):
        while(do):
            self.set_move_cosmetic(left_ear=1)
            rospy.sleep(1)
            self.set_move_cosmetic(left_ear=-1)
            rospy.sleep(1)
    def right_ear_waggle(self,do):
        self.set_move_cosmetic(right_ear=1)
        rospy.sleep(1)
        self.set_move_cosmetic(right_ear=-1)
        rospy.sleep(1)

    def stop_movements(self):
        self.set_move_cmd(0,0)

    # Colours
    def get_illumination(self, red = 0, green = 0, blue = 0):
        # changing the rgb format into android format to be published in MiRo message
        color_detail = (int(red), int(green), int(blue))
        color = '0xFF%02x%02x%02x'%color_detail
        color = int(color, 16)
        return color
    
    def green_light(self):
        color = self.get_illumination(green = 150)
        self.color_change.data = [
            color,
            color,
            color,
            color,
            color,
            color
        ]
    
    def red_light(self):
        color = self.get_illumination(red = 150)
        self.color_change.data = [
            color,
            color,
            color,
            color,
            color,
            color
        ]

    def no_process(self):
        color = self.get_illumination()
        self.color_change.data = [
            color,
            color,
            color,
            color,
            color,
            color
        ]


    def execute(self,linV,angV,lift1,yaw1,pitch1,left_eye,right_eye,left_ear,right_ear,tail, lights):
        if(linV > 0.0):
            for x in range(100):
                self.execute_movements(linV,angV)
                rospy.sleep(0.02)
            self.stop_movements()
        elif(angV > 0.0):
            for x in range(100):
                self.execute_movements(linV,angV)
                rospy.sleep(0.02)
            self.stop_movements()
        
        learet = 1.0/3.0
        rearet = 1.0/3.0
        tailret = 0

        if(left_ear > 0):
            learet = -1
        if(right_ear > 0):
            rearet = -1
        if(tail > 0):
            tailret = -1

        if lights == 0: # 0 for green
            self.green_light()
            self.pub_illumination.publish(self.color_change)
        elif lights == 1:
            self.red_light() # 1 for red
            self.pub_illumination.publish(self.color_change)
            

        # print(yaw1)
        # print(math.radians(lift1))
        self.set_move_kinematic(lift=math.radians(lift1),yaw=math.radians(yaw1),pitch=math.radians(pitch1))
        self.set_move_cosmetic(tail_yaw=tail,left_ear=left_ear,right_ear=right_ear,left_eye=left_eye,right_eye=right_eye)
        rospy.sleep(1)
        self.set_move_cosmetic(tail_yaw = tailret, left_ear=learet, right_ear=rearet,left_eye=0,right_eye=0)
        rospy.sleep(0.5)
        self.set_move_cosmetic()
    
    def main(self):
        current_time = time.time()
        t = PredictResponse()
        response_list = t.predict(self.user_prompt)
        print("Responese List: ", response_list)
        if (response_list != []):
            for concProcess in response_list:
                for instruction in concProcess:
                    linV = 0
                    angV = 0
                    left_eye = 0
                    right_eye = 0
                    left_ear = 1.0/3.0
                    right_ear = 1.0/3.0
                    tail_wag = 0
                    lift = 34
                    yaw = 0
                    pitch = 0
                    lights = 2 # 2 is for "None"
                    print("Instruction: ", instruction)
                    movementType, *args = instruction
                    print("MovementType: ",movementType)
                    if(movementType == "move"):
                        linV = args[0]
                        angV = args[1]
                    elif(movementType == "head_pitch"):
                        pitch = args[0]
                    elif(movementType == "tail_wag"):
                        tail_wag = args[0]
                    elif(movementType == "ears_left_rotate"):
                        left_ear = args[0]
                    elif(movementType == "ears_right_rotate"):
                        right_ear = args[0]
                    elif(movementType == "head_yaw"):
                        yaw = args[0]
                    elif(movementType == "left_eye_close"):
                        left_eye = args[0]
                    elif(movementType == "right_eye_close"):
                        right_eye = args[0]
                    elif(movementType == "head_lift"):
                        lift = args[0]
                    elif(movementType == "lights"):
                        lights = args[0]
                    else:
                        print("NONONONO")
                
                    self.execute(linV,angV,lift,yaw,pitch,left_eye,right_eye,left_ear,right_ear,tail_wag, lights)

            response_list = []
            self.no_process()
            self.pub_illumination.publish(self.color_change)
    





if __name__ == '__main__':
    #movement = Movement()
    test_response_list = [[['move', 0, 1], ['head_yaw', 30, 1], ['head_pitch', -5, 1], ['tail_wag', 3, 0]], [['move', 0, 1], ['head_yaw', -30, 1], ['head_pitch', -5, 1], ['tail_wag', 3, 0]], [['move', 0, 1], ['head_yaw', 20, 1], ['head_pitch', -3, 1], ['tail_wag', 3, 0]], [['move', 0, 0.5], ['head_yaw', 40, 1], ['head_pitch', -5, 1], ['tail_wag', 3, 0]], [['move', 0, 0.5], ['head_yaw', -40, 1], ['head_pitch', -5, 1], ['tail_wag', 3, 0]]]
    response_list =[[['move', -0.25, 0.5], ['head_lift', 60.0, 2], ['tail_wag', 0, 2]], [['head_yaw', -30.0, 2], ['move', 0.25, 0.5]]]


    t = PredictResponse()

    print("attempting")
   

    while not rospy.is_shutdown():
        current_time = time.time()
        #right_ear_waggle(True)
        #wag_tail(True)
        if (response_list != []):
            for concProcess in test_response_list:
                for instruction in concProcess:
                    linV = 0
                    angV = 0
                    left_eye = 0
                    right_eye = 0
                    left_ear = 1.0/3.0
                    right_ear = 1.0/3.0
                    tail_wag = 0
                    lift = 34
                    yaw = 0
                    pitch = 0
                    movementType, *args = instruction
                    print(movementType)
                    if(movementType == "move"):
                        linV = args[0]
                        angV = args[1]
                        
                    elif(movementType == "head_pitch"):
                        pitch = args[0]
                    elif(movementType == "tail_wag"):
                        tail_wag = args[0]
                    elif(movementType == "ears_left_rotate"):
                        left_ear = args[0]
                    elif(movementType == "ears_right_rotate"):
                        right_ear = args[0]
                    elif(movementType == "head_yaw"):
                        yaw = args[0]
                    elif(movementType == "left_eye_close"):
                        left_eye = args[0]
                    elif(movementType == "right_eye_close"):
                        right_eye = args[0]
                    elif(movementType == "head_lift"):
                        lift = args[0]
                        print("2:",lift)
                    else:
                        print("NONONONO")
                
                    #execute(linV,angV,lift,yaw,pitch,left_eye,right_eye,left_ear,right_ear,tail_wag)

            response_list = []
