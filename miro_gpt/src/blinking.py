#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray
import time

def wiggle(v, n, m):
	v = v + float(n) / float(m)
	if v > 2.0:
		v -= 2.0
	elif v > 1.0:
		v = 2.0 - v
	return v
class Blinking():

    def __init__(self):
        rospy.init_node("blink", anonymous=True)
        self.rate = rospy.Rate(10)
        self.ctrl_c = False        
        rospy.on_shutdown(self.shutdownhook)
        self.user_blink = 0
        self.pub_cos = rospy.Publisher("miro/control/cosmetic_joints", Float32MultiArray, queue_size=0)
        self.cos_joints = Float32MultiArray()
        self.cos_joints.data = [0.0, 0.5, 0.0, 0.0, 0.3333, 0.3333]
        self.can_blink = False
        self.done = False
        self.last = time.time()
    
    def shutdownhook(self):
         self.ctrl_c = True

    def blink(self):
        wiggle_n = 20


        self.cos_joints.data[2] = wiggle(0.0, self.user_blink, wiggle_n) # left eye
        self.cos_joints.data[3] = wiggle(0.0, self.user_blink, wiggle_n) # right eye

        self.user_blink += 1
        if self.user_blink > (2 * wiggle_n):
            self.user_blink = 0
            self.done = True
 
        print(self.user_blink)
        self.pub_cos.publish(self.cos_joints)
        if self.done and (self.user_blink == 0):
             self.can_blink = False
             
        

    def main(self):        
        while not self.ctrl_c:
            if time.time() - self.last > 4.0:
                self.last = time.time()
                self.can_blink = True
            
            if self.can_blink:
                 self.blink()
        



if __name__ == "__main__":
    node = Blinking()
    node.main()