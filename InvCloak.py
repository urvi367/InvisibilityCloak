#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from imutils.video import WebcamVideoStream
import imutils
import cv2
import numpy as np
import time


# In[ ]:


# Creating a VideoCapture object
# This will be used for image acquisition later in the code.
vs = WebcamVideoStream(src=0).start()
 
# We give some time for the camera to warm-up!
time.sleep(3)
 
background=0
 
print('------------------------------Capturing background!------------------------------')
for i in range(30):
    background = vs.read()
print('------------------------------Background captured!------------------------------')

# Laterally invert the image / flip the image.
background = np.flip(background,axis=1)


# In[ ]:


while True:
    pressed_key = cv2.waitKey(1)
    
    # Capturing the live frame
    img = vs.read()

    # Laterally invert the image / flip the image
    img  = np.flip(img, axis=1)
 
    # converting from BGR to HSV color space
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
 
    # setting the lower and upper range for mask1,[H,S,V], for lighter shades
    lower_red = np.array([0,70,70])
    upper_red = np.array([10,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
 
    # setting the lower and upper range for mask2, for darker shades
    lower_red = np.array([170,70,70])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(hsv,lower_red,upper_red)
 
    # Generating the final mask to detect red color
    mask1 = mask1+mask2
    
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    mask1 = cv2.dilate(mask1, np.ones((3, 3), np.uint8), iterations = 1)
    #mask1 = only cloth
 
 
    #creating an inverted mask to segment out the cloth from the frame, everything accept cloth
    mask2 = cv2.bitwise_not(mask1)
 
 
    #Segmenting the cloth out of the frame using bitwise and with the inverted mask
    #everything from image except cloth
    res1 = cv2.bitwise_and(img,img,mask=mask2)
    
    # creating image showing static background frame pixels only for the masked region
    #cloth ka space from background
    res2 = cv2.bitwise_and(background, background, mask = mask1)
 
 
    #Generating the final output
    final_output = cv2.addWeighted(res1,1,res2,1,0)
    cv2.imshow("Invisibility Cloakkk!",final_output)

    if pressed_key == 27:
        break

cv2.destroyAllWindows()
vs.stop() 
