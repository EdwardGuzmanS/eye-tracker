#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 20:50:16 2025

@author: edu
"""

import cv2
from methods import thresholding_pupil_detection


'''
def find_available_cameras():
    index = 0
    available_cameras = []
    while index < 10:  # Intenta hasta 10 índices
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cameras.append(index)
            cap.release()
        index += 1
    return available_cameras

print("Cámaras disponibles:", find_available_cameras())
'''
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Cannot open camera")
    #exit()
MODE = "pupil"

while True:
    # Capture frame-by-frame
    ret, image = cap.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Our operations on the frame come here
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if MODE == "pupil":
        (cx, cy), processed = thresholding_pupil_detection(gray)
        if cx is not None and cy is not None:
            cv2.circle(processed, (cx, cy), 5, (100, 255, 255), -1)  # Dibuja un punto
    if MODE == "starburst"
    
    cv2.imshow('image', processed)
    if cv2.waitKey(1) == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
