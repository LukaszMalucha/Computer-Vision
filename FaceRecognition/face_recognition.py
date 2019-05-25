# -*- coding: utf-8 -*-
"""
Created on Sat May 25 17:38:53 2019

@author: MaximusMinimus
"""


import cv2

# Loading cascades  - frontalface and eye

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Build the detection function
def detect(gray, frame):
    # get coordinates of the face rectangle
    # as arguments - image in greyscale, scale factor - img reduced 1.3 times, min number of neighbours that got accepted 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # iterate through faces (coordinates x & y, widht & height)
    for (x,y, w, h) in faces:
        # draw a rectangle(image, coords of upper-left corner, coords of lower-right corner, color, edge thickness)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        
        # two regions of interest - one for greyscale, other for original camera img
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        
        
        # same for eyes within the recoginzed face
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex,ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
    return frame                     

# Perform Face Recognition with the webcam
    
# Get the last frame from the webcam
video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()  # get second element
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #transform img to gray
    canvas = detect(gray, frame)# apply detect function
    cv2.imshow('Video', canvas) # display outputs
    
    # Quit loop wit q key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()        


























