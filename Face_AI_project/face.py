import cv2
from random import randrange
#load some pre trained data on face frontals from opencv(haar cascade algorithm)
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choose an image to detect faces in 
# img =cv2.imread('abc.png')
# imp =cv2.imread('Qaxi_Aaron.png')
# To capture video from webcam.
webcam=cv2.VideoCapture(0)
# Iterate forever over frames untill webcam turns off
while True:
    # Read the current frame 
    successful_frame_read, frame=webcam.read()
    # Must convert to grayscale 
    grayscalled_frame=cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    frame2=cv2.cvtColor(frame,cv2.BORDER_REPLICATE)
    
    # detect faces
    face_coordinates=trained_face_data.detectMultiScale(grayscalled_frame)
    #Draw recgtangles
    for(x, y, w, h) in face_coordinates:
        (x,y,w,h)=face_coordinates[0]
        cv2.rectangle(frame,(x,y),(x+w,y+w),(randrange(256),randrange(256),randrange(256)),10)
    
    #display
    cv2.imshow('jaydeep',frame)
    cv2.imshow("gray",grayscalled_frame)
    cv2.imshow(" ",frame2)
    cv2.waitKey(1)