import cv2 as cv
import numpy as np

def overlay(background, overlay, coord):
    y,x = coord
    w,h,_ = background.shape
    print(coord)
    if (x+160 < w and x>=0 and y+160 < h and y >=0):
        background[x:x+160, y:y+160] = overlay
    return background


def faceDetect(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_bounding_box = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 1)
    coord = (0,0)
    for (a,b,c,d) in face_bounding_box:
        cv.rectangle(frame, (a,b), (a+c, b+d), (0,255,0), thickness=2)
        coord = (a,b)
        break
    
    return frame, coord
    


head = cv.imread('Head.png')
h_x,h_y,_ = head.shape

background = cv.imread('white.jpg')

capture = cv.VideoCapture(0)
haar_cascade = cv.CascadeClassifier('haar_face.xml')


while True:
    _, frame = capture.read()    

    #Detect where Face is In frame
    display,coord = faceDetect(frame)
    cv.imshow('Canvas', cv.flip(display, 1))

    #Display Head Overlay on top of blank background
    blank = np.zeros((frame.shape), np.uint8)
    cv.imshow('Overlay', cv.flip(overlay(blank, head, coord), 1))

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destoryAllWindows()