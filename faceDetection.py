import cv2 as cv
import numpy as py

capture = cv.VideoCapture(0)
haar_cascade = cv.CascadeClassifier('haar_face.xml')

while True:
    _, frame = capture.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_bounding_box = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 1)

    for (a,b,c,d) in face_bounding_box:
        cv.rectangle(frame, (a,b), (a+c, b+d), (0,255,0), thickness=2)


    cv.imshow('Video', cv.flip(frame, 1))

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destoryAllWindows()