import cv2
import mediapipe as mp
import numpy as np
import time
import dlib
import math
from PIL import Image
import facialLandmarkDetector as fld

def genEmpties(bg):
    height, width, _ = bg.shape

    empty = np.zeros((height, width, 4), dtype=np.uint8)
    cv2.imwrite('empty.jpg', empty)
    empty = cv2.imread('empty.jpg', cv2.IMREAD_UNCHANGED)
    empty = cv2.cvtColor(bg, cv2.COLOR_BGR2RGBA)
    return empty
    
def padToFit(x, y, empty, fg):
    height , length, _ = fg.shape

    for i in range(length):
        for j in range(height):
            for k in range(0,3):
                empty[y+j, x+i, k] = fg[j,i,k]
            empty[y+j, x+i, 3] = abs(255-fg[j,i,3])
    return empty


def overlayPng(bg, fg):
    fg_alpha = (255-fg[:,:,3])/255
    bg_alpha = (bg[:,:,3])/255
    copy = bg.copy()
    for color in range(0, 3):
        copy[:,:,color] = fg_alpha * fg[:,:,color] + (1-fg_alpha) * bg[:,:,color] * bg_alpha

    return copy

def findBB(coords):
    min = [999999,999999]
    max = [0,0]

    for x in coords[:,0]:
        if x < min[0]:
            min[0] = int(x) - 0
        if x > max[0]:
            max[0] = int(x) + 0

    for y in coords[:,1]:
        if y < min[1]:
            min[1] = int(y) - 0
        if y > max[1]:
            max[1] = int(y) + 0
    
    #max = [min[0] + 100, min[1] + 100]
    return min, max

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

width = int(cap.get(3))
height = int(cap.get(4))
bg = np.zeros((height, width, 4), dtype=np.uint8)
bg[:,:,0] = np.ones((height, width), dtype=np.uint8)*255.0
cv2.imwrite('background.jpg', bg)
bg = cv2.imread('background.jpg', cv2.IMREAD_UNCHANGED)
bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGBA)



emptyCanvas = genEmpties(bg)

#Load in Avatar Components

while cap.isOpened():
    emptyCanvas = genEmpties(bg)
    #Set up Background
    
    success, frame = cap.read()

    start = time.time()

    image = frame

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(image)
    
    # To improve performance
    image.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    upperLip = []
    lowerLip = []


    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                #if idx == 90 or idx == 180 or idx == 85 or idx == 16 or idx == 315 or idx == 404 or idx == 320 or idx == 74 or idx == 73 or idx == 72 or idx == 11 or idx == 302 or idx == 303 or idx == 304 or idx == 88 or idx == 178 or idx == 87 or idx == 14 or idx == 317 or idx == 402 or idx == 318 or idx == 42 or idx == 81 or idx == 82 or idx == 13 or idx == 312 or idx == 311 or idx == 310 or idx == 409 or idx == 291 or idx == 185 or idx == 61:
                ##upper lip
                #if idx == 74 or idx == 11 or idx == 304 or idx == 81 or idx == 13 or idx == 311:
                #    x, y = int(lm.x * img_w), int(lm.y * img_h)
                #    if idx == 74 or idx == 11 or idx == 304:
                #        y -= 10
                #    # Get the 2D Coordinates
                #    face_2d.append([x, y])
                #    cv2.circle(emptyCanvas, (x, y), 1, (0, 255, 255), 1)
                ##lower lip
                #if idx == 180 or idx == 16 or idx == 404 or  idx == 178 or idx == 14 or idx == 402:
                #    x, y = int(lm.x * img_w), int(lm.y * img_h)
                #    if idx == 90 or idx == 180 or idx == 85 or idx == 16 or idx == 315 or idx == 404 or idx == 320:
                #        y += 10
                #    # Get the 2D Coordinates
                #    face_2d.append([x, y])
                #    cv2.circle(emptyCanvas, (x, y), 1, (255, 255, 0), 1)
                #if idx == 291 or idx == 61:
                #    x, y = int(lm.x * img_w), int(lm.y * img_h)

                #upper lip
                if idx == 81 or idx == 13 or idx == 311:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    # Get the 2D Coordinates
                    upperLip.append([x, y])
                    face_2d.append([x,y])
                    cv2.circle(emptyCanvas, (x, y), 1, (0, 255, 255), 1)
                #lower lip
                if idx == 178 or idx == 14 or idx == 402:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    # Get the 2D Coordinates
                    lowerLip.append([x, y])
                    face_2d.append([x,y])
                    cv2.circle(emptyCanvas, (x, y), 1, (255, 255, 0), 1)
                if idx == 291 or idx == 61:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    upperLip.append([x, y])
                    lowerLip.append([x, y])
                    face_2d.append([x,y])

                    cv2.circle(emptyCanvas, (x, y), 1, (255, 255, 255), 1)
            
            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)
            upperLip = np.array(upperLip, dtype=np.float64)
            lowerLip = np.array(lowerLip, dtype=np.float64)

            upperLip = upperLip[upperLip[:, 0].argsort()]
            lowerLip = lowerLip[lowerLip[:, 0].argsort()]
            min, max= findBB(face_2d)
            cv2.polylines(emptyCanvas, np.int32([upperLip]), False, (255, 255, 0), 5)
            cv2.polylines(emptyCanvas, np.int32([lowerLip]), False, (255, 255, 0), 5)
            #cv2.line(emptyCanvas, (int(upperLip[0, 0]), int(upperLip[0, 1])), (int(upperLip[1, 0]), int(upperLip[1, 1])), (255, 255, 0), 1) 

            

        end = time.time()
        totalTime = end - start

        mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
        
    
    cv2.imshow('Head Pose Estimation', image)
    mouth = emptyCanvas[(min[1]-5):(max[1]+5), (min[0]-5):(max[0]+5)]
    #mouth = cv2.resize(mouth, (200,200))
    cv2.imshow('mouth', mouth)

    if cv2.waitKey(5) & 0xFF == 27:
        break


cap.release()

