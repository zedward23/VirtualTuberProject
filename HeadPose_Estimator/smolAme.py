import cv2
import mediapipe as mp
import numpy as np
import time
import dlib
import math
from PIL import Image
import facialLandmarkDetector as fld
import keyboard

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

def vec2DDist(p1, p2):
    return math.sqrt((float(p1[0])-float(p2[0])) ** 2. + (float(p1[1]-p2[1])) ** 2.)

def EAR(points):
    p1 = points[0]
    p2 = points[1]
    p3 = points[2]
    p4 = points[3]
    p5 = points[4]
    p6 = points[5]

    return (vec2DDist(p2, p6) + vec2DDist(p3, p5)) / 2. * vec2DDist(p1, p4)


currExpression = 0

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

#Avatar Filepaths
filepaths = [r"C:\Users\Edward\Desktop\VirtualTuberProject\HeadPose_Estimator\smol\eyes.png",
             r"C:\Users\Edward\Desktop\VirtualTuberProject\HeadPose_Estimator\smol\head.png",
             r"C:\Users\Edward\Desktop\VirtualTuberProject\HeadPose_Estimator\smol\blink.png",
             r"C:\Users\Edward\Desktop\VirtualTuberProject\HeadPose_Estimator\smol\background.png",
             r"C:\Users\Edward\Desktop\VirtualTuberProject\HeadPose_Estimator\smol\angry.png",
             r"C:\Users\Edward\Desktop\VirtualTuberProject\HeadPose_Estimator\smol\surprise.png",
             r"C:\Users\Edward\Desktop\VirtualTuberProject\HeadPose_Estimator\smol\smug.png",]
imgs = [cv2.imread(fp, cv2.IMREAD_UNCHANGED) for fp in filepaths]
index = 0

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

width = int(cap.get(3))
height = int(cap.get(4))

bg = np.zeros((height, width, 4), dtype=np.uint8)
bg[:,:,0] = np.ones((height, width), dtype=np.uint8)*255.0
cv2.imwrite('background.jpg', bg)
bg = cv2.imread('background.jpg', cv2.IMREAD_UNCHANGED)
bg = imgs[3]
bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGBA)

emptyCanvas = genEmpties(bg)
h,w,_ = emptyCanvas.shape
head = imgs[1]

emptyHead = genEmpties(imgs[1])
fullhead = padToFit(50,50, emptyHead.copy(), imgs[0])
fullhead = overlayPng(imgs[1], fullhead)

cv2.imshow('Head Pose Estimation', fullhead)
cv2.waitKey(20)

blink_thresh = 310
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("HeadPose_Estimator\shape_predictor_68_face_landmarks.dat")
points = [(0,0)]*6

l_eye_min = [999999,999999]
l_eye_max = [0,0]
r_eye_min = [999999.,999999.]
r_eye_max = [0,0]

ear1 = 0
#Load in Avatar Components

while cap.isOpened():
    #Set up Background
    success, frame = cap.read()

    start = time.time()

    image = frame
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)
    for face in faces:
        l_eye_min = [999999.,999999.]
        l_eye_max = [0,0]
        r_eye_min = [999999.,999999.]
        r_eye_max = [0,0]
        face_landmarks = dlib_facelandmark(gray, face)
        for n in range(36, 42):
            x = face_landmarks.part(n).x
            if (x < l_eye_min[0]):
                l_eye_min[0] = x
            if (x > l_eye_max[0]):
                l_eye_max[0] = x
            y = face_landmarks.part(n).y
            if (y < l_eye_min[1]):
                l_eye_min[1] = y
            if (y > l_eye_max[1]):
                l_eye_max[1] = y
            points[n-36] = (x,y)
            cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)
        ear1 = EAR(points)

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
                #upper lip
                if idx == 42 or idx == 13 or idx == 310:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    # Get the 2D Coordinates
                    upperLip.append([int(x*.3), int(y*.2)])
                #lower lip
                if idx == 88 or idx == 178 or idx == 14 or idx == 402 or idx == 318:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    # Get the 2D Coordinates
                    lowerLip.append([int(x*.3), int(y*.2)])
                if idx == 183 or idx == 415:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    upperLip.append([int(x*.3), int(y*.2)])
                    lowerLip.append([int(x*.3), int(y*.2)])

                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])       
            
            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360 - 10
            y = angles[1] * 360
            z = angles[2] * 360
          

            # See where the user's head tilting
            if y < -10:
                #"Looking Left"
                index = 0
            elif y > 10:
                #"Looking Right"
                index = 1
            else:
                index = 0

            # Display the nose direction
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
            
            cv2.line(image, p1, p2, (255, 0, 0), 3)

            upperLip = np.array(upperLip, dtype=np.float64)
            lowerLip = np.array(lowerLip, dtype=np.float64)

            upperLip = upperLip[upperLip[:, 0].argsort()]
            lowerLip = lowerLip[lowerLip[:, 0].argsort()]
            

        end = time.time()
        totalTime = end - start

        mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
        
    
    cv2.imshow('Head Pose Estimation', image)

    idx = currExpression
    if (ear1 < blink_thresh):
        idx = 2
    #fullhead = padToFit(int(100+2*y),int(120-2*x), emptyHead.copy(), imgs[idx])
    fullhead = padToFit(int(50+(y*.4)),50, emptyHead.copy(), imgs[idx])
    fullhead = overlayPng(imgs[1], fullhead)

    #position = np.array([[int(nose_2d[0]-100), int(nose_2d[1]-150)]])
    position = np.array([[int(220+y*.2), 280]])
    avatar = padToFit(position[0,0], position[0,1], emptyCanvas.copy(), fullhead)

    offsetX = int(92 + y*.2)
    offsetY = 120 

    position[0,0] += offsetX
    position[0,1] += offsetY
    upperLip = upperLip - upperLip[0] + position[0]
    lowerLip = lowerLip - lowerLip[0] + position[0]

    bg = cv2.imread('background.jpg', cv2.IMREAD_UNCHANGED)
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGBA)
    bg = imgs[3]

    avatar = overlayPng(bg, avatar)
    avatar = cv2.polylines(avatar, np.int32([upperLip]), False, (0, 0, 0), 4)
    avatar = cv2.polylines(avatar, np.int32([lowerLip]), False, (0, 0, 0), 4)
    cv2.imshow('Avatar', avatar)

    keyPress = cv2.waitKey(1) & 0xFF
    if keyPress == 27:
        print("esc pressed")
        break
    elif keyPress == 49:
        currExpression = 0
        print("1 pressed")
    elif keyPress == 50:
        currExpression = 4
        print("2 pressed")
    elif keyPress == 51:
        currExpression = 5
        print("3 pressed")
    elif keyPress == 52:
        currExpression = 6
        print("4 pressed")

cap.release()

