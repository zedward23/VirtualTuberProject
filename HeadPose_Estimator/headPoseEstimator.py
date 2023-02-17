import cv2
import mediapipe as mp
import numpy as np
import time
from PIL import Image

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

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

#Avatar Filepaths
filepaths = [r"C:\Users\Edward\Desktop\VirtualTuberProject\HeadPose_Estimator\imgs\forward.png", 
             r"C:\Users\Edward\Desktop\VirtualTuberProject\HeadPose_Estimator\imgs\side.png",
             r"C:\Users\Edward\Desktop\VirtualTuberProject\HeadPose_Estimator\imgs\eyes.png",
             r"C:\Users\Edward\Desktop\VirtualTuberProject\HeadPose_Estimator\imgs\head.png",
             r"C:\Users\Edward\Desktop\VirtualTuberProject\HeadPose_Estimator\imgs\Vtuber_head.png",
             r"C:\Users\Edward\Desktop\VirtualTuberProject\HeadPose_Estimator\imgs\Vtuber_eyes.png",
             r"C:\Users\Edward\Desktop\VirtualTuberProject\HeadPose_Estimator\imgs\Vtuber_mouth.png"]
imgs = [cv2.imread(fp, cv2.IMREAD_UNCHANGED) for fp in filepaths]
index = 0



cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

width = int(cap.get(3))
height = int(cap.get(4))
bg = np.zeros((height, width, 4), dtype=np.uint8)
bg[:,:,0] = np.ones((height, width), dtype=np.uint8)*255.0
cv2.imwrite('background.jpg', bg)
bg = cv2.imread('background.jpg', cv2.IMREAD_UNCHANGED)
bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGBA)

emptyCanvas = genEmpties(bg)
h,w,_ = emptyCanvas.shape
head = cv2.resize(imgs[4], (200,200))

emptyHead = genEmpties(imgs[4])
mouth = imgs[6] #cv2.resize(imgs[6], (390,390))

headWithMouth = padToFit(0,0, emptyHead.copy(), mouth)
headWithMouth = overlayPng(imgs[4], headWithMouth)

fullhead = padToFit(100,100, emptyHead.copy(), imgs[5])
fullhead = overlayPng(headWithMouth, fullhead)

#headWithMouth = cv2.resize(fullhead, (200,200))

#cv2.imshow("display", headWithMouth)
#cv2.waitKey(200)


#Load in Avatar Components

while cap.isOpened():
    #Set up Background
    
    success, image = cap.read()

    start = time.time()

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

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
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


            #if x < -10:
                #"Looking Down"
            #elif x > 10:
                #"Looking Up"
            

            # Display the nose direction
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
            
            cv2.line(image, p1, p2, (255, 0, 0), 3)

        end = time.time()
        totalTime = end - start

        #mp_drawing.draw_landmarks(
        #            image=image,
        #            landmark_list=face_landmarks,
        #            connections=mp_face_mesh.FACEMESH_CONTOURS,
        #            landmark_drawing_spec=drawing_spec,
        #            connection_drawing_spec=drawing_spec)
    

    cv2.imshow('Head Pose Estimation', image)

    fullhead = padToFit(int(100+2*y),int(120-2*x), emptyHead.copy(), imgs[5])
    fullhead = overlayPng(headWithMouth, fullhead)
    fullhead = cv2.resize(fullhead, (200,200))

    avatar = padToFit(int(nose_2d[0]-100), int(nose_2d[1]-150), emptyCanvas.copy(), fullhead)



    cv2.imshow('Avatar', overlayPng(bg, avatar))

    if cv2.waitKey(5) & 0xFF == 27:
        break


cap.release()