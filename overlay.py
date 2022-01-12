import cv2
import numpy as np

background = cv2.imread('white.jpg')
overlay = cv2.imread('Head.png')


h, w = overlay.shape[:2]
shapes = np.zeros_like(background, np.uint8)
shapes[0:h, 0:w] = overlay
alpha = 0.8
mask = shapes.astype(bool)

# option first
background[mask] = cv2.addWeighted(shapes, alpha, shapes, 1 - alpha, 0)[mask]
cv2.imwrite('combined.png', background)
# option second
#background[mask] = cv2.addWeighted(background, alpha, overlay, 1 - alpha, 0)[mask]
# NOTE : above both option will give you image overlays but effect would be changed
cv2.imwrite('combined.1.png', background)

#rows,cols,channels = overlay.shape

#overlay=cv2.addWeighted(background[0:rows, 0:0+cols],0.5,overlay,0.5,0)

#background[0:rows, 0:0+cols ] = overlay
cv2.imshow("overlay", background)

while True:
    cv2.imshow('Video', background)

    if cv2.waitKey(20) & 0xFF==ord('d'):
        break

cv2.destoryAllWindows()