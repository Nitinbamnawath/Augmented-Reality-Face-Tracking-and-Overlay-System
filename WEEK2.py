import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import warnings
warnings.filterwarnings("ignore")

# Face Mesh

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture("video.mp4")


#overlay

im = Image.open("Pepsi.jpg")
imagecv2 = np.array(im)
img = cv2.cvtColor(imagecv2, cv2.COLOR_RGB2BGR)
im_y,im_x,__ = imagecv2.shape

#something

mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (50, 50, im_x-50, im_y-50)

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2)|(mask == 0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

while True:
    # Image
    ret, image = cap.read()
    if ret is not True:
        break
    height, width, _ = image.shape
    print("Height, width", height, width)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    # Facial landmarks
    result = face_mesh.process(rgb_image)

    for facial_landmarks in result.multi_face_landmarks:
    
        pt1 = facial_landmarks.landmark[4]
        pt2 = facial_landmarks.landmark[243] #left cheek
        pt3 = facial_landmarks.landmark[454] #right cHeek
        x1 = int(pt1.x * width)
        y1 = int(pt1.y * height)
        x2 = int(pt2.x * width)
        y2 = int(pt2.y * height)
        x3 = int(pt3.x * width)
        y3 = int(pt3.y * height)

        x_ = x3-x2
        y_ = int(x_ * (im_y / im_x))
        mask_resize = cv2.resize(mask2,(x_,y_))
        img_resize = cv2.resize(img,(x_,y_))

        coordinates_x = int(x1 - x_/2)
        coordinates_y = int(y1 - y_/2)

        img_backcut = image[coordinates_y : coordinates_y + y_ , coordinates_x : coordinates_x + x_]
        img_backcut = img_backcut * ( 1 - mask_resize[:,:,np.newaxis] ) + img_resize

        image[coordinates_y : coordinates_y + y_ , coordinates_x : coordinates_x + x_] = img_backcut

            #cv2.circle(image, (x, y), 2, (100, 100, 0), -1)
            #cv2.putText(image, str(i), (x, y), 0, 1, (0, 0, 0))

    cv2.imshow("Image", image)
    cv2.waitKey(1)