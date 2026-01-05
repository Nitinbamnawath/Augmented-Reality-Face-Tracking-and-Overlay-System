import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
im = Image.open("C:/Users/Nitin/Documents/CODE/WiDS/Pepsi.jpg")
im_back = Image.open("C:/Users/Nitin/Documents/CODE/WiDS/pepsi_img.jpg")
imagecv2 = np.array(im)
image_backcv2 = np.array(im_back)

img = cv2.resize(imagecv2,(800,800))
img_back = cv2.resize(image_backcv2,(800,800))

mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (125, 125, 675, 675)

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2)|(mask == 0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]


x = int(input("Resize width: "))
y = int(input("Resize height: "))
mask_resize = cv2.resize(mask2,(x,y))
img_resize = cv2.resize(img,(x,y))

coordinates_x = int(input("Initial x coordinates: "))
coordinates_y = int(input("Initial y coordinates: "))

img_backcut = img_back[coordinates_y : coordinates_y + y , coordinates_x : coordinates_x + x]
img_backcut = img_backcut * ( 1 - mask_resize[:,:,np.newaxis] ) + img_resize

img_back[coordinates_y : coordinates_y + y , coordinates_x : coordinates_x + x] = img_backcut


plt.imshow(img_back)
plt.axis('off')
plt.show()

plt.imsave("final_grab.jpg", img_back)