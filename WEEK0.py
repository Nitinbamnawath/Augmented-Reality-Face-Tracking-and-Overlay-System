import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
im = Image.open("image.jpg")
imagecv2 = np.array(im)
user_input = int(input("1.To convert RGB to BGR\n2.To sharpen the image\n3.To resize the image:\n"))

if user_input == 1 :
    print("RGB to BGR:")
    finalimage = cv2.cvtColor(imagecv2, cv2.COLOR_RGB2BGR)
elif user_input == 2 :
    print("Sharpening")
    kernel_sharpening = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    finalimage = cv2.filter2D(imagecv2, -1, kernel_sharpening)
elif user_input == 3 :
    x = int(input("Resize width:"))
    y = int(input("Resize height:"))
    finalimage = cv2.resize(imagecv2,(x,y))
else :
    print("incorrect input showing orignal image")
    finalimage = imagecv2

plt.imshow(finalimage)
plt.axis('off')
plt.show()

plt.imsave("final_output.jpg", finalimage)
