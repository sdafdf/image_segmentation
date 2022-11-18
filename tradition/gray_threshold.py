import cv2
import matplotlib.pyplot as plt
img = cv2.imread("./test1.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#将灰度图中灰度值小于175的点置为0，灰度值大于175的点置为255
ret, thresh1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(gray,127,255,cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(gray,127,255,cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(gray,127,255,cv2.THRESH_TOZERO_INV)
title = ["img", "thresh1", "thresh2", "thresh3", "thresh4", "thresh5"]
imgaes = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(len(imgaes)):
    plt.subplot(2,3,i+1)
    plt.imshow(imgaes[i],"gray")
    plt.title(title[i])
    plt.xticks([])
    plt.yticks([])

plt.show()