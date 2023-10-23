import cv2
import numpy as np
from matplotlib import pyplot as plt

import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = r'img/test3.png'
img_src = cv2.imread(img_path, cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img_gray,(3,3),0)

edges = cv2.Canny(img, 50, 150, apertureSize = 3)
cv2.imshow('edges',edges)
cv2.waitKey(0)
lines = cv2.HoughLines(edges,1,np.pi/180,118)

result = img.copy()

#经验参数

minLineLength = 200

maxLineGap = 15

lines = cv2.HoughLinesP(edges,1,np.pi/180,80,minLineLength,maxLineGap)

# 4.显示直线
for line in lines:
    # rho, theta = line[0]
    rho = line[0]
    theta = line[1]
    a = np.cos(theta)
    b = np.sin(theta)

    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a)
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 5.显示图片
plt.subplot(121)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.subplot(122)
plt.imshow(img)
plt.axis("off")

plt.show()


# for x1,y1,x2,y2 in lines[0]:
#     cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
#
# cv2.imshow('Result', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()