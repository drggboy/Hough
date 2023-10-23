import cv2
import numpy as np

def img_show(title: str, img: np.ndarray, rate = 0.6):
    img_show = cv2.resize(img, None, fx=rate, fy=rate, interpolation=cv2.INTER_AREA)
    cv2.imshow(title, img_show)
    cv2.waitKey(0)

# 读取图像文件
img = cv2.imread('img/test.jpg')

# 将图像转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 对灰度图像进行高斯模糊处理
gaus = cv2.GaussianBlur(gray, (3, 3), 0)
# img_show('gauss', gaus)

# 使用Canny边缘检测算法检测边缘
edges = cv2.Canny(gaus, 50, 150, apertureSize=3)

# 定义Hough线变换参数
minLineLength = 100
maxLineGap = 10

# 使用Hough线变换检测图像中的直线
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)

# 遍历检测到的每条线
for x1, y1, x2, y2 in lines[0]:
    # 在原始图像上绘制检测到的线
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 显示包含检测到的直线的图像
cv2.imshow("houghline", img)

# 等待用户按下键盘任意键，然后关闭窗口
cv2.waitKey()

# 关闭所有OpenCV窗口
cv2.destroyAllWindows()
