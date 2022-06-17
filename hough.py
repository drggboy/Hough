import cv2
import matplotlib.pyplot as plt
import numpy as np


# 1.导入图片
img_path = r'img/test.jpg'
# img_path = r'img/test2.png'
img_src = cv2.imread(img_path, cv2.IMREAD_COLOR)
# img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
# cv2.imshow('img_gray',img_gray)
# cv2.waitKey(0)

img_rgb = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
img_show = img_rgb.copy()
# HSV
# def on_color_change(param):
#     pass
#
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# # cv2.resizeWindow('image',500,250)
# cv2.moveWindow('image',0,0)
# # 创建颜色变化的轨迹栏
# cv2.createTrackbar('Hmin','image',15,180,on_color_change)
# cv2.createTrackbar('Hmax','image',66,180,on_color_change)
# cv2.createTrackbar('Smin','image',27,255,on_color_change)
# cv2.createTrackbar('Smax','image',255,255,on_color_change)
# cv2.createTrackbar('Vmin','image',173,255,on_color_change)
# cv2.createTrackbar('Vmax','image',255,255,on_color_change)
#
# while True:
#     hmin = cv2.getTrackbarPos('Hmin','image')
#     hmax = cv2.getTrackbarPos('Hmax','image')
#     smin = cv2.getTrackbarPos('Smin','image')
#     smax = cv2.getTrackbarPos('Smax','image')
#     vmin = cv2.getTrackbarPos('Vmin','image')
#     vmax = cv2.getTrackbarPos('Vmax','image')
#
#     lower=np.array([hmin,smin,vmin])
#     upper=np.array([hmax,smax,vmax])
#     hsv = cv2.cvtColor(img_src,cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv,lower,upper)
#     #
#     # im1 = img_src.copy()
#     # im1[mask > 0] = [255, 0, 0]
#     #
#     # mask3 = cv2.merge([mask, mask, mask])
#     # im1 = cv2.bitwise_and(img_src,mask3)
#     #
#     im1 = cv2.bitwise_and(img_src,img_src,mask=mask)
#     cv2.imshow('yello',im1)
#
#     ch = cv2.waitKey(1) & 0xFF
#     if ch == 27:  # 按下回车键
#         break


lower=np.array([15,27,173])
upper=np.array([31,255,255])
hsv = cv2.cvtColor(img_src,cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv,lower,upper)
im1 = cv2.bitwise_and(img_src, img_src, mask=mask)
# cv2.imshow('roi',im1)
# cv2.waitKey(0)


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # kernel大小，准备进行开运算
im2 = cv2.morphologyEx(im1, cv2.MORPH_OPEN, kernel)  # 开运算
cv2.imshow('open', im2)
cv2.waitKey(0)
# gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray)
# cv2.waitKey(0)
kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 2))
# 腐蚀操作
# dilated = cv2.dilate(im2, kernel_d,iterations = 2)
# dilated = cv2.dilate(im2, kernel_d,1)
# dilated = cv2.dilate(dilated, kernel_d,1)
# cv2.imshow('dilated', dilated)
# cv2.waitKey(0)
# # 十字闭运算
# kernel_c = cv2.getStructuringElement(cv2.MORPH_CROSS, (41, 41))
# im_close = cv2.morphologyEx(im2,cv2.MORPH_CLOSE,kernel_c,iterations=1)
# cv2.imshow('im_close', im_close)
# cv2.waitKey(0)
# im_zclose = im_close
# im_zclose_show = cv2.cvtColor(im_zclose,cv2.COLOR_BGR2RGB)
#横向闭运算
im_hclose = cv2.morphologyEx(im2,cv2.MORPH_CLOSE,kernel_d,iterations=1)
cv2.imshow('im_close', im_hclose)
cv2.waitKey(0)
#纵向闭运算
kernel_z = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,21))
im_zclose = cv2.morphologyEx(im_hclose,cv2.MORPH_CLOSE,kernel_z,iterations=1)
cv2.imshow('im_close', im_zclose)
cv2.waitKey(0)
im_zclose_show = cv2.cvtColor(im_zclose,cv2.COLOR_BGR2RGB)


# 二值化
roi_gray = cv2.cvtColor(im_zclose,cv2.COLOR_BGR2GRAY)
thresh,roi_binary = cv2.threshold(roi_gray,50,255,cv2.THRESH_BINARY)
cv2.imshow('roi_binary',roi_binary)
cv2.waitKey(0)
# thresh_otsu, binary_image = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow('gray_roi',roi_gray)
# cv2.waitKey(0)
# 2.执行canny
img_edges = cv2.Canny(roi_binary, 30, 110, apertureSize=3)
cv2.imshow('img_edges', img_edges)
cv2.waitKey(0)


# 3.霍夫曼直线检测
lines = cv2.HoughLines(img_edges, 1, np.pi / 180, 90)

# 4.显示直线
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)

    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 2000 * (-b))
    y1 = int(y0 + 2000 * a)
    x2 = int(x0 - 2000 * (-b))
    y2 = int(y0 - 2000 * a)
    cv2.line(img_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 5.显示图片
plt.subplot(221)
plt.imshow(img_show)
plt.axis("off")
plt.subplot(222)
plt.imshow(im_zclose_show)
plt.axis("off")
plt.subplot(223)
plt.imshow(img_edges, cmap="binary")   #单通道图才能使用cmap影响
plt.axis("off")
plt.subplot(224)
plt.imshow(img_rgb)
plt.axis("off")




plt.show()

