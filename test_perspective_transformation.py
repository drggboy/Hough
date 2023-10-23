# 透视变换
from cv2 import cv2
import numpy as np

# 显示图片
def img_show(title: str, img: np.ndarray):
    img = cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    cv2.imshow(title, img)
    cv2.waitKey(0)

def pers_trans(img):
    h, w = img.shape[0:2]
    # 起始坐标
    org = np.array([[611, 0],
                    [1920, 0],
                    [1530, 1080],
                    [0, 1080]], np.float32)
    # 目标坐标
    dst = np.array([[0, 0],
                    [w, 0],
                    [w, h],
                    [0, h]], np.float32)

    warpR = cv2.getPerspectiveTransform(org, dst)           # 透视变换矩阵
    result = cv2.warpPerspective(img, warpR, (w, h))        # 透视变换输出图像
    return result


if __name__ == '__main__':
    img = cv2.imread('img/parking9.jpg')
    result = pers_trans(img)
    img_show("img", img)
    img_show("result", result)



