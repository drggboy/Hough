import cv2
import numpy as np
## 参考：https://blog.csdn.net/QQ6550523/article/details/106420938

def trackChaned(self):
    pass

def slider_binary(img_src):
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    img_gray_show = cv2.resize(img_gray, (0, 0), fx=0.5, fy=0.5)
    cv2.namedWindow('thresh_binary')
    cv2.createTrackbar("thresh", "thresh_binary", 125, 255, trackChaned)
    while (True):
        thresh = cv2.getTrackbarPos("thresh", "thresh_binary")
        thresh_value, img_thresh = cv2.threshold(img_gray_show, thresh, 255, cv2.THRESH_BINARY)
        cv2.imshow("thresh_binary", img_thresh)
        if cv2.waitKey(1) == 27:  # esc
            break
    cv2.destroyAllWindows()
    _, img_thresh_raw = cv2.threshold(img_gray, int(thresh_value), 255, cv2.THRESH_BINARY)
    return img_thresh_raw

if __name__ == "__main__":
    img_path = r"img/test.jpg"
    img_src = cv2.imread(img_path)
    img_thresh = slider_binary(img_src)
    cv2.imshow("1",img_thresh)
    cv2.waitKey(0)
