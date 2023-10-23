import cv2
import numpy as np

def img_show(title: str, img: np.ndarray, rate = 0.6):
    img_show = cv2.resize(img, None, fx=rate, fy=rate, interpolation=cv2.INTER_AREA)
    cv2.imshow(title, img_show)
    cv2.waitKey(0)


if __name__ == '__main__':
    img_src = np.load('./data/jiao_dian.npy')
    img_show('img_src',img_src)

    gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    thresh_value, img_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 膨胀
    k_p = np.ones((11,11), np.uint8)  # 指定膨胀核大小
    mask = cv2.morphologyEx(img_thresh, cv2.MORPH_DILATE, k_p)
    # gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # mask = np.zeros(gray.shape, np.uint8)
    img_show('mask',mask)


    # 开运算
    # k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # kernel大小，准备进行开运算
    # img_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)  # 开运算
    # img_show('img_open',img_open)


    # 轮廓筛选
    _, img_open_thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    img_open_thresh = np.array(img_open_thresh).astype(np.uint8)
    contours, hierarchy = cv2.findContours(img_open_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_copy = img_src.copy()
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        if w > 5 and h > 5:
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            cv2.circle(img_copy, (cx, cy), 10, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(img_copy, (cx, cy), 2, (0, 0, 255), -1, cv2.LINE_AA)
    img_show('img_copy',img_copy)
    cv2.destroyAllWindows()