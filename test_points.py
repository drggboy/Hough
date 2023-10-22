import cv2
import numpy as np

def img_show(title: str, img: np.ndarray, rate = 0.6):
    img_show = cv2.resize(img, None, fx=rate, fy=rate, interpolation=cv2.INTER_AREA)
    cv2.imshow(title, img_show)
    cv2.waitKey(0)

# 霍夫直线绘制
def Hough_lines_draw(img_src, lines):
    img = np.copy(img_src)
    for line in lines:
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 2000 * (-b))
        y1 = int(y0 + 2000 * a)
        x2 = int(x0 - 2000 * (-b))
        y2 = int(y0 - 2000 * a)
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return img

if __name__ == '__main__':
    img_path = r'img/pers_trans.jpg'
    img_src = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # 绘制直线
    filted_hough_lines = np.load('filted_lines.npy')
    img_drawed = np.zeros_like(img_src)
    img_lines_only = Hough_lines_draw(img_drawed, filted_hough_lines)
    # img_show('img_lines_only', img_lines_only)

    # 角点检测
    gray = cv2.cvtColor(img_lines_only, cv2.COLOR_BGR2GRAY)
    img_show('1',gray)
    mask = np.zeros(gray.shape, np.uint8)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    thresh_value, img_thresh = cv2.threshold(dst, 0.01, 255, cv2.THRESH_BINARY)
    img_show('thresh', img_thresh)

    # 膨胀
    k_p = np.ones((5, 5), np.uint8)  # 指定膨胀核大小
    mask = cv2.morphologyEx(img_thresh, cv2.MORPH_DILATE, k_p)
    img_show('mask',mask)

    # 开运算
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # kernel大小，准备进行开运算
    img_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)  # 开运算
    img_show('img_open',img_open)


    # 轮廓筛选
    _, img_open_thresh = cv2.threshold(img_open, 127, 255, cv2.THRESH_BINARY)
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