# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 09:32:02 2016

@author: http://blog.csdn.net/lql0716
"""
import cv2
import numpy as np

# 显示图片函数
def img_show(title: str, img: np.ndarray, rate = 0.6):
    img_show = cv2.resize(img, None, fx=rate, fy=rate, interpolation=cv2.INTER_AREA)
    cv2.imshow(title, img_show)
    cv2.waitKey(0)

# current_pos = None
# tl = None
# br = None
#鼠标事件
def get_rect(im, title='get_rect'):   #   (a,b) = get_rect(im, title='get_rect')
    mouse_params = {'tl': None, 'br': None, 'current_pos': None,
        'released_once': False}

    cv2.namedWindow(title)
    cv2.moveWindow(title, 100, 100)

    def onMouse(event, x, y, flags, param):

        param['current_pos'] = (x, y)

        if param['tl'] is not None and not (flags & cv2.EVENT_FLAG_LBUTTON):
            param['released_once'] = True

        if flags & cv2.EVENT_FLAG_LBUTTON:
            if param['tl'] is None:
                param['tl'] = param['current_pos']
            elif param['released_once']:
                param['br'] = param['current_pos']

    cv2.setMouseCallback(title, onMouse, mouse_params)
    cv2.imshow(title, im)

    while mouse_params['br'] is None:
        im_draw = np.copy(im)

        if mouse_params['tl'] is not None:
            cv2.rectangle(im_draw, mouse_params['tl'],mouse_params['current_pos'], (255, 0, 0))

        cv2.imshow(title, im_draw)
        _ = cv2.waitKey(10)

    cv2.destroyWindow(title)

    tl = (min(mouse_params['tl'][0], mouse_params['br'][0]),
        min(mouse_params['tl'][1], mouse_params['br'][1]))
    br = (max(mouse_params['tl'][0], mouse_params['br'][0]),
        max(mouse_params['tl'][1], mouse_params['br'][1]))

    return (tl, br)  #tl=(y1,x1), br=(y2,x2)


def get_qd(img, rate = 0.6):
    img_resize = cv2.resize(img, None, fx=rate, fy=rate, interpolation=cv2.INTER_AREA)
    (a, b) = get_rect(img_resize, title='get_rect')
    a, b = np.array(a), np.array(b)
    a_true = np.round(a/rate).astype(np.uint32)
    b_true = np.round(b/rate).astype(np.uint32)
    return  a_true,b_true

def img_qd(img_src,tune = False, a = [482, 633], b = [1018, 965]):
    if tune == True:
        (a, b) = get_qd(img_src)
    print("目标圈定a=",a)
    print("目标圈定b=",b)
    ax, ay = a
    bx, by = b
    mask = np.zeros((img_src.shape[0], img_src.shape[1])).astype(np.uint8)
    mask[ay:by, ax:bx] = 1
    img_obj = cv2.bitwise_and(img_src, img_src, mask=mask)
    return img_obj


# 删去不感兴趣区域，按回车键进行下一次删除，按esc键结束
def img_Sub(img_src,tune = False, A = [[482, 633]], B = [[1018, 965]]):
    mask = np.ones((img_src.shape[0], img_src.shape[1])).astype(np.uint8)
    if tune == True:
        A = []
        B = []
        img_drawing = img_src
        while True:
            (a, b) = get_qd(img_drawing)
            A.append(a)
            B.append(b)
            ax, ay = a
            bx, by = b
            mask[ay:by, ax:bx] = 0
            img_obj = cv2.bitwise_and(img_src, img_src, mask=mask)
            img_drawing = img_obj
            img_obj_show = cv2.resize(img_obj, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
            cv2.imshow("img_obj",img_obj_show)
            ch = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if ch == 27:  # 按下esc键
                cv2.destroyAllWindows()
                break
    else:
        for i,a in enumerate(A):
            ax, ay = a
            bx, by = B[i]
            mask[ay:by, ax:bx] = 0
        img_obj = cv2.bitwise_and(img_src, img_src, mask=mask)
    return img_obj,A,B
        # print("sub_a=",a)
        # print("sub_b=",b)
    # ax, ay = a
    # bx, by = b
    # mask = np.ones((img_src.shape[0], img_src.shape[1])).astype(np.uint8)
    # mask[ay:by, ax:bx] = 0
    # img_obj = cv2.bitwise_and(img_src, img_src, mask=mask)
    # return img_obj


if __name__ == '__main__':
    img_path = r'img/pers_trans.jpg'
    img_src = cv2.imread(img_path, cv2.IMREAD_COLOR)

    img_obj = img_qd(img_src)
    # img_show("img_obj",img_obj)

    img_sub,A,B= img_Sub(img_obj, tune=True)
    img_show("img_sub", img_sub)
    img_new,_,_ = img_Sub(img_obj, tune=False,A=A,B =B)
    img_show("img_new", img_new)
