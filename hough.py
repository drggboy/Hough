import cv2
import matplotlib.pyplot as plt
import numpy as np
import DBSCAN

# 显示图片函数
def img_show(title: str, img: np.ndarray):
    img_show = cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    cv2.imshow(title, img_show)
    cv2.waitKey(0)

## HSV ##
# 回调函数
def on_color_change(param):
    pass

# hsv过滤，返回感兴趣区域
def hsv_filter(img, tune = False):
    if tune == True:
        cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow('image', 600, 400)   #调整滑块窗口大小
        cv2.moveWindow('image', 0, 0)

        # 创建颜色变化的轨迹栏
        cv2.createTrackbar('Hmin','image',15,180,on_color_change)
        cv2.createTrackbar('Hmax','image',31,180,on_color_change)
        cv2.createTrackbar('Smin','image',27,255,on_color_change)
        cv2.createTrackbar('Smax','image',255,255,on_color_change)
        cv2.createTrackbar('Vmin','image',173,255,on_color_change)
        cv2.createTrackbar('Vmax','image',255,255,on_color_change)

        while True:
            hmin = cv2.getTrackbarPos('Hmin','image')
            hmax = cv2.getTrackbarPos('Hmax','image')
            smin = cv2.getTrackbarPos('Smin','image')
            smax = cv2.getTrackbarPos('Smax','image')
            vmin = cv2.getTrackbarPos('Vmin','image')
            vmax = cv2.getTrackbarPos('Vmax','image')

            # 获得掩码
            lower=np.array([hmin,smin,vmin])
            upper=np.array([hmax,smax,vmax])
            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv,lower,upper)

            # 显示掩码内的图像
            img_obj = cv2.bitwise_and(img,img,mask=mask)
            img_obj_show = cv2.resize(img_obj, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
            cv2.imshow('obj',img_obj_show)

            ch = cv2.waitKey(1) & 0xFF
            if ch == 27:  # 按下esc键
                cv2.destroyAllWindows()
                return img_obj
    else:
        lower = np.array([15, 27, 173])
        upper = np.array([31, 255, 255])
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        img_obj = cv2.bitwise_and(img, img, mask=mask)
        return img_obj

# 用于绘制轮廓面积折线图
def line_chart(list):
    plt.plot(list,'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='xxx')
    plt.legend(loc="upper right")
    plt.xlabel('index')
    plt.ylabel('value')
    plt.show()

# 轮廓检测，输入待检测图片、用于绘制轮廓的图片，输出绘制后图片
def cnt_detector(roi_binary, img_drawed):
    contours, hierarchy = cv2.findContours(roi_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnt_area = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        cnt_area.append(area)
    ## 观察面积分布，以确定过滤参数 ##
    # 绘制面积折现图
    # line_chart(cnt_area)

    ## 对轮廓进行过滤 ##
    contours_draw = []
    painted = []
    for i in range(len(cnt_area)):
        current = len(cnt_area) - i - 1
        # if cnt_area[current]>30000 and cnt_area[current]<50000:
        if cnt_area[current] > 15000:
            if current in painted:
                continue
            painted.append(hierarchy[0][current][3])
            painted.append(current)
            contours_draw.append(contours[current])
    ## 绘制轮廓 ##
    img_contours = cv2.drawContours(img_drawed, contours_draw, -1, (0, 255, 0), 2)
    return img_contours

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
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return img

# 密度聚类前后散点图绘制
def scatter_compared(data1,data):
    x_rho_raw = data1[:, 0]
    y_theta_raw = data1[:, 1]
    plt.subplot(121)
    plt.title('Result Analysis')
    plt.xlabel('rho')  # 横坐标轴标题
    plt.ylabel('theta')  # 纵坐标轴标题
    plt.scatter(x_rho_raw, y_theta_raw, c='k', marker='.')

    x_rho = data[:, 0]
    y_theta = data[:, 1]
    plt.subplot(122)
    plt.title('Result Analysis')
    plt.xlabel('rho')  # 横坐标轴标题
    plt.ylabel('theta')  # 纵坐标轴标题
    plt.scatter(x_rho, y_theta, c='k', marker='.')
    plt.show()

# 使用plt显示四张图片
def plt_four(img1,img2,img3,img4):
    ## 显示图片 ##
    # 原图
    img_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    plt.subplot(221)
    plt.imshow(img_rgb)
    plt.axis("off")

    # 车位线提取
    im_zclose_show = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    plt.subplot(222)
    plt.imshow(im_zclose_show)
    plt.axis("off")

    # plt.subplot(223)
    # plt.imshow(img_edges, cmap="binary")   #单通道图才能使用cmap影响

    # 未过滤直线检测
    img_blue_line_raw = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    plt.subplot(223)
    plt.imshow(img_blue_line_raw)  # 单通道图才能使用cmap影响
    plt.axis("off")

    # 过滤后
    img_blue_line_filted = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
    plt.subplot(224)
    plt.imshow(img_blue_line_filted)
    plt.axis("off")

    plt.show()



if __name__ == '__main__':
    # 图片路径
    # img_path = r'img/test.jpg'
    img_path = r'img/pers_trans.jpg'
    # img_path = r'img/test2.png'

    # 读取图片
    img_src = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # 获得车道线图像
    img_obj = hsv_filter(img_src, tune=False)
    img_show('img_obj',img_obj)

    ## 车位线图像处理 ##
    # 开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # kernel大小，准备进行开运算
    img_open = cv2.morphologyEx(img_obj, cv2.MORPH_OPEN, kernel, iterations=2)  # 开运算
    img_show('img_open',img_open)

    # 横向闭运算
    kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 2))
    im_hclose = cv2.morphologyEx(img_open,cv2.MORPH_CLOSE,kernel_d,iterations=1)
    img_show('im_close', im_hclose)

    # 纵向闭运算
    kernel_z = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,21))
    im_zclose = cv2.morphologyEx(im_hclose,cv2.MORPH_CLOSE,kernel_z,iterations=1)
    img_show('im_close', im_zclose)

    # 十字闭运算
    kernel_c2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (41, 41))
    im_cross_close = cv2.morphologyEx(im_zclose,cv2.MORPH_CLOSE,kernel_c2,iterations=1)
    img_show('im_cross_close', im_cross_close)

    # 十字闭运算图片减去运算前图片，获得方块角点
    sub = im_cross_close - im_zclose
    img_show('sub',sub)

    # 二值化纵横闭运算图
    roi_gray = cv2.cvtColor(im_zclose,cv2.COLOR_BGR2GRAY)
    thresh,roi_binary = cv2.threshold(roi_gray,50,255,cv2.THRESH_BINARY)
    img_show('roi_binary',roi_binary)

    ## 使用二值图进行轮廓检测 ##
    img_contours = cnt_detector(roi_binary, im_zclose)
    img_show('contours', img_contours)

    ## 霍夫曼直线检测 ##
    # 执行canny边缘检测
    img_edges = cv2.Canny(roi_binary, 30, 110, apertureSize=3)
    img_show('img_edges', img_edges)

    lines = cv2.HoughLines(img_edges, 1, np.pi / 180, 90)
    # 转化为n行2列的 ndarray数组
    hough_lines = np.array(lines).reshape(-1,2)
    img_blue_line_raw = Hough_lines_draw(img_src, hough_lines)

    filted_hough_lines = DBSCAN.DBSCAN_drawlist(hough_lines)
    img_blue_line_filted = Hough_lines_draw(img_src,filted_hough_lines)

    # 过滤前后散点图对比
    scatter_compared(hough_lines,filted_hough_lines)

    # 显示图像
    plt_four(img_src, im_zclose, img_blue_line_raw, img_blue_line_filted)


