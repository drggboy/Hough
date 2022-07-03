import numpy as np
def hough_drawlist(lines: np.ndarray, min_rho=10, min_theta=np.pi / 90):
    '''
        函数根据HoughLine函数返回参数进行作图，且自动剔除相近直线
        函数将把theta差值小于一定范围，并且在此基础上，rho差值小于一定范围的直线剔除
    '''
    draw_list = []  # 创建空列表用于存储已经作图的theta
    i = 0
    j = 0
    draw_flag = 1
    # 检查lines是否为空，否则作出第一条线
    if len(lines) != 0:
        rho, theta = lines[0, 0]  # 读取第一条直线数据
        i += 1
        draw_list.append([theta, rho])  # 在drew_list中存入第一个直线数据
    for rho, theta in lines[1:, 0]:  # lines是一个三维深度的数组，此处遍历每个[rho, theta]元素
        for past_line in draw_list:
            theta_error = abs(past_line[0] - theta)
            rho_error = abs(past_line[1] - rho)
            if theta_error <= min_theta and rho_error <= min_rho:  # 若两条直线的theta差值小于阈值
                j += 1
                draw_flag = 0
                break  # 跳出遍历
        # 画图
        if draw_flag == 1:
            draw_list.append([theta, rho])  # 存入drew_list
            i += 1
        else:
            draw_flag = 1  # 不作图，但恢复标志位
        # pre_rho = rho  # 保存参数
        # pre_theta = theta
    print(f'输入{len(lines)}条直线，共作{i}条直线，{j}条被合并')
    # print(drew_list)
    return draw_list