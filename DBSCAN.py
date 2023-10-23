import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import preprocessing
from sklearn.cluster import DBSCAN

#计算两个向量之间的欧式距离
# def calDist(X1, X2):
#     sum = 0
#     for x1, x2 in zip(X1, X2):
#         sum += (x1 - x2) ** 2
#     return sum ** 0.5

#获取一个点的ε-邻域（记录的是索引）
# def getNeibor(data, dataSet, e):
#     res = []
#     for i in range(np.shape(dataSet)[0]):
#         if calDist(data, dataSet[i]) < e:
#             res.append(i)
#     return res

#密度聚类算法,不调用库实现
# def DBSCAN(dataSet, e, minPts):
#     coreObjs = {} #初始化核心对象集合
#     C = {}
#     n = np.shape(dataSet)[0]
#     #找出所有核心对象，key是核心对象的index，value是ε-邻域中对象的index
#     for i in range(n):
#         neibor = getNeibor(dataSet[i], dataSet, e)
#         if len(neibor) >= minPts:
#             coreObjs[i] = neibor
#     oldCoreObjs = coreObjs.copy()
#     k = 0 #初始化聚类簇数
#     notAccess = list(range(n))#初始化未访问样本集合（索引）
#     while len(coreObjs) > 0:
#         OldNotAccess = []
#         OldNotAccess.extend(notAccess)
#         cores = coreObjs.keys()
#         #随机选取一个核心对象
#         randNum = random.randint(0,len(cores)-1)
#         cores=list(cores)
#         core = cores[randNum]
#         queue = []
#         queue.append(core)
#         notAccess.remove(core)
#         while len(queue)>0:
#             q = queue[0]
#             del queue[0]
#             if q in oldCoreObjs.keys() :
#                 delte = [val for val in oldCoreObjs[q] if val in notAccess]#Δ = N(q)∩Γ
#                 queue.extend(delte)#将Δ中的样本加入队列Q
#                 notAccess = [val for val in notAccess if val not in delte]#Γ = Γ\Δ
#         k += 1
#         C[k] = [val for val in OldNotAccess if val not in notAccess]
#         for x in C[k]:
#             if x in coreObjs.keys():
#                 del coreObjs[x]
#     return C

# def draw(C, dataSet):
#     color = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
#     for i in C.keys():
#         X = []
#         Y = []
#         datas = C[i]
#         for j in range(len(datas)):
#             X.append(dataSet[datas[j]][0])
#             Y.append(dataSet[datas[j]][1])
#         plt.scatter(X, Y, marker='o', color=color[i % len(color)], label=i)
#     plt.legend(loc='upper right')
#     plt.show()

# def loadDataSet(filename):
#     dataSet = []
#     fr = open(filename)
#     for line in fr.readlines():
#         curLine = line.strip().split(',')
#         fltLine = map(float, curLine)
#         #python2.x map函数返回list
#         #dataSet.append(fltLine)
#         #python3.x map函数返回迭代器
#         dataSet.append(list(fltLine))
#     return dataSet

# 函数：返回距离质心最近的数据点。
def cen_point(ele):
    ele = np.array(ele)
    ele = np.reshape(ele, (-1, 2))  # 将输入的数据重新整形为二维数组
    mean = np.mean(ele, axis=0)  # 计算数据点的均值，即质心
    min_dist = np.linalg.norm(ele[0] - mean)  # 计算第一个数据点到质心的距离作为初始最小距离
    min_index = 0  # 记录最小距离对应的数据点索引
    for i in range(len(ele)):
        dist = np.linalg.norm(ele[i] - mean)  # 计算当前数据点到质心的距离
        if dist < min_dist:  # 如果距离更小，则更新最小距离和索引
            min_index = i
            min_dist = dist
    return ele[min_index]  # 返回距离质心最近的数据点

# 函数：计算DBSCAN聚类结果中每个簇的簇质心和离群点
def cen(scaler, y_pred):
    num = max(y_pred) + 1  # 计算聚类簇数
    ex = np.where(y_pred == -1)  # 找出离群点的索引，返回元组，ex[0]为列表，包含离群点索引

    type2 = []  # 存储每个簇的数据点
    ele = []  # 临时存储数据点
    for i in range(num):
        index = np.where(y_pred == i)  # 找出属于当前簇的数据点索引
        for k in index:
            ele.append(scaler[k])  # 将数据点添加到ele中
        type2.append(ele)  # 将当前簇的数据点列表添加到type2中
        ele = []  # 清空ele，准备存储下一个簇的数据点
    cen = []  # 存储簇质心和离群点
    for i in range(num):
        cen.append(cen_point(type2[i]))  # 计算并添加当前簇的簇质心
    for j in ex[0]:  # 添加离群点
        w = scaler[j]
        cen.append(w)
    return cen  # 返回簇质心和离群点的列表

# 函数：对给定的数据点进行DBSCAN聚类并返回聚类结果的簇质心和离群点
def DBSCAN_drawlist(lines: np.ndarray):
    dataSet = np.reshape(lines, (-1, 2))  # 重新整形输入数据为二维数组
    ss_X = preprocessing.StandardScaler()  # 创建一个标准化处理器对象
    scaler = ss_X.fit_transform(dataSet)  # 对数据进行标准化处理

    y_pred = DBSCAN(eps=0.07, min_samples=2).fit_predict(scaler)  # 使用DBSCAN进行聚类

    type1 = cen(scaler, y_pred)  # 调用cen函数计算簇质心和离群点
    type1 = np.array(type1)  # 将结果转换为NumPy数组
    lines_return = ss_X.inverse_transform(type1)  # 将簇质心和离群点的标准化结果逆转换为原始数据空间
    lines_return = np.array(lines_return)  # 转换为NumPy数组
    return lines_return  # 返回簇质心和离群点的坐标

# 主函数
def main():
    dataSet = np.load('data.npy')  # 从文件加载数据集
    scaler = preprocessing.StandardScaler().fit_transform(dataSet)  # 对数据进行标准化处理

    # 绘制原始数据点的散点图
    plt.subplot(221)
    plt.scatter(scaler[:, 0], scaler[:, 1], marker='o')

    # 使用DBSCAN聚类并绘制聚类结果的散点图
    plt.subplot(222)
    y_pred = DBSCAN(eps=0.07, min_samples=2).fit_predict(scaler)
    plt.scatter(scaler[:, 0], scaler[:, 1], c=y_pred)

    # 计算簇质心和离群点，并绘制它们的散点图
    type1 = cen(scaler, y_pred)
    type1 = np.array(type1)
    plt.subplot(223)
    plt.scatter(type1[:, 0], type1[:, 1], c=range(len(type1)))

    plt.show()  # 显示图形

    dataSet = scaler.tolist()  # 将数据集转换为列表格式
    # C = DBSCAN(dataSet, 0.11, 5)  # 基于旧的DBSCAN实现计算聚类结果，此部分被注释掉
    # draw(C, dataSet)  # 基于旧的绘制聚类结果的函数，此部分被注释掉

if __name__ == '__main__':
    main()  # 调用主函数
