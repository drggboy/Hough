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

#密度聚类算法
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

# 函数返回距离质心最近的数据点。
def cen_point(ele):
    ele = np.array(ele)
    ele = np.reshape(ele,(-1,2))
    mean = np.mean(ele, axis=0)
    min_dist = np.linalg.norm(ele[0] - mean)
    min_index = 0
    for i in range(len(ele)):
        dist = np.linalg.norm(ele[i] - mean)
        if dist < min_dist:
            min_index = i
            min_dist = dist
    return ele[min_index]

def cen(scaler, y_pred):
    num = max(y_pred) + 1
    ex = np.where(y_pred == -1)    #返回的是元组，ex[0]为列表，其中包含离群点索引

    type2 = []
    ele = []
    for i in range(num):
        index = np.where(y_pred == i)
        for k in index:
            ele.append(scaler[k])
        type2.append(ele)
        ele = []
    cen = []
    # cen = np.zeros(shape=(num, 2))
    # cen = np.array([])
    for i in range(num):
        # cen[i,:] = cen_point(type2[i])
        # cen = np.append(cen,cen_point(type2[i]))
        cen.append(cen_point(type2[i]))
    # w = scaler[ex[0]]
    # cen.append(w)
    for j in ex[0]:
        w = scaler[j]
        # a = np.array(w)
        cen.append(w)
    return cen

def DBSCAN_drawlist(lines: np.ndarray):
    dataSet = np.reshape(lines,(-1,2))
    # dataSet=dataSet.tolist()
    ss_X = preprocessing.StandardScaler()
    scaler = ss_X.fit_transform(dataSet)

    # plt.subplot(221)
    # plt.scatter(scaler[:, 0], scaler[:, 1], marker='o')
    #
    # plt.subplot(222)
    y_pred = DBSCAN(eps=0.07, min_samples=2).fit_predict(scaler)
    # plt.scatter(scaler[:, 0], scaler[:, 1], c=y_pred)

    type1 = cen(scaler, y_pred)
    type1 = np.array(type1)
    lines_return = ss_X.inverse_transform(type1)
    lines_return = np.array(lines_return)
    return lines_return
    # plt.subplot(223)
    # plt.scatter(type1[:, 0], type1[:, 1], c=range(len(type1)))
    # plt.show()


def main():
    dataSet = np.load('data.npy')
    # dataSet=dataSet.tolist()
    scaler = preprocessing.StandardScaler().fit_transform(dataSet)

    plt.subplot(221)
    plt.scatter(scaler[:, 0], scaler[:, 1], marker='o')

    plt.subplot(222)
    y_pred = DBSCAN(eps=0.07, min_samples=2).fit_predict(scaler)
    plt.scatter(scaler[:, 0], scaler[:, 1], c=y_pred)

    type1 = cen(scaler, y_pred)
    type1 = np.array(type1)
    plt.subplot(223)
    plt.scatter(type1[:, 0], type1[:, 1], c=range(len(type1)))
    plt.show()

    dataSet = scaler.tolist()
    # # dataSet = loadDataSet("dataSet.txt")
    # print(dataSet)
    # C = DBSCAN(dataSet, 0.11, 5)
    # draw(C, dataSet)

if __name__ == '__main__':
    main()
