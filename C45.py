#coding:utf-8

from math import log
import math
import numpy as np
import pandas as pd
import operator

np.set_printoptions(precision = 3)      # 设置numpy输出的小数为3位, 否则精度会非常大, 导致溢出


# 通过排序返回出现次数最多的类别
def majorityCnt(trainLabel):
    labelCount = {}             # 统计每个类别的个数
    for i in trainLabel:
        if i not in labelCount.keys(): 
            labelCount[i] = 0
        labelCount[i] += 1
    sortedlabelCount = sorted(labelCount.items(), key=operator.itemgetter(1), reverse=True) # 将类别数降序排列
    return sortedlabelCount[0][0]    


# 计算信息熵
def calEnt(trainLabel):
    numEntries = len(trainLabel)            # 样本数D
    labelCount = {}
    for i in trainLabel:  
        if i not in labelCount.keys():      # 统计每个类别的个数
            labelCount[i] = 0
        labelCount[i] += 1
    Ent = 0.0
    for key in labelCount:  
        p = float(labelCount[key]) / numEntries
        Ent = Ent - p * log(p, 2)
    return Ent


# 划分数据集, 参数为数据集、划分特征、划分值 
def splitDataSet_c(trainData, trainLabel, feature, value):
    trainDataLeft = []
    trainDataRight= []
    trainLabelLeft = []
    trainLabelRight = []

    for i in range(len(trainData)):
        if float(trainData[i][feature]) <= value:
            trainDataLeft.append(trainData[i])
            trainLabelLeft.append(trainLabel[i])
        else:
            trainDataRight.append(trainData[i])
            trainLabelRight.append(trainLabel[i])

    return trainDataLeft, trainDataRight, trainLabelLeft, trainLabelRight 


# 选择最好的数据集划分方式
def chooseBestFeatureToSplit_c(trainData, trainLabel):

    featureNum = len(trainData[0])          # 特征数
    baseEnt = calEnt(trainLabel)            # 计算信息熵
    bestGainRatio = 0.0
    bestFeature = -1
    bestPartValue = None 
    IV = 0.0

    for feature in range(featureNum):           # 对每个特征, 计算信息增益率
        featList = [i[feature] for i in trainData]
        uniqueVals = set(featList)              # 去重
        bestPartValuei = None

        sortedUniqueVals = list(uniqueVals)     # 升序排列
        sortedUniqueVals.sort()

        minEnt = float("inf")
        for i in range(len(sortedUniqueVals) - 1):      
            partValue = (sortedUniqueVals[i] + sortedUniqueVals[i + 1]) / 2             # 计算划分点

            (trainDataLeft, trainDataRight, trainLabelLeft, trainLabelRight) = splitDataSet_c(trainData, trainLabel, feature, partValue)     # 对每个划分点, 计算ΣEnt(D^v)
            pLeft = len(trainDataLeft) / float(len(trainData))
            pRight = len(trainDataRight) / float(len(trainData))
            Ent = pLeft * calEnt(trainLabelLeft) + pRight * calEnt(trainLabelRight)     # 计算ΣEnt(D^v)
   
            
            if Ent < minEnt:        # ΣEnt(D^v)越小, 则信息增益Gain = Ent(D) - ΣEnt(D^v)越大
                minEnt = Ent
                IV = -(pLeft * log(pLeft, 2) + pRight * log(pRight, 2))                 # 计算IV
                bestPartValuei = partValue

        Gain = baseEnt - minEnt     # 计算信息增益Gain
        GainRatio = Gain / IV       # 计算信息增益率GainRatio

        if GainRatio > bestGainRatio:       # 取最大的信息增益率对应的特征
            bestGainRatio = GainRatio
            bestFeature = feature
            bestPartValue = bestPartValuei

    return bestFeature, bestPartValue


# 创建树
# @params list类型的m*n样本, list类型的1*m分类标签
def createTree_c(trainData, trainLabel):
    if trainLabel.count(trainLabel[0]) == len(trainLabel):  # 如果只有一个类别，返回该类别
        return {'label': trainLabel[0]}

    bestFeat, bestPartValue = chooseBestFeatureToSplit_c(trainData, trainLabel)     # 获取最优划分特征的索引, 以及该特征的划分值
    myTree = {'feature': bestFeat, 'value': bestPartValue}

    (trainDataLeft, trainDataRight, trainLabelLeft, trainLabelRight) = splitDataSet_c(trainData, trainLabel, bestFeat, bestPartValue)


    # 构建左子树
    myTree['leftTree'] = createTree_c(trainDataLeft, trainLabelLeft)
    # 构建右子树
    myTree['rightTree'] = createTree_c(trainDataRight, trainLabelRight)
    return myTree


# 测试算法
def classify_c(myTree, testData):
    if 'label' in myTree.keys():        # 叶节点
        return myTree['label']
    feature = myTree['feature']
    partValue = myTree['value']
    if testData[feature] <= partValue:
        return classify_c(myTree['leftTree'], testData)
    else: 
        return classify_c(myTree['rightTree'], testData)



# 后剪枝
def postPruningTree(myTree, trainData, trainLabel, verifyData, verifyLabel):

    if 'label' in myTree.keys():    # 叶节点
        return myTree

    (trainDataLeft, trainDataRight, trainLabelLeft, trainLabelRight) = splitDataSet_c(trainData, trainLabel, myTree['feature'], myTree['value'])
    (verifyDataLeft, verifyDataRight, verifyLabelLeft, verifyLabelRight) = splitDataSet_c(verifyData, verifyLabel, myTree['feature'], myTree['value'])

    myTree['leftTree'] = postPruningTree(myTree['leftTree'], trainDataLeft, trainLabelLeft, verifyDataLeft, verifyLabelLeft)            # 对左子树剪枝
    myTree['rightTree'] = postPruningTree(myTree['rightTree'], trainDataRight, trainLabelRight, verifyDataRight, verifyLabelRight)      # 对右子树剪枝

    predict = []                            # 预测结果
    for i in verifyData:
        predict.append(classify_c(myTree, i))

    majorLabel = majorityCnt(trainLabel)    # 选取最多的类别来进行剪枝判断

    error1 = 0.0
    error2 = 0.0

    for i in range(len(verifyLabel)):       # 计算剪枝与不剪枝的误差
        error1 = error1 + (predict[i] - verifyLabel[i])**2
        error2 = error2 + (majorLabel - verifyLabel[i])**2

    if error1 <= error2:                    # 若不剪枝误差小
        return myTree

    return {'label': majorityCnt(trainLabel)}   
    

if __name__ == '__main__':
    trainData = pd.read_csv('trainData.csv', encoding = 'utf-8', header = None)     # 读取trainData, 并返回DataFrame
    trainData = np.array(trainData)                                                 # 转换为m * n的array
    trainData = trainData.tolist()                                                  # 转换为list

    trainLabel = pd.read_csv('trainLabel.csv', encoding = 'utf-8', header = None)   # 读取trainLabel, 并返回DataFrame
    trainLabel = np.array(trainLabel)  
    trainLabel = trainLabel.ravel()                                                 # 从m * 1转换为1 * m
    trainLabel = trainLabel.tolist()                                                # 转换为list

    myTree = createTree_c(trainData, trainLabel)
    
    verifyData = pd.read_csv('verifyData.csv', encoding = 'utf-8', header = None)   # 读取verifyData, 并返回DataFrame
    verifyData = np.array(verifyData)                                               # 转换为m * n的array
    verifyData = verifyData.tolist()                                                # 转换为list

    verifyLabel = pd.read_csv('verifyLabel.csv', encoding = 'utf-8', header = None) # verifyLabel, 并返回DataFrame
    verifyLabel = np.array(verifyLabel)  
    verifyLabel = verifyLabel.ravel()                                               # 从m * 1转换为1 * m
    verifyLabel = verifyLabel.tolist()                                              # 转换为list


    myTree = postPruningTree(myTree, trainData, trainLabel, verifyData, verifyLabel)    # 后剪枝

    testData = pd.read_csv('testData.csv', encoding = 'utf-8', header = None)       # 读取testData, 并返回DataFrame
    testData = np.array(testData)                                                   # 转换为m * n的array
    testData = testData.tolist()                                                    # 转换为list

    predict = []                                                                    # 预测结果
    for i in testData:
        predict.append(classify_c(myTree, i))

    sub = pd.DataFrame(predict)    
    sub.columns = ['Predicted']
    sub.insert(0, 'id', [i for i in range(1, len(testData) + 1)])                   # 插入id列

    sub.to_csv('./submit.csv', index = 0, encoding = "utf-8")                       # index=0表示不保留行索引
    