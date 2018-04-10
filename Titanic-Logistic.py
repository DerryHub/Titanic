from pandas import *
from random import *
from numpy import *


def loadTrainData(fileName):
    trainData = read_csv(fileName)
    surList = list(map(classifyEmpty, trainData['Survived']))
    dataSet = []
    m, n = trainData.shape
    pclassList = list(map(classifyEmpty, trainData['Pclass']))
    sexList = list(map(classifySex, trainData['Sex']))
    ageList = list(map(classifyEmpty, trainData['Age']))
    sibspList = list(map(classifyEmpty, trainData['SibSp']))
    parchList = list(map(classifyEmpty, trainData['Parch']))
    fareList = list(map(classifyEmpty, trainData['Fare']))
    embarkedList = list(map(classifyEmbarked, trainData['Embarked']))
    for i in range(m):
        currentList = [1]
        currentList.append(pclassList[i])
        currentList.append(sexList[i])
        currentList.append(ageList[i])
        currentList.append(sibspList[i])
        currentList.append(parchList[i])
        currentList.append(fareList[i])
        currentList.append(embarkedList[i])
        dataSet.append(currentList)
    return dataSet, surList


def sigmoid(x):
    if x > 0:
        return 1.0
    if x < 0:
        return 0.0
    else:
        return 0.5


def getWeight(dataSet, surList, num=100):
    m, n = shape(dataSet)
    weights = ones(n)
    dataSet = array(dataSet)
    for j in range(num):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = m / (10 * (1.0 + i + j)) + 0.01
            randIndex = int(uniform(0, len(dataIndex)))
            h = sigmoid(sum(weights * dataSet[randIndex]))
            error = surList[randIndex] - h
            weights = weights + error * alpha * dataSet[randIndex]
            del dataIndex[randIndex]
    return weights


def classifySex(sex):
    if sex == 'male':
        return 1.0
    elif sex == 'female':
        return 2.0
    else:
        return 0.0


def classifyEmpty(str):
    num = float(str)
    if isnan(num) or str == '':
        return 0.0
    else:
        return num


def classifyEmbarked(em):
    if em == 'S':
        return 1.0
    elif em == 'C':
        return 2.0
    elif em == 'Q':
        return 3.0
    else:
        return 0.0


def calRate(dataSet, surList, weights):
    count = 0.0
    m, n = shape(dataSet)
    for i in range(m):
        cu = sigmoid(sum(dataSet[i] * weights))
        if cu == surList[i]:
            count = count + 1
    rate = count / m * 100
    print('正确率为%.2f' % rate + '%')


fileName = 'train.csv'
dataSet, surList = loadTrainData(fileName)
weights = getWeight(dataSet, surList)
calRate(dataSet, surList, weights)
