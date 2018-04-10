from pandas import *
from numpy import *
from random import *
from csv import *


def loadTrainData(fileName):
    trainData = read_csv(fileName)
    dataSet = []
    m, n = trainData.shape
    pclassList = list(map(classifyEmpty, trainData['Pclass']))
    sexList = list(map(classifySex, trainData['Sex']))
    ageList = list(map(classifyEmpty, trainData['Age']))
    sibspList = list(map(classifyEmpty, trainData['SibSp']))
    parchList = list(map(classifyEmpty, trainData['Parch']))
    fareList = list(map(classifyEmpty, trainData['Fare']))
    embarkedList = list(map(classifyEmbarked, trainData['Embarked']))
    surList = list(map(classifyEmpty, trainData['Survived']))
    for i in range(m):
        currentList = []
        currentList.append(pclassList[i])
        currentList.append(sexList[i])
        currentList.append(ageList[i])
        currentList.append(sibspList[i])
        currentList.append(parchList[i])
        currentList.append(fareList[i])
        currentList.append(embarkedList[i])
        currentList.append(surList[i])
        dataSet.append(currentList)
    return mat(dataSet)


def splitData(dataSet):
    dataSet = dataSet.tolist()
    trainData = []
    testData = []
    m = shape(dataSet)[0]
    indexList = list(range(m))
    while (len(trainData) <= 0.7 * m):
        i = choice(indexList)
        trainData.append(dataSet[i])
        indexList.remove(i)
    for i in indexList:
        testData.append(dataSet[i][:])
    return mat(trainData), mat(testData)


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


def majorLabel(surList):
    count0 = 0.0
    count1 = 0.0
    for value in surList:
        if value == 0.0:
            count0 += 1.0
        elif value == 1.0:
            count1 += 1.0
    if count0 > count1:
        return 0.0
    else:
        return 1.0


def binSplitDataSet(dataSet, feature, value):
    left = dataSet[nonzero(dataSet[:, feature] < value)[0], :]
    right = dataSet[nonzero(dataSet[:, feature] >= value)[0], :]
    return left, right


def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType=majorLabel, errType=regErr, ops=(1, 4)):
    tolS = ops[0]
    tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet[:, -1].T.tolist()[0])
    m, n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0.0
    for featIndex in range(n - 1):
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
            left, right = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(left)[0] < tolN) or (shape(right)[0] < tolN): continue
            newS = errType(left) + errType(right)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:
        return None, leafType(dataSet[:, -1].T.tolist()[0])
    left, right = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(left)[0] < tolN) or (shape(right)[0] < tolN):
        return None, leafType(dataSet[:, -1].T.tolist()[0])
    return bestIndex, bestValue


def createTree(dataSet, leafType=majorLabel, errType=regErr, ops=(1, 4)):
    feature, value = chooseBestSplit(dataSet, leafType, errType, ops)
    if feature == None: return value
    retTree = {}
    retTree['spIndex'] = feature
    retTree['spValue'] = value
    num0, num1 = calNum(dataSet)
    retTree['num0'] = num0
    retTree['num1'] = num1
    leftDataSet, rightDataSet = binSplitDataSet(dataSet, feature, value)
    retTree['left'] = createTree(leftDataSet, leafType, errType, ops)
    retTree['right'] = createTree(rightDataSet, leafType, errType, ops)
    return retTree


def calNum(dataSet):
    num0 = 0
    num1 = 0
    m = shape(dataSet)[0]
    for i in range(m):
        if dataSet[i, -1] == 0:
            num0 += 1
        else:
            num1 += 1
    return num0, num1


'''
def calRate(testDataSet, surList, tree):
    m = shape(testDataSet)[0]
    rightCount = 0
    testDataSet = testDataSet.tolist()
    for i in range(m):
        if isSurvivedOrNot(testDataSet[i], tree) == surList[i]:
            rightCount += 1
    rate = float(rightCount) / m * 100
    print('正确率为%.2f' % rate + '% !')
'''


def printSurList(testDataSet, tree):
    m = shape(testDataSet)[0]
    testDataSet = testDataSet.tolist()
    surList = []
    for i in range(m):
        sur = isSurvivedOrNot(testDataSet[i], tree)
        surList.append(int(sur))
    print(surList)
    return surList


def isSurvivedOrNot(each, tree):
    if type(tree) == float:
        return tree
    index = tree['spIndex']
    value = tree['spValue']
    if each[index] < value:
        sur = isSurvivedOrNot(each, tree['left'])
    elif each[index] >= value:
        sur = isSurvivedOrNot(each, tree['right'])
    return sur


def loadTestData(fileName):
    trainData = read_csv(fileName)
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
        currentList = []
        currentList.append(pclassList[i])
        currentList.append(sexList[i])
        currentList.append(ageList[i])
        currentList.append(sibspList[i])
        currentList.append(parchList[i])
        currentList.append(fareList[i])
        currentList.append(embarkedList[i])
        dataSet.append(currentList)
    return mat(dataSet)


def cutTree(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree)
    if isTree(tree['left']) or isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spIndex'], tree['spValue'])
    if isTree(tree['left']): tree['left'] = cutTree(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] = cutTree(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spIndex'], tree['spValue'])
        pl = float(len(lSet)) / (len(lSet) + len(rSet))
        pr = 1 - pl
        errorNoMerge = pl * calRate(tree, 'left', lSet) + pr * calRate(tree, 'right', rSet)
        tm = treeMean(tree)
        errorMerge = float(sum(power(testData[:][-1] - tm, 2))) / len(testData)
        if errorMerge < errorNoMerge:
            return tm
        else:
            return tree
    return tree


def getMean(tree):
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    return treeMean(tree)


def isTree(tree):
    return type(tree).__name__ == 'dict'


def treeMean(tree):
    if tree['num0'] > tree['num1']:
        return tree['left']
    else:
        return tree['right']


def calRate(tree, index, testDate):
    m = len(testDate)
    n = sum(power(testData[:][-1] - tree[index], 2))
    rate = float(n) / (m + 1)
    return rate


def CSVtoDict(fileName):
    fp = read_csv(fileName)
    f = open(fileName, 'r')
    fc = reader(f)
    head = [s for s in fc][0]
    dict = {}
    for h in head:
        list = []
        for val in fp[h]:
            list.append(int(val))
        dict[h] = list
    return dict


fileName = 'train.csv'
dataSet = loadTrainData(fileName)
trainData, testData = splitData(dataSet)
tree = createTree(trainData, ops=(-1, 1))
print(tree)
tree = cutTree(tree, testData)
print(tree)

fileName = 'test.csv'
testDataSet = loadTestData(fileName)
surList = printSurList(testDataSet, tree)
fileName = 'gender_submission.csv'
dict = CSVtoDict(fileName)
dict['Survived'] = surList
print(dict)
csv = DataFrame(dict)
csv.to_csv(fileName, index=False)
