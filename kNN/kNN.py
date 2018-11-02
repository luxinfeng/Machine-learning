
# coding: utf-8

# In[1]:


import numpy as np
import operator
from os import listdir
def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     #对distances中的元素进行排序，并返回对应的索引值
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1   #输出字典classCount中voteIlabel对应的值，如果没有，初始化为0；
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
group,labels = createDataSet()
classify0([0,0],group,labels,3)


# In[2]:


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()   #获取文件的行数
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines,3))  #创建一个和源文件行数相等，列数为3的矩阵
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index +=1
    return returnMat,classLabelVector
datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')


# In[4]:


import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()   #初始化图像
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*np.array(datingLabels),15.0*np.array(datingLabels))  #创建图像
plt.show()


# In[5]:


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals,(m,1))    #原数据减去最小值
    normDataSet = normDataSet/np.tile(ranges,(m,1))   #减去之后的值除以原数据最大值与最小值之差
    return normDataSet,ranges,minVals
norm,ranges,minVals = autoNorm(datingDataMat)


# In[6]:


def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix("datingTestSet2.txt")
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)  #选择一部分来测试分类器
    errorCount = 0.0    #初始化错误率
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)   # K临近算法
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult,datingLabels[i]))
        if (classifierResult !=datingLabels[i]):
            errorCount += 1.0
    print ("the total error rate is: &f" %(errorCount/float(numTestVecs)))
    print(errorCount)


# In[7]:


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')    #将文本数据转换为矩阵数据
    normMat, ranges, minVals = autoNorm(datingDataMat)              #归一化
    inArr = np.array([ffMiles, percentTats, iceCream, ])             #将输入数据转换为矩阵
    classifierResult = classify0((inArr -  minVals)/ranges, normMat, datingLabels, 3)    
    print("You will probably like this person: %s" % resultList[classifierResult - 1])


# In[ ]:


classifyPerson()

