
# coding: utf-8

# In[9]:


from math import log
import operator
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)     #特征总数
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]     #取最后一个特征（是否是鱼）
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0      #特征集中没有此特征，则新建一个
        labelCounts[currentLabel] +=1          #特征相同，数目加一
    shannonEnt =0.0
    for key in labelCounts:  
        prob = float(labelCounts[key])/numEntries
        shannonEnt -=prob*log(prob,2)              #香农熵的计算公式
    return shannonEnt
def createDataSet():
    dataSet =[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels
myDat,labels = createDataSet()
myDat
calcShannonEnt(myDat)   


# In[27]:


#按照给定特征划分数据集
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] ==value:
            reduceFeatVec =featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet
#选择最好的特征数据集进行分类
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet ,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain>bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
#采用多数表决的方法决定该叶子节点的分类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] +=1
    sortedClassCount = sorted(classCount,iteritems(),key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]
#递归创建决策树
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]   #获取每一个样本的类别
    if classList.count(classList[0]) == len(classList):  #如果数据集全部为同一类别，则无需继续划分
        return classList[0]
    if len(dataSet[0])==1:    #分完特征，就返回根据多数表决的原则选出出现次数最多的类别
        return majority(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree


# In[28]:


#利用决策树进行分类
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key])._name_=='dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


# In[29]:


#存储决策树
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


# In[30]:


fr =open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age','prescript','astigmatic','tearRate']
lensesTree=createTree(lenses,lensesLabels)
lensesTree

