import sys
import numpy as np
import random
from featData import *
datPath = "./income-data/"


#function alias to print to std error
def eprint(*args, **kwargs):
    print(*args, file=sys.stdout, **kwargs)

#create a dictionary that maps features to indexes in an np.array 
#generate featuers for each individual age and hours, use rules file "features.txt" to find others
def featDict(uniqFeat="features.txt"):
    features = [ 'a'+ str(age) for age in range(17,91) ] + ['h'+str(hour) for hour in range(100)]  
    try:
        with open(uniqFeat, 'r') as featFile:
            for line in featFile:
                features.append(line.strip())
    except FileNotFoundError:
        eprint("%s not found in dir. Please contact students."% filename )
    features.append("bias")
    fdict = { x: i for i,x in enumerate(features)}
    return fdict


def genWeightVec(size, rand=True):
    if rand:
        return np.array( [random.uniform(-1, 1) for _ in range( size )] )
    else:
        return np.zeros(size)


#generate basic perceptron
def genPerceptron(trainSet, epochs=5):
    weight = genWeightVec(len(trainSet.dat[0]) +1)
    for _ in range(epochs):

        for featVec,truth in zip( trainSet.dat, trainSet.truth):
            dotProduct = sum( [ w * xi for w,xi in zip(weight, np.append( featVec, truth) ) ])
            
            if dotProduct*truth <=0:    #should this be <0?
                weight += np.append(featVec*truth, truth)
        testWeightVector(trainSet, weight)

    return weight

#gen average perceptron
def genAvgPerceptron(trainSet, epochs=5):
    pass

#gen MIRA perceptron, default is not aggressive
def genMIRAPerceptron(trainSet, aggressive=False):
    pass

def testWeightVector(testSet, weight):
    wrong = 0
    right = 0
    for featVec, truth in zip(testSet.dat, testSet.truth):
        dotProduct = sum( [w*xi for w,xi in zip(weight, np.append(featVec, truth) ) ] )
        if dotProduct*truth <=0:
            wrong+=1
        else:
            right+=1
    print("Percentage wrong = ", wrong/(wrong+right) )

    return None

#func for main execution
def main():
    featMap = featDict()    
    print("Quantity of features: %d" % len(featMap))

    #usage of feature builder
    train   = featData(datPath + "income.train.txt", featMap) #full path, feature map
    test    = featData(datPath + "income.dev.txt", featMap)

    weight = genPerceptron(train)
    testWeightVector(train,weight) #test on training set
    #testWeightVector(test, weight)
#pythonic main execution 
if __name__ == "__main__":
    main()




