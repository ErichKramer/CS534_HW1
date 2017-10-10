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

def genWeightVec(size, rand=True):
    if rand:
        return np.array( [random.uniform(-1, 1) for _ in range( size )] )
    else:
        return np.zeros(size)


#generate basic perceptron
def genPerceptron(trainSet, epochs=5):
    dev = featData(datPath+"income.dev.txt")
    weight = genWeightVec(trainSet.numFeat)
    count=1
    for _ in range(epochs):
        for featVec,truth in zip( trainSet.dat, trainSet.truth):
            dotProduct = np.dot(weight, featVec )
            if dotProduct*truth <=0:
                weight += featVec*truth#bias = 1, truth*1 = truth increment on bias
            count +=1
            if count %5000==0:
                print(" Epoch number: {}".format( round(count/len(trainSet.dat),2) ))
                testWeightVector( trainSet, weight)
                testWeightVector( dev,      weight)
    return weight

#gen average perceptron
def genAvgPerceptron(trainSet, naive=False, epochs=5):
    dev = featData(datPath+"income.dev.txt")
    weight = genWeightVec(trainSet.numFeat)
    wPrime = genWeightVec(trainSet.numFeat, rand=False)
    count=1
    for _ in range(epochs):
        for featVec,truth in zip( trainSet.dat, trainSet.truth):
            dotProduct = np.dot(weight, featVec )
            if dotProduct*truth <=0:
                weight += featVec*truth#bias = 1, truth*1 = truth increment on bias
                
                if naive:   wPrime += weight
                else:       wPrime += count*featVec*truth
            count +=1
            if count %5000==0:

                if naive:   tmpWeight = wPrime/count
                else:       tmpWeight = weight - (wPrime/count)
                print(" Epoch number: {}".format( round(count/len(trainSet.dat),2) ))
                testWeightVector( trainSet, tmpWeight)
                testWeightVector( dev,      tmpWeight)
    if naive:
        return wPrime/count
    else:
        return weight - (wPrime/count)

#gen MIRA perceptron, default is not aggressive, set p value for margin
def genMIRA(trainSet, p=0, averaged=True, epochs=5):
    
    dev = featData(datPath+"income.dev.txt")
    weight = genWeightVec(len(trainSet.dat[0]))
    wPrime = genWeightVec(trainSet.numFeat, False)
    count=0
    for _ in range(epochs):
        for featVec, truth in zip(trainSet.dat, trainSet.truth):
            dotProd = np.dot(weight, featVec)
            if dotProd*truth <= p:
                update = ((truth - dotProd)/np.dot(featVec, featVec))*featVec   
                weight += update   
                if averaged: wPrime += update*featVec*count

            count +=1
            if count %5000==0:
                if averaged: tmpWeight = weight - (wPrime/count)
                else: tmpWeight = weight
                print(" Epoch number: {}".format( round(count/len(trainSet.dat),2) ))
                testWeightVector( trainSet, tmpWeight)
                testWeightVector( dev,      tmpWeight)

    return weight - (wPrime/count)


#change to return string instead of print side effect?
def testWeightVector(testSet, weight):
    wrong = 0
    right = 0
    for featVec, truth in zip(testSet.dat, testSet.truth):
        dotProduct = np.dot(weight, featVec)
        if dotProduct*truth <=0:
            wrong+=1
        else:
            right+=1
    print("Percentage wrong in {}= ".format(testSet.filen), round( wrong/(wrong+right), 5) )
    return wrong/(wrong+right)

#func for main execution
def main():

    #usage of feature builder
    train   = featData(datPath + "income.train.txt") #full path, feature map
    test    = featData(datPath + "income.dev.txt")

    #weight = genPerceptron(train)
    weight = genAvgPerceptron(train, naive=True)
    #weight = genMIRA(train, p=.1)
    #weight = genMIRA(train, p=.5)
    #weight = genMIRA(train, p=.9)

    #testWeightVector(train, weight)

#pythonic main execution 
if __name__ == "__main__":
    main()




