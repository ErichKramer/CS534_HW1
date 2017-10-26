import sys, time
import matplotlib.pyplot as plt
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
def genAvgPerceptron(trainSet, naive=False, graph=False, epochs=5):
    dev = featData(datPath+"income.dev.txt")
    weight = genWeightVec(trainSet.numFeat, False)
    wPrime = genWeightVec(trainSet.numFeat, rand=False)
    count=1
    minList = [(0,1)]*3
    errList = [1]*3
    if graph:
        trainSet.devErrAvg = []
        trainSet.devErrSimple = []
    
    for _ in range(epochs):
        for featVec,truth in zip( trainSet.dat, trainSet.truth):
            dotProduct = np.dot(weight, featVec )
            
            #if naive: wPrime +=weight
            if dotProduct*truth <=0:

                update = featVec*truth
                weight += update#bias = 1, truth*1 = truth increment on bias
                
                if not naive:       wPrime += count*update
                else:               wPrime +=weight
            count +=1
            
            if count%200==0 and graph:
                if round(count/len(trainSet.dat), 2) < 1.00:
                    if naive:tmpWeight = wPrime/count
                    else:tmpWeight = weight - (wPrime/count)
                    trainSet.devErrAvg.append(testWeightVector(dev, weight))
                    trainSet.devErrSimple.append(testWeightVector(dev, tmpWeight))
            if count %5000==0:
                if naive:   tmpWeight = wPrime/count
                else:       tmpWeight = weight - (wPrime/count)
                epoch = round(count/len(trainSet.dat), 2)
                errList[0] = testWeightVector( trainSet, weight)
                errList[1] = testWeightVector( dev, weight)
                errList[2] = testWeightVector( dev, tmpWeight)
                for i  in range(3):
                    if minList[i][1] > errList[i]:
                        minList[i] = (epoch, errList[i])

                print("Epoch number: {}, train: {} dev: {} avg dev: {}".format( \
                        epoch ,*errList  ))
            else:pass

    print("\nMinimums:\nEpoch number: {}, train: {} dev: {} avg train: {}".format( \
        round(count/len(trainSet.dat),2), *minList  ))
    
    if naive:   tmpW =  wPrime/count
    else:       tmpW =  weight - (wPrime/count)
    return tmpW



#gen MIRA perceptron, default is not aggressive, set p value for margin
def genMIRA(trainSet, p=0, averaged=True, epochs=5):
    
    dev = featData(datPath+"income.dev.txt")
    weight = genWeightVec(len(trainSet.dat[0]))
    wPrime = genWeightVec(trainSet.numFeat, False)
    tmpList = [0]*3
    count=0
    for _ in range(epochs):
        for featVec, truth in zip(trainSet.dat, trainSet.truth):
            dotProd = np.dot(weight, featVec)
            if dotProd*truth <= p:
                update = ((truth - dotProd)/np.dot(featVec, featVec))*featVec   
                weight += update   
                if averaged: wPrime += update*count

            count +=1
            if count %5000==0:
                if averaged: tmpWeight = weight - (wPrime/count)
                else: tmpWeight = weight
                trnErr = testWeightVector(trainSet, weight)
                devErr = testWeightVector(dev, weight)
                avgErr = testWeightVector(dev, tmpWeight)
                
                print("[MIRA] Epoch number: {}, train: {} dev: {} avgDev: {}".format( \
                        round(count/len(trainSet.dat),2), trnErr, devErr, avgErr ))
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
    #print("Percentage wrong in {}= ".format(testSet.filen), round( wrong/(wrong+right), 5) )
    return round( wrong/(wrong+right), 4)

#outputs a new file with predicted brackets
def predict(fobj, filename, weight):
    pos = 0
    total = 0
    with open(filename, 'r') as test, open("prediction.txt", 'w') as oFile:
        for line in test:
            line = line.strip('\n')
            featVec, _ = fobj.parseBinQuant(line)
            pred = np.dot(featVec, weight)
            
            if pred > 0:
                line += " >50K"
                print( line, file=oFile)
                pos+=1
            else:
                line += " <=50K"
                print( line, file=oFile)
                pass
            total+=1

    pass


#func for main execution
def main():

    #usage of feature builder
    train   = featData(datPath + "income.train.txt") #full path, feature map
    dev     = featData(datPath + "income.dev.txt")
    test    = featData(datPath + "income.test.txt")

    genMIRA(train, averaged=False)

    print("Number of features: {}".format(len(train.dat[0])) )
    #weight = genPerceptron(train)
    currTime = time.time()
    weight = genAvgPerceptron(train, graph=True, naive=True)
    timeTaken = time.time()-currTime
    print("Time taken for naive = {}".format(timeTaken))

    currTime = time.time()
    weight = genAvgPerceptron(train, naive=False)
    timeTaken = time.time()-currTime
    print("Time taken for smart = {}".format(timeTaken))
    

    predict(test, datPath + "income.test.txt", weight)

    input("Continue?")
    rg = [ i*200 for i,_ in enumerate(train.devErrSimple)]
    
    plt.figure(figsize=(20,10))
    plt.plot(rg , train.devErrSimple, 'r--', label="Basic Perceptron")
    plt.plot(rg, train.devErrAvg, 'b--', label="Average Perceptron")
    plt.ylabel("Error Rate")
    plt.xlabel("Number of data points (Epoch 1.00)")
    plt.title("First epoch simple vs. average perceptron")
    plt.savefig('epoch_perceptron.png')
    plt.show()
    
   

    #weight = genMIRA(train)
    print(*train.getMost(weight, 5), sep = '\n')
    print(*train.getMost(weight, len(weight)), sep = '\n')
    print(*train.getMost(weight, 0, "Male", "Female"), sep = '\n')

    i = 0
    for featVec, truth in zip(dev.dat, dev.truth):
        i +=1
        dotProduct = np.dot(weight, featVec)
        if dotProduct*truth <=0:
            print(i)
            

    input("Continue?")

#pythonic main execution 
if __name__ == "__main__":
    main()





