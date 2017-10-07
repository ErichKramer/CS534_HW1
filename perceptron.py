import sys
import numpy as np

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
        eprint("%s not found in dir. Please contact students.")
    features.append("bias")
    fdict = { x: i for i,x in enumerate(features)}
    return fdict


#   Input: Filename for formatted income + features, feature mapping to list index
#   Output:Tuple: ( 2d array :row = data point as binary features , list of y values empty list if none)

def fileToFeature(filename, featMap):
    filename = datPath + filename   
    datRows = [] #use normal list for better `.append()` behavior
    gTruth = []
    with open(filename, 'r') as dat:
        for line in dat:
            parsed = line.strip(' \n,').split(', ')
            parsed[0], parsed[7] = 'a'+parsed[0], 'h'+parsed[7] #key hack to differ hours from age
            tmp_row = np.zeros( len(featMap), dtype=np.bool_) #entry arr
            for feature in parsed[:9]:
                tmp_row[ featMap[feature] ] = 1     #entry features
            tmp_row[ featMap['bias'] ] = 1          #bias
            datRows.append(tmp_row)

            #map hack, append to truth array
            if parsed[9:10]: gTruth.append( {">50K":1 , "<=50K":-1}.get(parsed[9], None) )

    return np.array(datRows), gTruth


#func for main execution
def main():
    featMap = featDict()    
    print("Quantity of features: %d" % len(featMap))
    input("Press enter to see the dump of unique features. Their value corresponds to their index in the numpy array.")
    print(featMap)

    #usage of feature builder
    dev, devTruth       = fileToFeature("income.dev.txt", featMap)
    test, testTruth     = fileToFeature("income.test.txt", featMap) #testTruth is an empty list
    train, trainTruth   = fileToFeature("income.train.txt", featMap)



#pythonic main execution 
if __name__ == "__main__":
    main()




