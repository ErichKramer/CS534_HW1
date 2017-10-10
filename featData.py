import numpy as np
import sys
import random


class featData:
    
    def __init__(self, filename):
        self.filen = filename.split('.')[2]
        self.featDict()
        datRows = []
        gTruth = []
        with open(filename, 'r') as dat:
            for line in dat:
                parsed = line.strip(' \n,').split(', ')#strip removes an trailling in the string
                parsed[0], parsed[7] = 'a'+parsed[0], 'h'+parsed[7] #key hack to differ hours from age
                tmp_row = np.zeros( len(self.featMap), dtype=np.int) #entry arr
                for feature in parsed[:9]:
                    tmp_row[ self.featMap[feature] ] = 1     #entry features
                tmp_row[ self.featMap['bias'] ] = 1          #bias
                datRows.append(tmp_row)

                #map hack, append to truth array
                if parsed[9:10]: gTruth.append( {">50K":1 , "<=50K":-1}.get(parsed[9], None) )

        self.dat, self.truth =  np.array(datRows), np.array(gTruth)
        self.numFeat = len(self.dat[0])
        return None

    def featDict(self, uniqFeat="features.txt"):
        
        features = [ 'a'+ str(age) for age in range(17,91) ] + ['h'+str(hour) for hour in range(100)]
        try:
            with open(uniqFeat, 'r') as featFile:
                for line in featFile:
                    features.append(line.strip())
        except FileNotFoundError:
            print("{} not found in dir. Please contact students.".format(filename), file=sys.stderr )
        features.append("bias")
        self.featMap    = { x: i for i,x in enumerate(features)}
        self.featUnMap  = { i: x for i,x in enumerate(features)}
        return None

    def getMost(self, weight, n):
        tmpType = [ ("index", int), ("value", float)]
        sortW = np.sort(np.array( [(i,x) for i,x in enumerate(weight)], \
                dtype=tmpType), order="value")[::np.sign(n)*-1]
        
        return ["Key:"+ str(self.featUnMap[elem[0]]) + " Weight:"+ str(elem[1]) for elem in sortW[:abs(n)] ]
          

    #shuffle data and truth, use same rng state for same shuffle
    def shuffle(self):
        rng = np.random.get_state()
        np.random.shuffle( self.dat )
        np.random.set_state(rng)
        np.random.shuffle( self.truth )
        return None




