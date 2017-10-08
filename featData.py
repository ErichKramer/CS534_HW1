import numpy as np
import sys
import random


class featData:
    
    def __init__(self, filename, featMap):
        datRows = []
        gTruth = []
        with open(filename, 'r') as dat:
            for line in dat:
                parsed = line.strip(' \n,').split(', ')#strip removes an trailling in the string
                parsed[0], parsed[7] = 'a'+parsed[0], 'h'+parsed[7] #key hack to differ hours from age
                tmp_row = np.zeros( len(featMap), dtype=np.bool_) #entry arr
                for feature in parsed[:9]:
                    tmp_row[ featMap[feature] ] = 1     #entry features
                tmp_row[ featMap['bias'] ] = 1          #bias
                datRows.append(tmp_row)

                #map hack, append to truth array
                if parsed[9:10]: gTruth.append( {">50K":1 , "<=50K":-1}.get(parsed[9], None) )

        self.dat, self.truth =  np.array(datRows), np.array(gTruth)

        return None

    #shuffle data and truth, use same rng state for same shuffle
    def shuffle(self):
        rng = numpy.random.get_state()
        np.random.shuffle( self.dat )
        np.random.set_state()
        np.random.shuffle( self.truth )
        return None




