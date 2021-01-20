'''
To predict the profiling counter from tuning space features. Here we use the .sav file as trained knowledge base, this file include the
result from generate-knowledge-base.py. The application of this function is similar to exact one that implemented in base.py but here we
predict the actualPC instead of search in within inputed .csv file.

'''

import pickle
import numpy as np
import pandas as pd
from base import *



########################### Temp function
def scoreTuningConfigurationsPredictor(changeImportance, tuningparamsNames, actualConf, tuningSpace, scoreDistrib, trainedKnowledgeBase):
    def mulfunc(a, b, c):
        if (a * (b - c)) > 0.0:
            return 1.0
        if (a * (b - c)) < 0.0:
            return -1.0
        else:
            return 0.0

    newScoreDistrib = [0.0] * len(tuningSpace)
    actualPC = []

    # Using ML predictor
    filename = trainedKnowledgeBase
    loaded_model = pickle.load(open(filename, 'rb'))
    predictedPC = loaded_model.predict([actualConf])
    actualPC = list(predictedPC.flatten())

    if len(actualPC) == 0 :
        for i in range(0, len(tuningSpace)) :
            uniformScoreDistrib = [1.0] * len(tuningSpace)
            if scoreDistrib[i] == 0.0 :
                uniformScoreDistrib[i] = 0.0
        return uniformScoreDistrib

    cmIdx = 0

    #################################################### Using ML predictor
    predictedMyPC = loaded_model.predict(tuningSpace)
    predictedMyPC1 = np.array(predictedMyPC)
    actualPC1 = np.array(actualPC)
    n = len(changeImportance) - len(actualPC1)
    changeImportance = changeImportance[:len(changeImportance)-n]
    changeImportance1 = np.array(changeImportance)

    vfunc = np.vectorize(mulfunc)
    mul = vfunc(changeImportance1, predictedMyPC1, actualPC1)
    res = np.array(mul * abs(changeImportance1 * 2.0 * (predictedMyPC1 - actualPC1) / (predictedMyPC1+actualPC1)))
    res = np.nan_to_num(res)
    newScoreDistrib = res.sum(axis=1)


    minScore = min(newScoreDistrib)
    maxScore = max(newScoreDistrib)
    if VERBOSE > 0 :
        print("scoreDistrib interval: ", minScore, maxScore)
    for i in range(0, len(tuningSpace)) :
        if newScoreDistrib[i] < CUTOFF :
            newScoreDistrib[i] = 0.0
        else :
            if newScoreDistrib[i] < 0.0 :
                newScoreDistrib[i] = 1.0 - (newScoreDistrib[i] / minScore)
            else :
                if newScoreDistrib[i] > 0.0 :
                    newScoreDistrib[i] = 1.0 + (newScoreDistrib[i] / maxScore)
            newScoreDistrib[i] = newScoreDistrib[i]**EXP
            if newScoreDistrib[i] == 0.0 :
                newScoreDistrib[i] = 0.0001

        # if was 0, set to 0 (explored)
        if scoreDistrib[i] == 0.0 :
            newScoreDistrib[i] = 0.0

    if VERBOSE > 2 :
        print("Predictro newScoreDistrib", newScoreDistrib)

    return newScoreDistrib

# auxiliary functions
def readPCList (filename) :
    ret = []
    pcListFile = open(filename, 'r')
    if pcListFile.readline().rstrip() != 'Profiling counter' :
        print('Malformed PC list file!')
        return ret
    for line in pcListFile.readlines():
        ret.append(line.rstrip())
    pcListFile.close()

    return ret
