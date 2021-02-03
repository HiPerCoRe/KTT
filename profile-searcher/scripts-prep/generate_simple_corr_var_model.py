#!/usr/bin/env python

"""Extract average correlation and variance between tuning parameters and profiler counters

Usage:
  getstatistics.py -s <source> -t <tuningpar_interval> -c <counters_interval> -o <prefix> [-m]

Options:
  -h Show this screen.
  -s Source CSV file.
  -t Interval <from,to> of tuning parameters indices.
  -c Interval <from,to> of profiler counter indices.
  -o Output file prefix.
  -m  Generate also non-linear models.

"""

from docopt import docopt
import math
import subprocess
import numpy as np

if __name__ == '__main__':

    # parse command line
    arguments = docopt(__doc__)
    src = open(arguments['-s'])
    tuningRange = arguments['-t']
    countersRange = arguments['-c']
    dstCorr = open(arguments['-o'] + "-corr.csv", 'w')
    dstVar = open(arguments['-o'] + "-var.csv", 'w')

    try:
        rangeT = eval("np.r_[" + tuningRange + "]")
        rangeC = eval("np.r_[" + countersRange + "]")
    except:
        print("Intervals must be in format <i:j,[k,l,m]> or i:j (for example- 1:5,[7,9]  means cols 1 to 4 and 7 and 9).")
        exit()

    # parse CSV head
    words = src.readline().split(',')
    tuningParams = []
    print("Tuning parameters: "),
    for i in rangeT:
        tuningParams.append(words[i])
        print(words[i]),

    profilingCounters = []
    print("")
    print("Profiling counters: "),
    for i in rangeC:
        profilingCounters.append(words[i])
        print(words[i]),

    #parse CVS data
    tuningData = []
    countersData = []

    for line in src.readlines():
        words = line.split(',')
        if len(words) <= 1: break

        tunRow = []
        for i in rangeT:
            tunRow.append(float(words[i]))
        tuningData.append(list(tunRow))
        profRow = []
        for i in rangeC:
            profRow.append(float(words[i]))
        countersData.append(list(profRow))

    # create headers for output files
    dstCorr.write("Tuning parameter"),
    dstVar.write("Tuning parameter"),
    for p in profilingCounters:
        dstCorr.write("," + p),
        dstVar.write("," + p),
    dstCorr.write("\n")
    dstVar.write("\n")

    # compute statistics for each tuning parameter
    for i in range(0, len(tuningParams)):
        #print("Tuning parameter ", i)
        avgCorr = [0] * len(countersData[0])
        avgVariance = [0] * len(countersData[0])
        avgSamples = 0;
        used = [0] * len(tuningData)
        # seek for groups where only affected tuning parameter (i) changes
        while 1:
            startFrom = -1
            for j in range(0, len(tuningData)):
                if used[j] == 0:
                    startFrom = j
                    break
            if startFrom == -1:
                break
            used[startFrom] = 1;

            #if size of the group > 1, construct list of counters indices
            myGroup = [startFrom]
            for j in range(startFrom+1, len(tuningData)):
                identical = 1
                for k in range(0, len(tuningData[j])):
                    if (k != i) and (tuningData[startFrom][k] != tuningData[j][k]):
                        identical = 0
                        break
                if (identical == 1):
                    used[j] = 1
                    myGroup.append(j)

            #print(used)
            #print(myGroup)

            # compute statistics
            if len(myGroup) > 1:
                for j in range(0, len(countersData[0])):
                    mean = 0.0
                    for k in myGroup:
                        mean = mean + countersData[k][j]
                    mean = mean / len(myGroup)
                    variance = 0.0
                    for k in myGroup:
                        variance = variance + (countersData[k][j] - mean) ** 2
                    variance = variance / (len(myGroup) - 1)
                    meanTP = 0.0
                    for k in myGroup:
                        meanTP = meanTP + tuningData[k][i]
                    meanTP = meanTP / len(myGroup)
                    corr = 0.0
                    for k in myGroup:
                        corr = corr + (tuningData[k][i] - meanTP) * (countersData[k][j] - mean)
                    corrDenomA = 0.0
                    corrDenomB = 0.0
                    for k in myGroup:
                        corrDenomA = corrDenomA + (tuningData[k][i] - meanTP) ** 2
                        corrDenomB = corrDenomB + (countersData[k][j] - mean) ** 2
                    if (corr != 0.0):
                        corr = corr / math.sqrt(corrDenomA * corrDenomB)

                    #print(j, variance, corr)

                    avgVariance[j] = avgVariance[j] + variance;
                    avgCorr[j] = avgCorr[j] + corr;
                    avgSamples = avgSamples + 1;

        avgSamples = avgSamples / len(countersData[0])
        for j in range(0, len(countersData[0])):
            if avgSamples > 0:
                avgVariance[j] = avgVariance[j] / avgSamples
                avgCorr[j] = avgCorr[j] / avgSamples

            #print(avgVariance[j], avgCorr[j])

        #create output line
        dstCorr.write(tuningParams[i]),
        dstVar.write(tuningParams[i]),
        for j in range(0, len(countersData[0])):
            dstCorr.write("," + str(avgCorr[j])),
            dstVar.write("," + str(avgVariance[j])),
        dstCorr.write("\n")
        dstVar.write("\n")

    if arguments['-m']:
        command = "Rscript ./create_nonlinear_models.R "  + arguments['-s'] + " " + arguments['-o'] + " " + arguments['-t'] + " " + arguments['-c']
        subprocess.call(command.split())
