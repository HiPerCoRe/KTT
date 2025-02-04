import sys
import numpy as np
import pyktt as ktt

from base import *
from mlKTTPredictor import *

BATCH = 5

class PyProfilingSearcher(ktt.Searcher):
    ccMajor = 0
    ccMinor = 0
    cc = 0
    multiprocessors = 0
    profilingCountersModel = 0
    bestDuration = -1
    bestConf = None
    preselectedBatch = []
    tuningParamsNames = []
    currentConfiguration = ktt.KernelConfiguration()
    tuner = None
    model = None

    def __init__(self):
        ktt.Searcher.__init__(self)

    def OnInitialize(self):
        for i in range(0, BATCH) :
            self.preselectedBatch.append(self.GetRandomConfiguration())
        self.currentConfiguration = self.preselectedBatch[0]

        tp = self.currentConfiguration.GetPairs()
        for p in tp :
            self.tuningParamsNames.append(p.GetName())

    def Configure(self, tuner, modelFile):
        self.tuner = tuner
        self.ccMajor = tuner.GetCurrentDeviceInfo().GetCUDAComputeCapabilityMajor()
        self.ccMinor = tuner.GetCurrentDeviceInfo().GetCUDAComputeCapabilityMinor()
        self.cc = self.ccMajor + round(0.1 * self.ccMinor, 1)
        self.multiprocessors = tuner.GetCurrentDeviceInfo().GetMaxComputeUnits()

        self.profilingCountersModel = readPCList(modelFile + ".pc")
        self.model = loadMLModel(modelFile)

    def CalculateNextConfiguration(self, previousResult):
        # select the new configuration
        if len(self.preselectedBatch) > 0:
            # we are testing current batch
            if (self.bestConf == None) or (previousResult.GetKernelDuration() < self.bestDuration) :
                self.bestDuration = previousResult.GetKernelDuration()
                self.bestConf = self.currentConfiguration
            self.currentConfiguration = self.preselectedBatch.pop(0)
        else :
            if self.bestDuration != -1 :
                # we run the fastest one once again, but with profiling
                self.currentConfiguration = self.bestConf
                self.bestDuration = -1
                self.tuner.SetProfiling(True)
            else :
                # get PCs from the last tuning run
                if len(previousResult.GetResults()) > 1:
                    print("Warning: this version of profile-based searcher does not support searching kernels collections. Using counters from kernels 0 only.")
                globalSize = previousResult.GetResults()[0].GetGlobalSize()
                localSize = previousResult.GetResults()[0].GetLocalSize()
                profilingCountersRun = previousResult.GetResults()[0].GetProfilingData().GetCounters() #FIXME this supposes there is no composition profiled
                pcNames = ["Global size", "Local size"]
                pcVals = [globalSize.GetTotalSize(), localSize.GetTotalSize()]
                for pd in profilingCountersRun :
                    pcNames.append(pd.GetName())
                    if (pd.GetType() == ktt.ProfilingCounterType.Int) :
                        pcVals.append(pd.GetValueInt())
                    elif (pd.GetType() == ktt.ProfilingCounterType.UnsignedInt) or (pd.GetType() == ktt.ProfilingCounterType.Throughput) or (pd.GetType() == ktt.ProfilingCounterType.UtilizationLevel):
                        pcVals.append(pd.GetValueUint())
                    elif (pd.GetType() == ktt.ProfilingCounterType.Double) or (pd.GetType() == ktt.ProfilingCounterType.Percent) :
                        pcVals.append(pd.GetValueDouble())
                    else :
                        print("Fatal error, unsupported PC value passed to profile-based searcher!")
                        exit(1)

                # select candidate configurations according to position of the best one plus some random sample
                candidates = self.GetNeighbourConfigurations(self.bestConf, 2, 100)
                for i in range (0, 10) :
                    candidates.append(self.GetRandomConfiguration())
                print("Profile-based searcher: evaluating model for " + str(len(candidates)) + " candidates...")

                # get tuning space from candidates
                candidatesTuningSpace = []
                for c in candidates :
                    tp = c.GetPairs()
                    candidateParams = []
                    for p in tp :
                        candidateParams.append(p.GetValue())
                    candidatesTuningSpace.append(candidateParams)
                myTuningSpace = []
                tp = self.bestConf.GetPairs()
                for p in tp :
                    myTuningSpace.append(p.GetValue())

                # score the configurations
                scoreDistrib = [1.0]*len(candidates)
                bottlenecks = analyzeBottlenecks(pcNames, pcVals, 6.1, self.multiprocessors, self.convertSM2Cores() * self.multiprocessors)
                changes = computeChanges(bottlenecks, self.profilingCountersModel, self.cc)
                scoreDistrib = scoreTuningConfigurationsPredictor(changes, self.tuningParamsNames, myTuningSpace, candidatesTuningSpace, scoreDistrib, self.model)

                # select next batch
                for i in range(0, BATCH) :
                    idx = weightedRandomSearchStep(scoreDistrib, len(candidates))
                    self.preselectedBatch.append(candidates[idx])
                self.currentConfiguration = self.preselectedBatch[0]
                self.bestConf = None
                self.tuner.SetProfiling(False)

        return True

    def GetCurrentConfiguration(self):
        return self.currentConfiguration

    def convertSM2Cores(self):
        smToCoresDict = {
            0x30: 192,
            0x32: 192,
            0x35: 192,
            0x37: 192,
            0x50: 128,
            0x52: 128,
            0x53: 128,
            0x60: 64,
            0x61: 128,
            0x62: 128,
            0x70: 64,
            0x72: 64,
            0x75: 64,
            0x80: 64,
            0x86: 64
        }
        defaultSM = 64

        compact = (self.ccMajor << 4) + self.ccMinor
        if compact in smToCoresDict:
            return smToCoresDict[compact]
        else:
            print("Warning: unknown number of cores for SM " + str(self.ccMajor) + "." + str(self.ccMinor) + ", using default value of " + str(defaultSM))
            return defaultSM

def main():
    deviceIndex = 0
    kernelFile = "./CudaKernel.cu"

    argc = len(sys.argv)

    if argc >= 2:
        deviceIndex = sys.argv[1]

        if argc >= 3:
            kernelFile = sys.argv[2]

    numberOfAtoms = 256
    gridSize = 256
    gridSpacing = 0.5
    gridDimensions = ktt.DimensionVector(gridSize, gridSize, gridSize)
    blockDimensions = ktt.DimensionVector(1, 1)

    aX = 100.0 * np.random.rand(numberOfAtoms).astype('f')
    aY = 100.0 * np.random.rand(numberOfAtoms).astype('f')
    aZ = 100.0 * np.random.rand(numberOfAtoms).astype('f')
    aW = 100.0 * np.random.rand(numberOfAtoms).astype('f')
    aAll = np.zeros(numberOfAtoms*4, dtype = np.single)
    for i in range(numberOfAtoms):
        aAll[4 * i] = aX[i]
        aAll[4 * i + 1] = aY[i]
        aAll[4 * i + 2] = aZ[i]
        aAll[4 * i + 3] = aW[i]
    energyGrid = np.zeros(gridSize*gridSize*gridSize, dtype = np.single)

    tuner = ktt.Tuner(0, deviceIndex, ktt.ComputeApi.CUDA)
    tuner.SetCompilerOptions("-use_fast_math")
    tuner.SetTimeUnit(ktt.TimeUnit.Microseconds)
    tuner.SetProfiling(False)

    definition = tuner.AddKernelDefinitionFromFile("directCoulombSum", kernelFile, gridDimensions, blockDimensions)

    aXId = tuner.AddArgumentVectorFloat(aX, ktt.ArgumentAccessType.ReadOnly)
    aYId = tuner.AddArgumentVectorFloat(aY, ktt.ArgumentAccessType.ReadOnly)
    aZId = tuner.AddArgumentVectorFloat(aZ, ktt.ArgumentAccessType.ReadOnly)
    aWId = tuner.AddArgumentVectorFloat(aW, ktt.ArgumentAccessType.ReadOnly)
    aAllId = tuner.AddArgumentVectorFloat(aAll, ktt.ArgumentAccessType.ReadOnly)
    numberOfAtomsId = tuner.AddArgumentScalarInt(numberOfAtoms)
    gridSpacingId = tuner.AddArgumentScalarFloat(gridSpacing)
    gridSizeId = tuner.AddArgumentScalarInt(gridSize)
    energyGridId = tuner.AddArgumentVectorFloat(energyGrid, ktt.ArgumentAccessType.WriteOnly)
    tuner.SetArguments(definition, [aAllId, aXId, aYId, aZId, aWId, numberOfAtomsId, gridSpacingId, gridSizeId, energyGridId])

    kernel = tuner.CreateSimpleKernel("directCoulombSum", definition)

    tuner.AddParameter(kernel, "WORK_GROUP_SIZE_X", [16, 32])
    tuner.AddThreadModifier(kernel, [definition], ktt.ModifierType.Local, ktt.ModifierDimension.X, "WORK_GROUP_SIZE_X", ktt.ModifierAction.Multiply)
    tuner.AddThreadModifier(kernel, [definition], ktt.ModifierType.Global, ktt.ModifierDimension.X, "WORK_GROUP_SIZE_X", ktt.ModifierAction.Divide)
    tuner.AddParameter(kernel, "WORK_GROUP_SIZE_Y", [1, 2, 4, 8])
    tuner.AddThreadModifier(kernel, [definition], ktt.ModifierType.Local, ktt.ModifierDimension.Y, "WORK_GROUP_SIZE_Y", ktt.ModifierAction.Multiply)
    tuner.AddThreadModifier(kernel, [definition], ktt.ModifierType.Global, ktt.ModifierDimension.Y, "WORK_GROUP_SIZE_Y", ktt.ModifierAction.Divide)
    tuner.AddParameter(kernel, "WORK_GROUP_SIZE_Z", [1])
    tuner.AddParameter(kernel, "Z_ITERATIONS", [1, 2, 4, 8, 16, 32])
    tuner.AddThreadModifier(kernel, [definition], ktt.ModifierType.Global, ktt.ModifierDimension.Z, "Z_ITERATIONS", ktt.ModifierAction.Divide)
    tuner.AddParameter(kernel, "INNER_UNROLL_FACTOR", [0, 1, 2, 4, 8, 16, 32])
    tuner.AddParameter(kernel, "USE_SOA", [0, 1])
    tuner.AddParameter(kernel, "VECTOR_SIZE", [1])
    unrollLimit = lambda vector: vector[0] < vector[1]
    tuner.AddConstraint(kernel, ["INNER_UNROLL_FACTOR", "Z_ITERATIONS"], unrollLimit)
    parallelBound = lambda vector: vector[0] * vector[1] >= 64
    tuner.AddConstraint(kernel, ["WORK_GROUP_SIZE_X", "WORK_GROUP_SIZE_Y"], parallelBound)

    # Make tuner user the searcher implemented in Python.
    searcher = PyProfilingSearcher()
    tuner.SetSearcher(kernel, searcher)
    searcher.Configure(tuner, "1070-coulomb_output_DT.sav")

    # Begin tuning utilizing the stop condition implemented in Python.
    results = tuner.Tune(kernel)
    tuner.SaveResults(results, "TuningOutput", ktt.OutputFormat.JSON)

def executeSearcher(tuner, kernel, model):
    searcher = PyProfilingSearcher()
    tuner.SetSearcher(kernel, searcher)
    searcher.Configure(tuner, model)

