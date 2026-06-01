#!/usr/bin/env python3

import os
import sys
import glob
from pathlib import Path

from ProfileBasedSearcher import ProfileBasedSearcher
from modules.info import BatchInfo, ModelInfo

import lib.pyktt as ktt
import numpy

LOG_DIRECTORY = './logs/profiled/'


def getLogPath(modelInfo: ModelInfo, batchInfo: BatchInfo) -> str:
    runningGpu, profilingGpu = (
        Path(modelInfo.counterPath).name.split('_')[0].split('-')
    )

    logPrefix = (
        LOG_DIRECTORY
        + f'{runningGpu}-{profilingGpu}/'
        + f'b{batchInfo.batchSize}-n{batchInfo.neighborSize}-'
        + f'r{batchInfo.randomSize}/'
    )

    # logPrefix = LOG_DIRECTORY  # TEST
    existingLogs = glob.glob(logPrefix + '*.xml')
    return logPrefix + f'output-{len(existingLogs) + 1}'


def runTuning(deviceIndex: int, kernelFile: str):
    numberOfAtoms = 256
    gridSize = 256
    gridSpacing = 0.5
    gridDimensions = ktt.DimensionVector(gridSize, gridSize, gridSize)
    blockDimensions = ktt.DimensionVector(1, 1)

    aX = 100.0 * numpy.random.rand(numberOfAtoms).astype('f')
    aY = 100.0 * numpy.random.rand(numberOfAtoms).astype('f')
    aZ = 100.0 * numpy.random.rand(numberOfAtoms).astype('f')
    aW = 100.0 * numpy.random.rand(numberOfAtoms).astype('f')
    aAll = numpy.zeros(numberOfAtoms * 4, dtype=numpy.single)
    for i in range(numberOfAtoms):
        aAll[4 * i] = aX[i]
        aAll[4 * i + 1] = aY[i]
        aAll[4 * i + 2] = aZ[i]
        aAll[4 * i + 3] = aW[i]
    energyGrid = numpy.zeros(gridSize * gridSize * gridSize, dtype=numpy.single)

    tuner = ktt.Tuner(0, deviceIndex, ktt.ComputeApi.CUDA)
    tuner.SetCompilerOptions('-use_fast_math')
    tuner.SetTimeUnit(ktt.TimeUnit.Microseconds)
    tuner.SetProfiling(False)

    definition = tuner.AddKernelDefinitionFromFile(
        'directCoulombSum', kernelFile, gridDimensions, blockDimensions
    )

    aXId = tuner.AddArgumentVectorFloat(aX, ktt.ArgumentAccessType.ReadOnly)
    aYId = tuner.AddArgumentVectorFloat(aY, ktt.ArgumentAccessType.ReadOnly)
    aZId = tuner.AddArgumentVectorFloat(aZ, ktt.ArgumentAccessType.ReadOnly)
    aWId = tuner.AddArgumentVectorFloat(aW, ktt.ArgumentAccessType.ReadOnly)
    aAllId = tuner.AddArgumentVectorFloat(aAll, ktt.ArgumentAccessType.ReadOnly)
    numberOfAtomsId = tuner.AddArgumentScalarInt(numberOfAtoms)
    gridSpacingId = tuner.AddArgumentScalarFloat(gridSpacing)
    gridSizeId = tuner.AddArgumentScalarInt(gridSize)
    energyGridId = tuner.AddArgumentVectorFloat(
        energyGrid, ktt.ArgumentAccessType.WriteOnly
    )
    tuner.SetArguments(
        definition,
        [
            aAllId,
            aXId,
            aYId,
            aZId,
            aWId,
            numberOfAtomsId,
            gridSpacingId,
            gridSizeId,
            energyGridId,
        ],
    )

    kernel = tuner.CreateSimpleKernel('directCoulombSum', definition)

    tuner.AddParameter(kernel, 'WORK_GROUP_SIZE_X', [16, 32])
    tuner.AddThreadModifier(
        kernel,
        [definition],
        ktt.ModifierType.Local,
        ktt.ModifierDimension.X,
        'WORK_GROUP_SIZE_X',
        ktt.ModifierAction.Multiply,
    )
    tuner.AddThreadModifier(
        kernel,
        [definition],
        ktt.ModifierType.Global,
        ktt.ModifierDimension.X,
        'WORK_GROUP_SIZE_X',
        ktt.ModifierAction.Divide,
    )
    tuner.AddParameter(kernel, 'WORK_GROUP_SIZE_Y', [1, 2, 4, 8])
    tuner.AddThreadModifier(
        kernel,
        [definition],
        ktt.ModifierType.Local,
        ktt.ModifierDimension.Y,
        'WORK_GROUP_SIZE_Y',
        ktt.ModifierAction.Multiply,
    )
    tuner.AddThreadModifier(
        kernel,
        [definition],
        ktt.ModifierType.Global,
        ktt.ModifierDimension.Y,
        'WORK_GROUP_SIZE_Y',
        ktt.ModifierAction.Divide,
    )
    tuner.AddParameter(kernel, 'WORK_GROUP_SIZE_Z', [1])
    tuner.AddParameter(kernel, 'Z_ITERATIONS', [1, 2, 4, 8, 16, 32])
    tuner.AddThreadModifier(
        kernel,
        [definition],
        ktt.ModifierType.Global,
        ktt.ModifierDimension.Z,
        'Z_ITERATIONS',
        ktt.ModifierAction.Divide,
    )
    tuner.AddParameter(kernel, 'INNER_UNROLL_FACTOR', [0, 1, 2, 4, 8, 16, 32])
    tuner.AddParameter(kernel, 'USE_CONSTANT_MEMORY', [0])
    tuner.AddParameter(kernel, 'USE_SOA', [0, 1])
    tuner.AddParameter(kernel, 'VECTOR_SIZE', [1])
    tuner.AddConstraint(
        kernel,
        ['INNER_UNROLL_FACTOR', 'Z_ITERATIONS'],
        lambda vector: vector[0] < vector[1],
    )

    tuner.AddConstraint(
        kernel,
        ['WORK_GROUP_SIZE_X', 'WORK_GROUP_SIZE_Y'],
        lambda vector: vector[0] * vector[1] >= 64,
    )

    # Make tuner use the profiling searcher
    searcher = ProfileBasedSearcher()
    tuner.SetSearcher(kernel, searcher)

    modelInfo = ModelInfo(
        deltaPath='./models/1070_all_XGBRegressor.sav',
        spacePath='./models/1070_coulomb_XGBRegressor.sav',
        counterPath='./models/2080-1070_all_XGBRegressor.sav',
    )
    batchInfo = BatchInfo(batchSize=10, neighborSize=40, randomSize=100)
    searcher.Configure(tuner, modelInfo, batchInfo)

    # Begin tuning utilizing the stop condition implemented in Python
    results = tuner.Tune(kernel, ktt.ConfigurationCount(50))

    logPath = getLogPath(modelInfo, batchInfo)
    os.makedirs(Path(logPath).parent, exist_ok=True)

    tuner.SaveResults(results, logPath, ktt.OutputFormat.XML)


if __name__ == '__main__':
    deviceIndex = int(sys.argv[1]) if len(sys.argv) >= 2 else 0
    kernelFile = sys.argv[2] if len(sys.argv) >= 3 else './CudaKernel.cu'

    runTuning(deviceIndex, kernelFile)
