import sys
import numpy as np
import pyktt as ktt

# Implement custom searcher in Python. The interface is the same as in C++, including helper methods defined in
# the parent class. Note that it is necessary to call the parent class constructor from inheriting constructor.
class PyRandomSearcher(ktt.Searcher):
    def __init__(self):
        ktt.Searcher.__init__(self)

    def OnInitialize(self):
        self.currentConfiguration = self.GetRandomConfiguration()

    def CalculateNextConfiguration(self, previousResult):
        self.currentConfiguration = self.GetRandomConfiguration()
        return True

    def GetCurrentConfiguration(self):
        return self.currentConfiguration

    currentConfiguration = ktt.KernelConfiguration()

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

    definition = tuner.AddKernelDefinitionFromFile("directCoulombSum", kernelFile, gridDimensions, blockDimensions)
    tuner.SetCompilerOptions("-use_fast_math")
    tuner.SetTimeUnit(ktt.TimeUnit.Microseconds)

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
    tuner.SetSearcher(kernel, PyRandomSearcher())

    # Begin tuning utilizing the stop condition implemented in Python.
    results = tuner.Tune(kernel)
    tuner.SaveResults(results, "TuningOutput", ktt.OutputFormat.JSON)

if __name__ == "__main__":
    main()
