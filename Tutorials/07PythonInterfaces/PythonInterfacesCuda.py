import ctypes
import sys
import pyktt as ktt

# Implement custom stop condition in Python. The interface is the same as in C++. Note that it is necessary to call
# the parent class constructor from inheriting constructor.
class PyConfigurationFraction(ktt.StopCondition):
    def __init__(self, fraction):
        ktt.StopCondition.__init__(self)
        self.fraction = max(min(fraction, 1.0), 0.0)

    def IsFulfilled(self):
        return self.currentCount / self.totalCount >= self.fraction

    def Initialize(self, configurationsCount):
        self.currentCount = 0;
        self.totalCount = max(1, configurationsCount)

    def Update(self, result):
        self.currentCount += 1

    def GetStatusString(self):
        return "Current fraction of explored configurations: " + str(self.currentCount / self.totalCount) + " / " + str(self.fraction)
    
    fraction = 0.0
    currentCount = 0
    totalCount = 0

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

    numberOfElements = 1024 * 1024
    gridDimensions = ktt.DimensionVector(numberOfElements)
    blockDimensions = ktt.DimensionVector()
    
    a = [i * 1.0 for i in range(numberOfElements)]
    b = [i * 1.0 for i in range(numberOfElements)]
    result = [0.0 for i in range(numberOfElements)]
    scalarValue = 3.0
    
    tuner = ktt.Tuner(0, deviceIndex, ktt.ComputeApi.CUDA)

    definition = tuner.AddKernelDefinitionFromFile("vectorAddition", kernelFile, gridDimensions, blockDimensions)
    
    aId = tuner.AddArgumentVectorFloat(a, ktt.ArgumentAccessType.ReadOnly)
    bId = tuner.AddArgumentVectorFloat(b, ktt.ArgumentAccessType.ReadOnly)
    resultId = tuner.AddArgumentVectorFloat(result, ktt.ArgumentAccessType.WriteOnly)
    scalarId = tuner.AddArgumentScalarFloat(scalarValue)
    tuner.SetArguments(definition, [aId, bId, resultId, scalarId])

    kernel = tuner.CreateSimpleKernel("Addition", definition)
    
    tuner.AddParameter(kernel, "multiply_block_size", [32, 64, 128, 256])
    tuner.AddThreadModifier(kernel, [definition], ktt.ModifierType.Local, ktt.ModifierDimension.X, "multiply_block_size",
        ktt.ModifierAction.Multiply)
    tuner.AddThreadModifier(kernel, [definition], ktt.ModifierType.Global, ktt.ModifierDimension.X, "multiply_block_size",
        ktt.ModifierAction.Divide)

    # Make tuner user the searcher implemented in Python.
    tuner.SetSearcher(kernel, PyRandomSearcher())
    
    tuner.SetTimeUnit(ktt.TimeUnit.Microseconds)
    
    # Begin tuning utilizing the stop condition implemented in Python.
    results = tuner.Tune(kernel, PyConfigurationFraction(0.4))
    tuner.SaveResults(results, "TuningOutput", ktt.OutputFormat.JSON)

if __name__ == "__main__":
    main()
