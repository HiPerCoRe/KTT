import ctypes
import sys
import numpy as np
import pyktt as ktt

def computeReference(a, b, scalar, buffer):
    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.POINTER(ctypes.c_float)
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_void_p]
    floatList = ctypes.pythonapi.PyCapsule_GetPointer(buffer, None)
    
    for i in range(len(a)):
        floatList[i] = a[i] + b[i] + scalar

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
    # Block size is initialized to one in this case, it will be controlled with tuning parameter which is added later.
    blockDimensions = ktt.DimensionVector()
    
    a = np.arange(1.0, numberOfElements + 1, dtype = np.single)
    b = np.arange(1.0, numberOfElements + 1, dtype = np.single)
    result = np.zeros(numberOfElements, dtype = np.single)
    scalarValue = 3.0
    
    tuner = ktt.Tuner(0, deviceIndex, ktt.ComputeApi.CUDA)

    definition = tuner.AddKernelDefinitionFromFile("vectorAddition", kernelFile, gridDimensions, blockDimensions)
    
    aId = tuner.AddArgumentVectorFloat(a, ktt.ArgumentAccessType.ReadOnly)
    bId = tuner.AddArgumentVectorFloat(b, ktt.ArgumentAccessType.ReadOnly)
    resultId = tuner.AddArgumentVectorFloat(result, ktt.ArgumentAccessType.WriteOnly)
    scalarId = tuner.AddArgumentScalarFloat(scalarValue)
    tuner.SetArguments(definition, [aId, bId, resultId, scalarId])

    kernel = tuner.CreateSimpleKernel("Addition", definition)
    
    # Set reference computation for the result argument which will be used by the tuner to automatically validate kernel output.
    # The computation function receives buffer on input, where the reference result should be saved. The size of buffer corresponds
    # to the validated argument size.
    reference = lambda buffer : computeReference(a, b, scalarValue, buffer)
    tuner.SetReferenceComputation(resultId, reference)

    # Add new kernel parameter. Specify parameter name and possible values. When kernel is tuned, the parameter value is added
    # to the beginning of kernel source as preprocessor definition. E.g., for value of this parameter equal to 32, it is added
    # as "#define multiply_block_size 32".
    tuner.AddParameter(kernel, "multiply_block_size", [32, 64, 128, 256])

    # In this case, the parameter also affects block size. This is specified by adding a thread modifier. ModifierType specifies
    # that parameter affects block size of a kernel, ModifierAction specifies that block size is multiplied by value of the
    # parameter, ModifierDimension specifies that dimension X of a thread block is affected by the parameter. It is also possible
    # to specify which definitions are affected by the modifier. In this case, only one definition is affected. The default block
    # size inside kernel definition was set to one. This means that the block size of the definition is controlled explicitly by
    # value of this parameter. E.g., size of one is multiplied by 32, which means that result size is 32.
    tuner.AddThreadModifier(kernel, [definition], ktt.ModifierType.Local, ktt.ModifierDimension.X, "multiply_block_size",
        ktt.ModifierAction.Multiply)

    # Previously added parameter affects thread block size of kernel. However, when block size is changed, grid size has to be
    # modified as well, so that grid size multiplied by block size remains constant. This means that another modifier which divides
    # grid size has to be added.
    tuner.AddThreadModifier(kernel, [definition], ktt.ModifierType.Global, ktt.ModifierDimension.X, "multiply_block_size",
        ktt.ModifierAction.Divide)

    tuner.SetTimeUnit(ktt.TimeUnit.Microseconds)

    # Perform tuning for the specified kernel. This generates multiple versions of the kernel based on provided tuning parameters
    # and their values. In this case, 4 different versions of kernel will be run.
    results = tuner.Tune(kernel)

    # Save tuning results to JSON file.
    tuner.SaveResults(results, "TuningOutput", ktt.OutputFormat.JSON)

if __name__ == "__main__":
    main()
