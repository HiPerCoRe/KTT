import ctypes
import random
import sys
import ktt

def reference(buffer, src):
    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.POINTER(ctypes.c_float)
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_void_p]
    result = ctypes.pythonapi.PyCapsule_GetPointer(buffer, None)
    resSize = len(src)
    resD = [0.0 for i in range(resSize)]

    for i in range(resSize):
        resD[i] = src[i]

    while resSize > 1:
        for i in range(int(resSize / 2)):
            resD[i] = resD[i * 2] + resD[i * 2 + 1]

        if resSize % 2 != 0:
            resD[int(resSize / 2) - 1] += resD[resSize - 1]

        resSize = int(resSize / 2)

    print("Reference in double: " + str(resD[0]))
    result[0] = resD[0]

def launcher(interface, definition, srcId, dstId, nId, outOffsetId, inOffsetId):
    globalSize = interface.GetCurrentGlobalSize(definition)
    localSize = interface.GetCurrentLocalSize(definition)
    pairs = interface.GetCurrentConfiguration().GetPairs()
    myGlobalSize = globalSize

    # change global size for constant numbers of work-groups
    # this may be done by thread modifier operators as well
    if ktt.ParameterPair.GetParameterValue(pairs, "UNBOUNDED_WG") == 0:
        myGlobalSize = ktt.DimensionVector(ktt.ParameterPair.GetParameterValue(pairs, "WG_NUM") * localSize.GetSizeX())

    # execute reduction kernel
    interface.RunKernel(definition, myGlobalSize, localSize)

    # execute kernel log n times, when atomics are not used 
    if ktt.ParameterPair.GetParameterValue(pairs, "USE_ATOMICS") == 0:
        n = int(globalSize.GetSizeX() / localSize.GetSizeX())
        inOffset = 0
        outOffset = n
        vectorSize = ktt.ParameterPair.GetParameterValue(pairs, "VECTOR_SIZE")
        wgSize = localSize.GetSizeX()
        iterations = 0 # make sure the end result is in the correct buffer

        while n > 1 or iterations % 2 == 1:
            interface.SwapArguments(definition, srcId, dstId)
            myGlobalSize.SetSizeX(int((n + vectorSize - 1) / vectorSize))
            myGlobalSize.SetSizeX(int((myGlobalSize.GetSizeX() - 1) / wgSize + 1) * wgSize)
            
            if myGlobalSize == localSize:
                outOffset = 0 # only one WG will be executed
            
            interface.UpdateScalarArgumentInt(nId, n)
            interface.UpdateScalarArgumentInt(outOffsetId, outOffset)
            interface.UpdateScalarArgumentInt(inOffsetId, inOffset)

            interface.RunKernel(definition, myGlobalSize, localSize)
            n = int((n + wgSize * vectorSize - 1) / (wgSize * vectorSize))
            inOffset = int(outOffset / vectorSize) # input is vectorized, output is scalar
            outOffset += n
            iterations += 1

def main():
    platformIndex = 0
    deviceIndex = 0
    kernelFile = "./Reduction.cu"
    argc = len(sys.argv)
    
    if argc >= 2:
        platformIndex = sys.argv[1]

        if argc >= 3:
            deviceIndex = sys.argv[2]
            
            if argc >= 4:
                kernelFile = sys.argv[3]

    n = 64 * 1024 * 1024
    nAlloc = int((n + 16 - 1) / 16) * 16 # pad to the longest vector size
    src = [0.0 for i in range(nAlloc)]
    dst = [0.0 for i in range(nAlloc)]
    random.seed(17)
    
    for i in range(n):
        src[i] = random.uniform(0.0, 1000.0)

    tuner = ktt.Tuner(platformIndex, deviceIndex, ktt.ComputeApi.CUDA)
    tuner.SetGlobalSizeType(ktt.GlobalSizeType.OpenCL)
    tuner.SetTimeUnit(ktt.TimeUnit.Microseconds)
    
    nUp = int((n + 512 - 1) / 512) * 512 # maximum WG size used in tuning parameters
    ndRangeDimensions = ktt.DimensionVector(nUp)
    workGroupDimensions = ktt.DimensionVector()
    definition = tuner.AddKernelDefinitionFromFile("reduce", kernelFile, ndRangeDimensions, workGroupDimensions)

    srcId = tuner.AddArgumentVectorFloat(src, ktt.ArgumentAccessType.ReadWrite)
    dstId = tuner.AddArgumentVectorFloat(dst, ktt.ArgumentAccessType.ReadWrite)
    nId = tuner.AddArgumentScalarInt(n)
    offset = 0
    inOffsetId = tuner.AddArgumentScalarInt(offset)
    outOffsetId = tuner.AddArgumentScalarInt(offset)
    tuner.SetArguments(definition, [srcId, dstId, nId, inOffsetId, outOffsetId])

    kernel = tuner.CreateSimpleKernel("Reduction", definition)

    # get number of compute units
    di = tuner.GetCurrentDeviceInfo()
    print("Number of compute units: " + str(di.GetMaxComputeUnits()))
    cus = di.GetMaxComputeUnits()
    
    tuner.AddParameter(kernel, "WORK_GROUP_SIZE_X", [32, 64, 128, 256, 512])
    tuner.AddThreadModifier(kernel, [definition], ktt.ModifierType.Local, ktt.ModifierDimension.X, "WORK_GROUP_SIZE_X",
        ktt.ModifierAction.Multiply)
    tuner.AddParameter(kernel, "UNBOUNDED_WG", [0, 1])
    tuner.AddParameter(kernel, "WG_NUM", [0, cus, cus * 2, cus * 4, cus * 8, cus * 16])
    tuner.AddParameter(kernel, "VECTOR_SIZE", [1, 2, 4])
    tuner.AddThreadModifier(kernel, [definition], ktt.ModifierType.Global, ktt.ModifierDimension.X, "VECTOR_SIZE",
        ktt.ModifierAction.Divide)
    tuner.AddParameter(kernel, "USE_ATOMICS", [0, 1])
    
    persistConstraint = lambda v: (v[0] != 0 and v[1] == 0) or (v[0] == 0 and v[1] > 0)
    tuner.AddConstraint(kernel, ["UNBOUNDED_WG", "WG_NUM"], persistConstraint)
    persistentAtomic = lambda v: (v[0] == 1) or (v[0] == 0 and v[1] == 1)
    tuner.AddConstraint(kernel, ["UNBOUNDED_WG", "USE_ATOMICS"], persistentAtomic)
    unboundedWG = lambda v: v[0] == 0 or v[1] >= 32
    tuner.AddConstraint(kernel, ["UNBOUNDED_WG", "WORK_GROUP_SIZE_X"], unboundedWG)
    
    referenceComp = lambda buffer: reference(buffer, src)
    tuner.SetReferenceComputation(dstId, referenceComp)

    tuner.SetValidationMethod(ktt.ValidationMethod.SideBySideComparison, float(n) * 10000.0 / 10000000.0)
    tuner.SetValidationRange(dstId, 1)

    kernelLauncher = lambda interface: launcher(interface, definition, srcId, dstId, nId, outOffsetId, inOffsetId)
    tuner.SetLauncher(kernel, kernelLauncher)
    
    results = tuner.Tune(kernel)
    tuner.SaveResults(results, "ReductionOutput", ktt.OutputFormat.JSON)

if __name__ == "__main__":
    main()
