import ctypes
import sys
import pyktt as ktt

def main():
    # Initialize device index and path to kernel.
    deviceIndex = 0
    kernelFile = "./CudaKernel.cu"

    argc = len(sys.argv)
    
    if argc >= 2:
        deviceIndex = sys.argv[1]

        if argc >= 3:
            kernelFile = sys.argv[2]

    # Declare kernel parameters and data variables.
    numberOfElements = 1024 * 1024
    # Dimensions of block and grid are specified with DimensionVector. Only single dimension is utilized in this tutorial.
    # In general, DimensionVector supports up to three dimensions.
    blockDimensions = ktt.DimensionVector(256)
    gridSize = int(numberOfElements / blockDimensions.GetSizeX())
    gridDimensions = ktt.DimensionVector(gridSize)
    
    a = [i * 1.0 for i in range(numberOfElements)]
    b = [i * 1.0 for i in range(numberOfElements)]
    result = [0.0 for i in range(numberOfElements)]
    
    # Create new tuner for the specified device, tuner uses CUDA as compute API. Platform index is ignored when using CUDA.
    tuner = ktt.Tuner(0, deviceIndex, ktt.ComputeApi.CUDA)

    # Add new kernel definition. Specify kernel function name, path to source file, default grid dimensions and block dimensions.
    # KTT returns handle to the newly added definition, which can be used to reference it in other API methods.
    definition = tuner.AddKernelDefinitionFromFile("vectorAddition", kernelFile, gridDimensions, blockDimensions)
    
    # Add new kernel arguments to tuner. Argument data is copied from std::vector containers. Specify whether the arguments are
    # used as input or output. KTT returns handle to the newly added argument, which can be used to reference it in other API
    # methods. 
    aId = tuner.AddArgumentVectorFloat(a, ktt.ArgumentAccessType.ReadOnly)
    bId = tuner.AddArgumentVectorFloat(b, ktt.ArgumentAccessType.ReadOnly)
    resultId = tuner.AddArgumentVectorFloat(result, ktt.ArgumentAccessType.WriteOnly)
    
    # Set arguments for the kernel definition. The order of argument ids must match the order of arguments inside corresponding
    # CUDA kernel function.
    tuner.SetArguments(definition, [aId, bId, resultId])
    
    # Create simple kernel from the specified definition. Specify name which will be used during logging and output operations.
    # In more complex scenarios, kernels can have multiple definitions. Definitions can be shared between multiple kernels.
    kernel = tuner.CreateSimpleKernel("Addition", definition)
    
    # Set time unit used during printing of kernel duration. The default time unit is milliseconds, but since computation in
    # this tutorial is very short, microseconds are used instead.
    tuner.SetTimeUnit(ktt.TimeUnit.Microseconds)

    # Run the specified kernel. The second argument is related to kernel tuning and will be described in further tutorials.
    # In this case, empty object is passed in its place. The third argument is used to retrieve the kernel output. For each kernel
    # argument that is retrieved, one BufferOutputDescriptor must be specified. Each of these descriptors contains id of the retrieved
    # argument and memory location where the argument data will be stored. Optionally, it can also include number of bytes to be retrieved,
    # if only a part of the argument is needed. Note that the memory location size needs to be equal or greater than the retrieved
    # argument size.
    array = (ctypes.c_float * numberOfElements)()
    ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
    arrayCapsule = ctypes.pythonapi.PyCapsule_New(array)
    tuner.Run(kernel, ktt.KernelConfiguration(), [ktt.BufferOutputDescriptor(resultId, arrayCapsule)])

    # Print first ten elements from the result to check they were computed correctly.
    print("Printing the first 10 elements from result: ")

    for i in range(10):
        print(array[i])

if __name__ == "__main__":
    main()
