import sys
import numpy as np
import pyktt as ktt

def main():
    # Initialize platform index, device index and paths to kernels.
    platformIndex = 0
    deviceIndex = 0
    kernelFile = "./CoulombSum2d.cl"
    referenceKernelFile = "./CoulombSum2dReference.cl"
    argc = len(sys.argv)
    
    if argc >= 2:
        platformIndex = sys.argv[1]

        if argc >= 3:
            deviceIndex = sys.argv[2]
            
            if argc >= 4:
                kernelFile = sys.argv[3]

                if argc >= 5:
                    referenceKernelFile = sys.argv[4]

    # Declare kernel parameters.
    ndRangeDimensions = ktt.DimensionVector(512, 512)
    workGroupDimensions = ktt.DimensionVector()
    referenceWorkGroupDimensions = ktt.DimensionVector(16, 16)
    # Total NDRange size matches number of grid points.
    numberOfGridPoints = ndRangeDimensions.GetSizeX() * ndRangeDimensions.GetSizeY()
    # If higher than 4k, computations with constant memory enabled will be invalid on many devices due to constant memory capacity limit.
    numberOfAtoms = 4000

    # Declare data variables.
    gridSpacing = 0.5
    
    rng = np.random.default_rng()
    atomInfo = np.zeros(4 * numberOfAtoms, dtype = np.single)
    atomInfoX = 40.0 * rng.random(numberOfAtoms, dtype = np.single)
    atomInfoY = 40.0 * rng.random(numberOfAtoms, dtype = np.single)
    atomInfoZ = 40.0 * rng.random(numberOfAtoms, dtype = np.single)
    atomInfoW = rng.random(numberOfAtoms, dtype = np.single)
    energyGrid = np.zeros(numberOfGridPoints, dtype = np.single)
    
    for i in range(numberOfAtoms):
        atomInfo[4 * i] = atomInfoX[i]
        atomInfo[4 * i + 1] = atomInfoY[i]
        atomInfo[4 * i + 2] = atomInfoZ[i]
        atomInfo[4 * i + 3] = atomInfoW[i]

    tuner = ktt.Tuner(platformIndex, deviceIndex, ktt.ComputeApi.OpenCL)
    tuner.SetCompilerOptions("-cl-fast-relaxed-math")
    tuner.SetTimeUnit(ktt.TimeUnit.Microseconds)

    # Add two kernels to tuner, one of the kernels acts as a reference kernel.
    definition = tuner.AddKernelDefinitionFromFile("directCoulombSum", kernelFile, ndRangeDimensions, workGroupDimensions)
    kernel = tuner.CreateSimpleKernel("CoulombSum", definition)

    referenceDefinition = tuner.AddKernelDefinitionFromFile("directCoulombSumReference", referenceKernelFile,
        ndRangeDimensions, referenceWorkGroupDimensions)
    referenceKernel = tuner.CreateSimpleKernel("CoulombSumReference", referenceDefinition)

    # Add several parameters to tuned kernel, some of them utilize constraint function and thread modifiers.
    tuner.AddParameter(kernel, "INNER_UNROLL_FACTOR", [0, 1, 2, 4, 8, 16, 32])
    tuner.AddParameter(kernel, "USE_CONSTANT_MEMORY", [0, 1])
    tuner.AddParameter(kernel, "VECTOR_TYPE", [1, 2, 4, 8])
    tuner.AddParameter(kernel, "USE_SOA", [0, 1, 2])

    # Using vectorized SoA only makes sense when vectors are longer than 1.
    vectorizedSoA = lambda vector: vector[0] > 1 or vector[1] != 2
    tuner.AddConstraint(kernel, ["VECTOR_TYPE", "USE_SOA"], vectorizedSoA)

    # Divide NDRange in dimension x by OUTER_UNROLL_FACTOR.
    tuner.AddParameter(kernel, "OUTER_UNROLL_FACTOR", [1, 2, 4, 8])
    tuner.AddThreadModifier(kernel, [definition], ktt.ModifierType.Global, ktt.ModifierDimension.X, "OUTER_UNROLL_FACTOR",
        ktt.ModifierAction.Divide)

    # Multiply work-group size in dimensions x and y by the following parameters (effectively setting work-group size to their values).
    tuner.AddParameter(kernel, "WORK_GROUP_SIZE_X", [4, 8, 16, 32])
    tuner.AddThreadModifier(kernel, [definition], ktt.ModifierType.Local, ktt.ModifierDimension.X, "WORK_GROUP_SIZE_X",
        ktt.ModifierAction.Multiply)
    tuner.AddParameter(kernel, "WORK_GROUP_SIZE_Y", [1, 2, 4, 8, 16, 32])
    tuner.AddThreadModifier(kernel, [definition], ktt.ModifierType.Local, ktt.ModifierDimension.Y, "WORK_GROUP_SIZE_Y",
        ktt.ModifierAction.Multiply)

    # Add all kernel arguments.
    atomInfoId = tuner.AddArgumentVectorFloat(atomInfo, ktt.ArgumentAccessType.ReadOnly)
    atomInfoXId = tuner.AddArgumentVectorFloat(atomInfoX, ktt.ArgumentAccessType.ReadOnly)
    atomInfoYId = tuner.AddArgumentVectorFloat(atomInfoY, ktt.ArgumentAccessType.ReadOnly)
    atomInfoZId = tuner.AddArgumentVectorFloat(atomInfoZ, ktt.ArgumentAccessType.ReadOnly)
    atomInfoWId = tuner.AddArgumentVectorFloat(atomInfoW, ktt.ArgumentAccessType.ReadOnly)
    numberOfAtomsId = tuner.AddArgumentScalarInt(numberOfAtoms)
    gridSpacingId = tuner.AddArgumentScalarFloat(gridSpacing)
    energyGridId = tuner.AddArgumentVectorFloat(energyGrid, ktt.ArgumentAccessType.ReadWrite)

    # Set arguments for both tuned and reference kernel definitions, order of arguments is important.
    tuner.SetArguments(definition, [atomInfoId, atomInfoXId, atomInfoYId, atomInfoZId, atomInfoWId, numberOfAtomsId,
        gridSpacingId, energyGridId])
    tuner.SetArguments(referenceDefinition, [atomInfoId, numberOfAtomsId, gridSpacingId, energyGridId])

    # Set searcher to random.
    tuner.SetSearcher(kernel, ktt.RandomSearcher())

    # Specify custom tolerance threshold for validation of floating-point arguments. Default threshold is 1e-4.
    tuner.SetValidationMethod(ktt.ValidationMethod.SideBySideComparison, 0.01)

    # Set reference kernel which validates results provided by the tuned kernel.
    tuner.SetReferenceKernel(energyGridId, referenceKernel, ktt.KernelConfiguration())

    # Launch kernel tuning, end after 1 minute.
    results = tuner.Tune(kernel, ktt.TuningDuration(60.0))

    # Save tuning results to JSON file.
    tuner.SaveResults(results, "CoulombSum2dOutput", ktt.OutputFormat.JSON)

if __name__ == "__main__":
    main()
