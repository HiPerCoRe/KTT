#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <Ktt.h>

#if defined(_MSC_VER)
const std::string kernelPrefix = "";
#else
const std::string kernelPrefix = "../";
#endif

  //void computeResult()
  //{
  //  int i, j, j1, j2;

  //  /* Determine mean of column vectors of input data matrix */
  //  for (j = 0; j < m; j++) {
  //    mean[j] = 0.0;
  //    for (i = 0; i < n; i++) {
  //      mean[j] += data[i * m + j];
  //    }
  //    mean[j] /= float_n;
  //  }

  //  /* Center the column vectors. */
  //  for (i = 0; i < n; i++) {
  //    for (j = 0; j < m; j++) {
  //      data[i * m + j] -= mean[j];
  //    }
  //  }

  //  ///* Calculate the m * m covariance matrix. */
  //  for (j1 = 0; j1 < m; j1++) {
  //    for (j2 = j1; j2 < m; j2++) {
  //      symmat[j1 * m + j2] = 0.0;
  //      for (i = 0; i < n; i++) {
  //        symmat[j1 * m + j2] += data[i * m + j1] * data[i * m + j2];
  //      }
  //      symmat[j2 * m + j1] = symmat[j1 * m + j2];
  //    }
  //  }
  //}

// Helper function to determine whether or not 'a' is a multiple of 'b'
bool IsMultiple(size_t a, size_t b) { return ((a / b) * b == a); };

int main(int argc, char** argv)
{
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = kernelPrefix + "../Examples/Covariance/Covariance.cl";
    std::string referenceKernelFile = kernelPrefix + "../Examples/Covariance/CovarianceReference.cl";
    std::string gemmFile = kernelPrefix + "../Examples/Covariance/Gemm.cl";

    if (argc >= 2)
    {
        platformIndex = std::stoul(std::string(argv[1]));

        if (argc >= 3)
        {
            deviceIndex = std::stoul(std::string(argv[2]));

            if (argc >= 4)
            {
                kernelFile = std::string(argv[3]);

                if (argc >= 5)
                {
                    referenceKernelFile = std::string(argv[4]);
                }
            }
        }
    }

    /* Problem size. */
    const int n = 1024;
    const int m = 1024;

    // New NVidia GPUs have max.workgroup size of 1024
    const int maxWorkGroupSize = 1024;

    // Kernel dimensions
    const ktt::DimensionVector ndRangeDim1D(m, 1);
    const ktt::DimensionVector workGroupDim1D(256, 1);
    const ktt::DimensionVector ndRangeDim2D(m, m);
    const ktt::DimensionVector workGroupDim2D(32, 8);

    // Declare data variables
    const float floatN = static_cast<float>(n);
    std::vector<float> data(m * n);
    std::vector<float> symmat(m * m, 0.0f);
    std::vector<float> mean(m, 0.0f);

    // Initialize data
    std::random_device device;
    std::default_random_engine engine(device());
    std::uniform_real_distribution<float> distribution(0.0f, 100.0f);

    for (auto& v : data)
    {
        v = distribution(engine);
    }

    // Create tuner object for specified platform and device
    ktt::Tuner tuner(platformIndex, deviceIndex, ktt::ComputeApi::OpenCL);
    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    // Add kernels to tuner, one of the kernels acts as reference kernel
    const ktt::KernelDefinitionId refMeanDefinition = tuner.AddKernelDefinitionFromFile("mean_kernel_reference", referenceKernelFile,
        ndRangeDim1D, workGroupDim1D);
    const ktt::KernelDefinitionId refReduceDefinition = tuner.AddKernelDefinitionFromFile("reduce_kernel_reference", referenceKernelFile,
        ndRangeDim2D, workGroupDim2D);
    const ktt::KernelDefinitionId refCovarDefinition = tuner.AddKernelDefinitionFromFile("covar_kernel_reference", referenceKernelFile,
        ndRangeDim1D, workGroupDim1D);

    const ktt::KernelDefinitionId meanDefinition = tuner.AddKernelDefinitionFromFile("mean_kernel", kernelFile, ndRangeDim1D,
        workGroupDim1D);
    const ktt::KernelDefinitionId reduceDefinition = tuner.AddKernelDefinitionFromFile("reduce_kernel", kernelFile, ndRangeDim2D,
        workGroupDim2D);
    const ktt::KernelDefinitionId covarDefinition = tuner.AddKernelDefinitionFromFile("covar_kernel", kernelFile, ndRangeDim1D,
        workGroupDim1D);
    const ktt::KernelDefinitionId gemmDefinition = tuner.AddKernelDefinitionFromFile("gemm_fast", gemmFile, ndRangeDim2D,
        ktt::DimensionVector());
    const ktt::KernelDefinitionId triangularToSymmetricDefinition = tuner.AddKernelDefinitionFromFile("triangular_to_symmetric",
        kernelFile, ndRangeDim2D, workGroupDim2D);

    const ktt::KernelId kernel = tuner.CreateCompositeKernel("Covariance", {refMeanDefinition, refReduceDefinition, refCovarDefinition,
        meanDefinition, reduceDefinition, covarDefinition, gemmDefinition, triangularToSymmetricDefinition }, [refMeanDefinition,
        refReduceDefinition, refCovarDefinition, meanDefinition, reduceDefinition, covarDefinition, gemmDefinition,
        triangularToSymmetricDefinition](ktt::ComputeInterface& interface)
    {
        const std::vector<ktt::ParameterPair>& parameterValues = interface.GetCurrentConfiguration().GetPairs();
        const uint64_t kernelVariant = ktt::ParameterPair::GetParameterValue<uint64_t>(parameterValues, "KERNEL");

        if (kernelVariant == 0)
        {
            interface.RunKernel(refMeanDefinition);
            interface.RunKernel(refReduceDefinition);
            interface.RunKernel(refCovarDefinition);
        }
        else if (kernelVariant == 1)
        {
            interface.RunKernel(meanDefinition);
            interface.RunKernel(reduceDefinition);
            interface.RunKernel(covarDefinition);
        }
        else if (kernelVariant == 2)
        {
            interface.RunKernel(meanDefinition);
            interface.RunKernel(reduceDefinition);
            interface.RunKernel(gemmDefinition);

            if (ktt::ParameterPair::GetParameterValue<uint64_t>(parameterValues, "SYM_STORE") == 0)
            {
                interface.RunKernel(triangularToSymmetricDefinition);
            }
        }
    });

    // Add parameters to tuned kernel
    // Some parameters are commented out to cut down the tuned space - it is now the same as the
    // simpler and commonly tuned space in CLBlast (plus our new parameters).
    // KERNEL: 1 - reference kernels, 0 - edited reference kernels, 2 - use GEMM as third kernel
    tuner.AddParameter(kernel, "KERNEL", std::vector<uint64_t>{1, 0, 2});
    tuner.AddParameter(kernel, "MWG", std::vector<uint64_t>{16, 32, 64 /* , 128 */});
    tuner.AddParameter(kernel, "NWG", std::vector<uint64_t>{16, 32, 64 /* , 128 */});
    tuner.AddParameter(kernel, "KWG", std::vector<uint64_t>{/* 16,  */ 32});
    tuner.AddParameter(kernel, "MDIMC", std::vector<uint64_t>{8, 16, 32});
    tuner.AddParameter(kernel, "NDIMC", std::vector<uint64_t>{8, 16, 32});
    tuner.AddParameter(kernel, "MDIMA", std::vector<uint64_t>{8, 16, 32});
    tuner.AddParameter(kernel, "NDIMB", std::vector<uint64_t>{8, 16, 32});
    tuner.AddParameter(kernel, "KWI", std::vector<uint64_t>{2 /* , 8 */});
    tuner.AddParameter(kernel, "VWM", std::vector<uint64_t>{1, 2, 4 /* , 8 */});
    tuner.AddParameter(kernel, "VWN", std::vector<uint64_t>{1, 2, 4 /* , 8 */});
    tuner.AddParameter(kernel, "STRM", std::vector<uint64_t>{0 /* , 1 */});
    tuner.AddParameter(kernel, "STRN", std::vector<uint64_t>{0 /* , 1 */});
    tuner.AddParameter(kernel, "SA", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "SB", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "SYMMETRIC", std::vector<uint64_t>{0, 1});
    // If SYMMETRIC == 1:
    // SYM_STORE == 1: if VWM == 1, store the symmetric value right in the GEMM kernel
    // SYM_STORE == 0: use fourth kernel to make the triangular matrix symmetric
    tuner.AddParameter(kernel, "SYM_STORE", std::vector<uint64_t>{0, 1});

    // Tests single precision (SGEMM)
    tuner.AddParameter(kernel, "PRECISION", std::vector<uint64_t>{32});

    // Set kernel sizes
    auto globalModifier = [](const uint64_t size, const std::vector<uint64_t>& v)
    {
        return (size * v[0] / v[1]);
    };

    tuner.AddThreadModifier(kernel, {gemmDefinition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, {"MDIMC", "MWG"}, globalModifier);
    tuner.AddThreadModifier(kernel, {gemmDefinition}, ktt::ModifierType::Global, ktt::ModifierDimension::Y, {"NDIMC", "NWG"}, globalModifier);
    
    auto localModifier = [](const uint64_t /*size*/, const std::vector<uint64_t>& v) { return (v[0]); };
    tuner.AddThreadModifier(kernel, {gemmDefinition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, {"MDIMC"}, localModifier);
    tuner.AddThreadModifier(kernel, {gemmDefinition}, ktt::ModifierType::Local, ktt::ModifierDimension::Y, {"NDIMC"}, localModifier);

    auto multipleOfX = [](const std::vector<uint64_t>& v) { return IsMultiple(v[0], v[1]); };
    auto multipleOfXMulY = [](const std::vector<uint64_t>& v) { return IsMultiple(v[0], v[1] * v[2]); };
    auto multipleOfXMulYDivZ = [](const std::vector<uint64_t>& v)
    {
        return IsMultiple(v[0], (v[1] * v[2]) / v[3]);
    };

    // Sets constraints: Requirement for unrolling the KWG loop
    tuner.AddConstraint(kernel, {"KWG", "KWI"}, multipleOfX);

    // Sets constraints: Required for integer MWI and NWI
    tuner.AddConstraint(kernel, {"MWG", "MDIMC", "VWM"}, multipleOfXMulY);
    tuner.AddConstraint(kernel, {"NWG", "NDIMC", "VWN"}, multipleOfXMulY);

    // Sets constraints: Required for integer MWIA and NWIB
    tuner.AddConstraint(kernel, {"MWG", "MDIMA", "VWM"}, multipleOfXMulY);
    tuner.AddConstraint(kernel, {"NWG", "NDIMB", "VWN"}, multipleOfXMulY);

    // Sets constraints: KWG has to be a multiple of KDIMA = ((MDIMC*NDIMC)/(MDIMA)) and KDIMB = (...)
    tuner.AddConstraint(kernel, {"KWG", "MDIMC", "NDIMC", "MDIMA"}, multipleOfXMulYDivZ);
    tuner.AddConstraint(kernel, {"KWG", "MDIMC", "NDIMC", "NDIMB"}, multipleOfXMulYDivZ);

    // Don't use parameters for polybench reference kernels,
    auto reference = [](const std::vector<uint64_t>& v)
    {
        if (v[0] == 2)
        {
            return true;
        }

        return v[1] == 32 && v[2] == 32 && v[3] == 32 && v[4] == 8 && v[5] == 8 && v[6] == 8 && v[7] == 8 && v[8] == 2
            && v[9] == 1 && v[10] == 1 && v[11] == 0 && v[12] == 0 && v[13] == 1 && v[14] == 1 && v[15] == 0;
    };

    tuner.AddConstraint(kernel, {"KERNEL", "MWG", "NWG", "KWG", "MDIMC", "NDIMC", "MDIMA", "NDIMB", "KWI", "VWM", "VWN", "STRM",
        "STRN", "SA", "SB", "SYMMETRIC"}, reference);

    // New NVidia GPUs have max. workgroup size
    auto maxWgSize = [maxWorkGroupSize](const std::vector<uint64_t>& v) { return v[0] * v[1] <= maxWorkGroupSize; };
    tuner.AddConstraint(kernel, {"MDIMC", "NDIMC"}, maxWgSize);

    // Symmetric store can't be used for vectors
    auto symmetric = [](const std::vector<uint64_t>& v)
    {
        if (v[0] == 1)
        {
            if (v[1] == 1)
            {
                return v[2] == 1;
            }

            return true;
        }

        return v[1] == 0;
    };

    tuner.AddConstraint(kernel, {"SYMMETRIC", "SYM_STORE", "VWM"}, symmetric);

    // Add all arguments utilized by kernels
    const ktt::ArgumentId dataId = tuner.AddArgumentVector(data, ktt::ArgumentAccessType::ReadWrite);
    const ktt::ArgumentId symmatId = tuner.AddArgumentVector(symmat, ktt::ArgumentAccessType::ReadWrite);
    const ktt::ArgumentId meanId = tuner.AddArgumentVector(mean, ktt::ArgumentAccessType::ReadWrite);
    const ktt::ArgumentId mId = tuner.AddArgumentScalar(m);
    const ktt::ArgumentId nId = tuner.AddArgumentScalar(n);
    const ktt::ArgumentId floatnId = tuner.AddArgumentScalar(floatN);

    tuner.SetArguments(refMeanDefinition, {meanId, dataId, floatnId, mId, nId});
    tuner.SetArguments(refReduceDefinition, {meanId, dataId, mId, nId});
    tuner.SetArguments(refCovarDefinition, {symmatId, dataId, mId, nId});
    tuner.SetArguments(meanDefinition, {meanId, dataId, floatnId, mId, nId});
    tuner.SetArguments(reduceDefinition, {meanId, dataId, mId, nId});
    tuner.SetArguments(covarDefinition, {symmatId, dataId, mId, nId});
    tuner.SetArguments(gemmDefinition, {mId, nId, dataId, symmatId});
    tuner.SetArguments(triangularToSymmetricDefinition, {symmatId, mId});

    tuner.SetValidationMethod(ktt::ValidationMethod::SideBySideComparison, 128.0);

    tuner.SetReferenceComputation(symmatId, [&data, m, n, floatN](void* buffer)
    {
        std::vector<float> mean(m, 0.0f);
        float* symmat = static_cast<float*>(buffer);

        for (int j = 0; j < m; ++j)
        {
            mean[j] = 0.0;

            for (int i = 0; i < n; ++i)
            {
                mean[j] += data[i * m + j];
            }

            mean[j] /= floatN;
        }

        /* Center the column vectors. */
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                data[i * m + j] -= mean[j];
            }
        }

        /* Calculate the m * m covariance matrix. */
        for (int j1 = 0; j1 < m; ++j1)
        {
            for (int j2 = j1; j2 < m; ++j2)
            {
                symmat[j1 * m + j2] = 0.0;

                for (int i = 0; i < n; ++i)
                {
                    symmat[j1 * m + j2] += data[i * m + j1] * data[i * m + j2];
                }

                symmat[j2 * m + j1] = symmat[j1 * m + j2];
            }
        }
    });

    tuner.SetReferenceComputation(meanId, [&data, m, n, floatN](void* buffer)
    {
        float* mean = static_cast<float*>(buffer);

        for (int j = 0; j < m; ++j)
        {
            mean[j] = 0.0;

            for (int i = 0; i < n; ++i)
            {
                mean[j] += data[i * m + j];
            }

            mean[j] /= floatN;
        }
    });

    // Launch kernel tuning
    const auto results = tuner.TuneKernel(kernel);
    tuner.SaveResults(results, "Covariance", ktt::OutputFormat::XML);

    return 0;
}
