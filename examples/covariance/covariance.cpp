#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "tuner_api.h"

#if defined(_MSC_VER)
#  define KTT_KERNEL_FILE "../examples/covariance/covariance.cl"
#  define KTT_REFERENCE_KERNEL_FILE "../examples/covariance/covariance_ref.cl"
#  define KTT_GEMM_KERNEL_FILE "../examples/covariance/gemm.cl";
#else
#  define KTT_KERNEL_FILE "../../examples/covariance/covariance.cl"
#  define KTT_REFERENCE_KERNEL_FILE "../../examples/covariance/covariance_ref.cl"
#  define KTT_GEMM_KERNEL_FILE "../../examples/covariance/gemm.cl";
#endif

/* Problem size. */
#define N 1024
#define M 1024

// New NVidia GPUs have max.workgroup size of 1024
#define MAX_WORK_GROUP_SIZE 1024

class CovarianceCpu : public ktt::ReferenceClass {
 public:
  CovarianceCpu(const ktt::ArgumentId symmatId, const ktt::ArgumentId meanId,
      const std::vector<float>& data, const std::vector<float>& symmat,
      const std::vector<float>& mean, const float float_n)
      : symmatId(symmatId),
        meanId(meanId),
        data(data),
        symmat(symmat),
        mean(mean),
        float_n(float_n) {}

  // Method inherited from ReferenceClass, which computes reference result for all arguments that
  // are validated inside the class.
  void computeResult() override {
    int i, j, j1, j2;

    /* Determine mean of column vectors of input data matrix */
    for (j = 0; j < M; j++) {
      mean[j] = 0.0;
      for (i = 0; i < N; i++) {
        mean[j] += data[i * M + j];
      }
      mean[j] /= float_n;
    }

    /* Center the column vectors. */
    for (i = 0; i < N; i++) {
      for (j = 0; j < M; j++) {
        data[i * M + j] -= mean[j];
      }
    }

    ///* Calculate the m * m covariance matrix. */
    for (j1 = 0; j1 < M; j1++) {
      for (j2 = j1; j2 < M; j2++) {
        symmat[j1 * M + j2] = 0.0;
        for (i = 0; i < N; i++) {
          symmat[j1 * M + j2] += data[i * M + j1] * data[i * M + j2];
        }
        symmat[j2 * M + j1] = symmat[j1 * M + j2];
      }
    }
  }

  // Method inherited from ReferenceClass, which returns memory location where reference result for
  // corresponding argument is stored.
  void* getData(const ktt::ArgumentId id) override {
    if (id == symmatId) {
      return symmat.data();
    }

    if (id == meanId) {
      return mean.data();
    }

    return nullptr;
  }

 private:
  ktt::ArgumentId symmatId;
  ktt::ArgumentId meanId;
  std::vector<float> data;
  std::vector<float> symmat;
  std::vector<float> mean;
  const float float_n;
};

class CovarianceManipulator : public ktt::TuningManipulator {
 public:
  CovarianceManipulator(const ktt::KernelId refMeanKId, const ktt::KernelId refReduceKId,
      const ktt::KernelId refCovarKId, const ktt::KernelId meanKId, const ktt::KernelId reduceKid,
      const ktt::KernelId covarKId, const ktt::KernelId gemmKId,
      const ktt::KernelId triangularToSymmetricKId)
      : refMeanKId(refMeanKId),
        refReduceKId(refReduceKId),
        refCovarKId(refCovarKId),
        meanKId(meanKId),
        reduceKid(reduceKid),
        covarKId(covarKId),
        gemmKId(gemmKId),
        triangularToSymmetricKId(triangularToSymmetricKId) {}

  // LaunchComputation is responsible for actual execution of tuned kernel
  void launchComputation(const ktt::KernelId kernelId) override {
    std::vector<ktt::ParameterPair> parameterValues = getCurrentConfiguration();

    if (getParameterValue("KERNEL", parameterValues) == 0) {
      runKernel(refMeanKId);
      runKernel(refReduceKId);
      runKernel(refCovarKId);
    } else if (getParameterValue("KERNEL", parameterValues) == 1) {
      runKernel(meanKId);
      runKernel(reduceKid);
      runKernel(covarKId);
    } else if (getParameterValue("KERNEL", parameterValues) == 2) {
      runKernel(meanKId);
      runKernel(reduceKid);
      runKernel(gemmKId);
      if (getParameterValue("SYM_STORE", parameterValues) == 0) {
        runKernel(triangularToSymmetricKId);
      }
    }
  }

 private:
  const ktt::KernelId refMeanKId;
  const ktt::KernelId refReduceKId;
  const ktt::KernelId refCovarKId;
  const ktt::KernelId meanKId;
  const ktt::KernelId reduceKid;
  const ktt::KernelId covarKId;
  const ktt::KernelId gemmKId;
  const ktt::KernelId triangularToSymmetricKId;
};

// Helper function to determine whether or not 'a' is a multiple of 'b'
bool IsMultiple(size_t a, size_t b) { return ((a / b) * b == a); };

int main(int argc, char** argv) {
  // Initialize platform index, device index and paths to kernels
  ktt::PlatformIndex platformIndex = 0;
  ktt::DeviceIndex deviceIndex = 0;
  std::string kernelFile = KTT_KERNEL_FILE;
  std::string refKernelFile = KTT_REFERENCE_KERNEL_FILE;
  std::string gemmFile = KTT_GEMM_KERNEL_FILE;

  if (argc >= 2) {
    platformIndex = std::stoul(std::string(argv[1]));
    if (argc >= 3) {
      deviceIndex = std::stoul(std::string(argv[2]));
      if (argc >= 4) {
        kernelFile = std::string(argv[3]);
        if (argc >= 5) {
          refKernelFile = std::string(argv[4]);
        }
      }
    }
  }

  // Kernel dimensions
  const ktt::DimensionVector ndRangeDim1D(M, 1);
  const ktt::DimensionVector workGroupDim1D(256, 1);
  const ktt::DimensionVector ndRangeDim2D(M, M);
  const ktt::DimensionVector workGroupDim2D(32, 8);

  // Declare data variables
  const float float_n = (float)N;
  std::vector<float> data(M * N);
  std::vector<float> symmat(M * M, 0.0f);
  std::vector<float> mean(M, 0.0f);

  // Initialize data
  std::random_device device;
  std::default_random_engine engine(device());
  std::uniform_real_distribution<float> distribution(0.0f, 100.0f);

  for (auto& v : data) v = distribution(engine);

  // Create tuner object for specified platform and device
  ktt::Tuner tuner(platformIndex, deviceIndex);
  tuner.setPrintingTimeUnit(ktt::TimeUnit::Microseconds);

  // Add kernels to tuner, one of the kernels acts as reference kernel
  ktt::KernelId refMeanKId =
      tuner.addKernelFromFile(refKernelFile, "mean_kernel", ndRangeDim1D, workGroupDim1D);
  ktt::KernelId refReduceKId =
      tuner.addKernelFromFile(refKernelFile, "reduce_kernel", ndRangeDim2D, workGroupDim2D);
  ktt::KernelId refCovarKId =
      tuner.addKernelFromFile(refKernelFile, "covar_kernel", ndRangeDim1D, workGroupDim1D);

  ktt::KernelId meanKId =
      tuner.addKernelFromFile(kernelFile, "mean_kernel", ndRangeDim1D, workGroupDim1D);
  ktt::KernelId reduceKid =
      tuner.addKernelFromFile(kernelFile, "reduce_kernel", ndRangeDim2D, workGroupDim2D);
  ktt::KernelId covarKId =
      tuner.addKernelFromFile(kernelFile, "covar_kernel", ndRangeDim1D, workGroupDim1D);
  ktt::KernelId gemmKId =
      tuner.addKernelFromFile(gemmFile, "gemm_fast", ndRangeDim2D, ktt::DimensionVector());
  ktt::KernelId triangularToSymmetricKId =
      tuner.addKernelFromFile(kernelFile, "triangular_to_symmetric", ndRangeDim2D, workGroupDim2D);

  ktt::KernelId kernelId = tuner.addComposition("Covariance",
      std::vector<ktt::KernelId>{refMeanKId, refReduceKId, refCovarKId, meanKId, reduceKid,
          covarKId, gemmKId, triangularToSymmetricKId},
      std::make_unique<CovarianceManipulator>(refMeanKId, refReduceKId, refCovarKId, meanKId,
          reduceKid, covarKId, gemmKId, triangularToSymmetricKId));

  // Add parameters to tuned kernel
  // Some parameters are commented out to cut down the tuned space - it is now the same as the
  // simpler and commonly tuned space in CLBlast (plus our new parameters).
  // KERNEL: 1 - reference kernels, 0 - edited reference kernels, 2 - use GEMM as third kernel
  tuner.addParameter(kernelId, "KERNEL", {1, 0, 2});
  tuner.addParameter(kernelId, "MWG", {16, 32, 64 /* , 128 */});
  tuner.addParameter(kernelId, "NWG", {16, 32, 64 /* , 128 */});
  tuner.addParameter(kernelId, "KWG", {/* 16,  */ 32});
  tuner.addParameter(kernelId, "MDIMC", {8, 16, 32});
  tuner.addParameter(kernelId, "NDIMC", {8, 16, 32});
  tuner.addParameter(kernelId, "MDIMA", {8, 16, 32});
  tuner.addParameter(kernelId, "NDIMB", {8, 16, 32});
  tuner.addParameter(kernelId, "KWI", {2 /* , 8 */});
  tuner.addParameter(kernelId, "VWM", {1, 2, 4 /* , 8 */});
  tuner.addParameter(kernelId, "VWN", {1, 2, 4 /* , 8 */});
  tuner.addParameter(kernelId, "STRM", {0 /* , 1 */});
  tuner.addParameter(kernelId, "STRN", {0 /* , 1 */});
  tuner.addParameter(kernelId, "SA", {0, 1});
  tuner.addParameter(kernelId, "SB", {0, 1});
  tuner.addParameter(kernelId, "SYMMETRIC", {0, 1});
  // If SYMMETRIC == 1:
  // SYM_STORE == 1: if VWM == 1, store the symmetric value right in the GEMM kernel
  // SYM_STORE == 0: use fourth kernel to make the triangular matrix symmetric
  tuner.addParameter(kernelId, "SYM_STORE", {0, 1});

  // Tests single precision (SGEMM)
  tuner.addParameter(kernelId, "PRECISION", {32});

  // Set kernel sizes
  auto globalModifier = [](const size_t size, const std::vector<size_t>& v) {
    return (size * v[0] / v[1]);
  };
  tuner.setCompositionKernelThreadModifier(kernelId, gemmKId, ktt::ModifierType::Global,
      ktt::ModifierDimension::X, {"MDIMC", "MWG"}, globalModifier);
  tuner.setCompositionKernelThreadModifier(kernelId, gemmKId, ktt::ModifierType::Global,
      ktt::ModifierDimension::Y, {"NDIMC", "NWG"}, globalModifier);
  auto localModifier = [](const size_t size, const std::vector<size_t>& v) { return (v[0]); };
  tuner.setCompositionKernelThreadModifier(kernelId, gemmKId, ktt::ModifierType::Local,
      ktt::ModifierDimension::X, {"MDIMC"}, localModifier);
  tuner.setCompositionKernelThreadModifier(kernelId, gemmKId, ktt::ModifierType::Local,
      ktt::ModifierDimension::Y, {"NDIMC"}, localModifier);

  auto MultipleOfX = [](const std::vector<size_t>& v) { return IsMultiple(v[0], v[1]); };
  auto MultipleOfXMulY = [](const std::vector<size_t>& v) { return IsMultiple(v[0], v[1] * v[2]); };
  auto MultipleOfXMulYDivZ = [](const std::vector<size_t>& v) {
    return IsMultiple(v[0], (v[1] * v[2]) / v[3]);
  };

  // Sets constraints: Requirement for unrolling the KWG loop
  tuner.addConstraint(kernelId, {"KWG", "KWI"}, MultipleOfX);

  // Sets constraints: Required for integer MWI and NWI
  tuner.addConstraint(kernelId, {"MWG", "MDIMC", "VWM"}, MultipleOfXMulY);
  tuner.addConstraint(kernelId, {"NWG", "NDIMC", "VWN"}, MultipleOfXMulY);

  // Sets constraints: Required for integer MWIA and NWIB
  tuner.addConstraint(kernelId, {"MWG", "MDIMA", "VWM"}, MultipleOfXMulY);
  tuner.addConstraint(kernelId, {"NWG", "NDIMB", "VWN"}, MultipleOfXMulY);

  // Sets constraints: KWG has to be a multiple of KDIMA = ((MDIMC*NDIMC)/(MDIMA)) and KDIMB = (...)
  tuner.addConstraint(kernelId, {"KWG", "MDIMC", "NDIMC", "MDIMA"}, MultipleOfXMulYDivZ);
  tuner.addConstraint(kernelId, {"KWG", "MDIMC", "NDIMC", "NDIMB"}, MultipleOfXMulYDivZ);

  // Don't use parameters for polybench reference kernels,
  auto reference = [](const std::vector<size_t>& v) {
    if (v[0] == 2)
      return true;
    else
      return v[1] == 32 && v[2] == 32 && v[3] == 32 && v[4] == 8 && v[5] == 8 && v[6] == 8 &&
             v[7] == 8 && v[8] == 2 && v[9] == 1 && v[10] == 1 && v[11] == 0 && v[12] == 0 &&
             v[13] == 1 && v[14] == 1 && v[15] == 0;
  };
  tuner.addConstraint(kernelId,
      {"KERNEL", "MWG", "NWG", "KWG", "MDIMC", "NDIMC", "MDIMA", "NDIMB", "KWI", "VWM", "VWN",
          "STRM", "STRN", "SA", "SB", "SYMMETRIC"},
      reference);

  // New NVidia GPUs have max. workgroup size
  auto maxWgSize = [](const std::vector<size_t>& v) { return v[0] * v[1] <= MAX_WORK_GROUP_SIZE; };
  tuner.addConstraint(kernelId, {"MDIMC", "NDIMC"}, maxWgSize);

  // Symmetric store can't be used for vectors
  auto symmetric = [](const std::vector<size_t>& v) {
    if (v[0] == 1)
      if (v[1] == 1)
        return v[2] == 1;
      else
        return true;
    else
      return v[1] == 0;
  };
  tuner.addConstraint(kernelId, {"SYMMETRIC", "SYM_STORE", "VWM"}, symmetric);

  // Add all arguments utilized by kernels
  ktt::ArgumentId dataId = tuner.addArgumentVector(data, ktt::ArgumentAccessType::ReadWrite);
  ktt::ArgumentId symmatId = tuner.addArgumentVector(symmat, ktt::ArgumentAccessType::ReadWrite);
  ktt::ArgumentId meanId = tuner.addArgumentVector(mean, ktt::ArgumentAccessType::ReadWrite);
  ktt::ArgumentId mId = tuner.addArgumentScalar(M);
  ktt::ArgumentId nId = tuner.addArgumentScalar(N);
  ktt::ArgumentId floatnId = tuner.addArgumentScalar(float_n);

  // Set kernel arguments for both tuned kernel and reference kernel, order of arguments is
  // important
  tuner.setCompositionKernelArguments(
      kernelId, refMeanKId, std::vector<ktt::ArgumentId>{meanId, dataId, floatnId, mId, nId});
  tuner.setCompositionKernelArguments(
      kernelId, refReduceKId, std::vector<ktt::ArgumentId>{meanId, dataId, mId, nId});
  tuner.setCompositionKernelArguments(
      kernelId, refCovarKId, std::vector<ktt::ArgumentId>{symmatId, dataId, mId, nId});
  tuner.setCompositionKernelArguments(
      kernelId, meanKId, std::vector<ktt::ArgumentId>{meanId, dataId, floatnId, mId, nId});
  tuner.setCompositionKernelArguments(
      kernelId, reduceKid, std::vector<ktt::ArgumentId>{meanId, dataId, mId, nId});
  tuner.setCompositionKernelArguments(
      kernelId, covarKId, std::vector<ktt::ArgumentId>{symmatId, dataId, mId, nId});
  tuner.setCompositionKernelArguments(
      kernelId, gemmKId, std::vector<ktt::ArgumentId>{mId, nId, dataId, symmatId});
  tuner.setCompositionKernelArguments(
      kernelId, triangularToSymmetricKId, std::vector<ktt::ArgumentId>{symmatId, mId});

  // Specify custom tolerance threshold for validation of floating point arguments. Default
  // threshold is 1e-4.
  tuner.setValidationMethod(ktt::ValidationMethod::SideBySideComparison, 128.0);

  // Set tuning manipulator, which implements custom method for launching the kernel
  tuner.setTuningManipulator(
      kernelId, std::make_unique<CovarianceManipulator>(refMeanKId, refReduceKId, refCovarKId,
                    meanKId, reduceKid, covarKId, gemmKId, triangularToSymmetricKId));

  // Set reference kernel which validates results provided by tuned kernel, provide list of
  // arguments which will be validated
  tuner.setReferenceClass(kernelId,
      std::make_unique<CovarianceCpu>(symmatId, meanId, data, symmat, mean, float_n),
      std::vector<ktt::ArgumentId>{symmatId, meanId});

  // Launch kernel tuning
  tuner.tuneKernel(kernelId);

  // Print tuning results to standard output and to output.csv file
  tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
  tuner.printResult(kernelId, "covariance.csv", ktt::PrintFormat::CSV);

  return 0;
}
