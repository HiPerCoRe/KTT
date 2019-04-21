#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "tuner_api.h"

#if defined(_MSC_VER)
#  define KTT_KERNEL_FILE "../examples/conv_3d/conv_3d.cl"
#  define KTT_REFERENCE_KERNEL_FILE "../examples/conv_3d/conv_3d_reference.cl"
#else
#  define KTT_KERNEL_FILE "../../examples/conv_3d/conv_3d.cl"
#  define KTT_REFERENCE_KERNEL_FILE "../../examples/conv_3d/conv_3d_reference.cl"
#endif

// Problem size
#define WIDTH 256
#define HEIGHT 128
#define DEPTH 128

// Half-filter and filter size - HFS > 1 not supported for Sliding plane kernel
#define HFS 1
#define FS (HFS + HFS + 1)

// New NVidia GPUs have max.workgroup size of 1024
// My Intel(R) HD Graphics Kabylake ULT GT2 has max of 512
#define MAX_WORK_GROUP_SIZE 1024

// Local memory size in bytes
#define MAX_LOCAL_MEM_SIZE 32768

class ConvolutionManipulator : public ktt::TuningManipulator {
 public:
  ConvolutionManipulator(const ktt::KernelId blockedKernelId, const ktt::KernelId referenceKernelId,
      const ktt::KernelId slidingPlaneKernelId)
      : blockedKernelId(blockedKernelId),
        referenceKernelId(referenceKernelId),
        slidingPlaneKernelId(slidingPlaneKernelId) {}

  // LaunchComputation is responsible for actual execution of tuned kernel
  void launchComputation(const ktt::KernelId kernelId) override {
    std::vector<ktt::ParameterPair> parameterValues = getCurrentConfiguration();
    auto algorithm = getParameterValue("ALGORITHM", parameterValues);
    if (algorithm == 0) {
      runKernel(referenceKernelId);
    } else if (algorithm == 1) {
      runKernel(blockedKernelId);
    } else {
      runKernel(slidingPlaneKernelId);
    }
  }

 private:
  ktt::KernelId blockedKernelId;
  ktt::KernelId referenceKernelId;
  ktt::KernelId slidingPlaneKernelId;
};

class ConvolutionCpu : public ktt::ReferenceClass {
 public:
  ConvolutionCpu(const std::vector<float> &src, const std::vector<float> &coeff,
      const std::vector<float> &dest)
      : src(src), coeff(coeff), dest(dest) {}

  // Method inherited from ReferenceClass, which computes reference result for
  // all arguments that are validated inside the class.
  void computeResult() override {
    for (int d = 0; d < DEPTH; d++) {
      for (int h = 0; h < HEIGHT; h++) {
        for (int w = 0; w < WIDTH; w++) {
          float acc = 0.0f;
          for (int k = -HFS; k <= HFS; k++) {
            for (int l = -HFS; l <= HFS; l++) {
              for (int m = -HFS; m <= HFS; m++) {
                acc += coeff[(k + HFS) * FS * FS + (l + HFS) * FS + (m + HFS)] *
                       src[(d + HFS + k) * (WIDTH + 2 * HFS) * (HEIGHT + 2 * HFS) +
                           (h + HFS + l) * (WIDTH + 2 * HFS) + (w + HFS + m)];
              }
            }
          }
          dest[d * WIDTH * HEIGHT + h * WIDTH + w] = acc;
        }
      }
    }
  }

  // Method inherited from ReferenceClass, which returns memory location where
  // reference result for corresponding argument is stored.
  void *getData(const ktt::ArgumentId id) override { return dest.data(); }

 private:
  const std::vector<float> &src;
  const std::vector<float> &coeff;
  std::vector<float> dest;
};

// Helper function to perform an integer division + ceiling (round-up)
size_t CeilDiv(const size_t a, const size_t b) { return (a + b - 1) / b; }

// Helper function to determine whether or not 'a' is a multiple of 'b'
bool IsMultiple(const size_t a, const size_t b) { return (a / b) * b == a; }

int main(int argc, char **argv) {
  // Initialize platform and device index
  ktt::PlatformIndex platformIndex = 0;
  ktt::DeviceIndex deviceIndex = 0;
  std::string kernelFile = KTT_KERNEL_FILE;
  std::string referenceKernelFile = KTT_REFERENCE_KERNEL_FILE;

  if (argc >= 2) {
    platformIndex = std::stoul(std::string(argv[1]));
    if (argc >= 3) {
      deviceIndex = std::stoul(std::string(argv[2]));
      if (argc >= 4) {
        kernelFile = std::string(argv[3]);
        if (argc >= 5) {
          referenceKernelFile = std::string(argv[4]);
        }
      }
    }
  }

  // Initialize data
  std::random_device device;
  std::default_random_engine engine(device());
  std::uniform_real_distribution<float> distribution(0.0f, 3.0f);

  std::vector<float> src((DEPTH + 2 * HFS) * (HEIGHT + 2 * HFS) * (WIDTH + 2 * HFS));
  std::vector<float> dest(DEPTH * HEIGHT * WIDTH, 0.0f);
  std::vector<float> coeff(FS * FS * FS);

  // Initialize source matrix padded by zeros
  for (int d = 0; d < DEPTH + 2 * HFS; d++)
    for (int h = 0; h < HEIGHT + 2 * HFS; h++)
      for (int w = 0; w < WIDTH + 2 * HFS; w++) {
        int index = d * (WIDTH + 2 * HFS) * (HEIGHT + 2 * HFS) + h * (WIDTH + 2 * HFS) + w;
        if (d < HFS || d > DEPTH - 1 + HFS || h < HFS || h > HEIGHT - 1 + HFS || w < HFS ||
            w > WIDTH - 1 + HFS)
          src[index] = 0.0f;
        else
          src[index] = distribution(engine);
      }

  // Creates the filter coefficients (gaussian blur)
  float sigma = 1.0f;
  float sum = 0.0f;
  for (int x = -HFS; x <= HFS; x++)
    for (int y = -HFS; y <= HFS; y++)
      for (int z = -HFS; z <= HFS; z++) {
        float exponent =
            -0.5f * (pow(x / sigma, 2.0f) + pow(y / sigma, 2.0f) + pow(z / sigma, 2.0f));
        float c =
            static_cast<float>(exp(exponent) / (pow(2.0f * 3.14159265f, 1.5f) * pow(sigma, 3.0f)));
        sum += c;
        coeff[(z + HFS) * FS * FS + (y + HFS) * FS + (x + HFS)] = c;
      }
  for (auto &item : coeff) {
    item = item / sum;
  }

  // Create tuner object for chosen platform and device
  ktt::Tuner tuner(platformIndex, deviceIndex);
  tuner.setPrintingTimeUnit(ktt::TimeUnit::Microseconds);

  // Kernel dimensions
  const ktt::DimensionVector ndRangeDimensions(WIDTH, HEIGHT, DEPTH);
  const ktt::DimensionVector workGroupDimensions;

  // Add 3 kernels to the tuner, one of them acts as reference kernel
  ktt::KernelId blockedKernelId =
      tuner.addKernelFromFile(kernelFile, "conv", ndRangeDimensions, workGroupDimensions);
  ktt::KernelId slidingPlaneKernelId =
      tuner.addKernelFromFile(kernelFile, "conv", ndRangeDimensions, workGroupDimensions);
  ktt::KernelId referenceKernelId = tuner.addKernelFromFile(
      referenceKernelFile, "conv_reference", ndRangeDimensions, workGroupDimensions);
  // Add a composition of the kernels, so we can choose which kernel to run in manipulator
  ktt::KernelId kernelId = tuner.addComposition("3D Convolution",
      std::vector<ktt::KernelId>{blockedKernelId, referenceKernelId, slidingPlaneKernelId},
      std::make_unique<ConvolutionManipulator>(
          blockedKernelId, referenceKernelId, slidingPlaneKernelId));

  // Add kernel parameters.
  // ALGORITHM 0 - Reference kernel, 1 - Blocked kernel, 2 - Sliding plane kernel
  tuner.addParameter(kernelId, "ALGORITHM", {0, 1, 2});
  tuner.addParameter(kernelId, "TBX", {8, 16, 32, 64});
  tuner.addParameter(kernelId, "TBY", {8, 16, 32, 64});
  tuner.addParameter(kernelId, "TBZ", {1, 2, 4, 8, 16, 32});
  tuner.addParameter(kernelId, "LOCAL", {0, 1, 2});
  tuner.addParameter(kernelId, "WPTX", {1, 2, 4, 8});
  tuner.addParameter(kernelId, "WPTY", {1, 2, 4, 8});
  tuner.addParameter(kernelId, "WPTZ", {1, 2, 4, 8});
  tuner.addParameter(kernelId, "VECTOR", {1, 2, 4});
  tuner.addParameter(kernelId, "UNROLL_FACTOR", {1, FS});
  tuner.addParameter(kernelId, "CONSTANT_COEFF", {0, 1});
  tuner.addParameter(kernelId, "CACHE_WORK_TO_REGS", {0, 1});
  tuner.addParameter(kernelId, "REVERSE_LOOP_ORDER", {0, 1});
  tuner.addParameter(kernelId, "REVERSE_LOOP_ORDER2", {0, 1});
  tuner.addParameter(kernelId, "REVERSE_LOOP_ORDER3", {0, 1});
  tuner.addParameter(kernelId, "PADDING", {0, 1});
  tuner.addParameter(kernelId, "Z_ITERATIONS", {4, 8, 16, 32});

  // Introduces a helper parameter to compute the proper number of threads for the LOCAL == 2 case.
  // In this case, the workgroup size (TBX by TBY) is extra large (TBX_XL by TBY_XL) because it uses
  // extra (halo) threads only to load the padding to local memory - they don't compute.
  std::vector<size_t> integers{1, 2, 3, 4, 8, 9, 10, 16, 17, 18, 32, 33, 34, 64, 65, 66};

  tuner.addParameter(kernelId, "TBX_XL", integers);
  tuner.addParameter(kernelId, "TBY_XL", integers);
  tuner.addParameter(kernelId, "TBZ_XL", integers);

  // Modify XY NDRange size for all kernels
  auto globalModifier = [](const size_t size, const std::vector<size_t> &v) {
    return (size * v[0] / (v[1] * v[2]));
  };
  tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::X,
      std::vector<std::string>{"TBX_XL", "TBX", "WPTX"}, globalModifier);
  tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::Y,
      std::vector<std::string>{"TBY_XL", "TBY", "WPTY"}, globalModifier);
  // Modify Z NDRange size for Blocked kernel
  tuner.setCompositionKernelThreadModifier(kernelId, blockedKernelId, ktt::ModifierType::Global,
      ktt::ModifierDimension::Z, std::vector<std::string>{"TBZ_XL", "TBZ", "WPTZ"}, globalModifier);
  // Modify Z NDRange size for Sliding plane kernel
  auto globalModifierZ = [](const size_t size, const std::vector<size_t> &v) {
    return (size * v[0] / (v[1] * v[2] * v[3]));
  };
  tuner.setCompositionKernelThreadModifier(kernelId, slidingPlaneKernelId,
      ktt::ModifierType::Global, ktt::ModifierDimension::Z,
      std::vector<std::string>{"TBZ_XL", "TBZ", "WPTZ", "Z_ITERATIONS"}, globalModifierZ);

  // Modify workgroup size for all kernels
  tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::X, "TBX_XL",
      ktt::ModifierAction::Multiply);
  tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "TBY_XL",
      ktt::ModifierAction::Multiply);
  tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::Z, "TBZ_XL",
      ktt::ModifierAction::Multiply);

  // For LOCAL == 2, extend block size by halo threads
  auto HaloThreads = [](const std::vector<size_t> &v) {
    if (v[0] == 2)
      return (v[1] == v[2] + CeilDiv(2 * HFS, v[3]));
    else
      return (v[1] == v[2]);
  };
  tuner.addConstraint(kernelId, {"LOCAL", "TBX_XL", "TBX", "WPTX"}, HaloThreads);
  tuner.addConstraint(kernelId, {"LOCAL", "TBY_XL", "TBY", "WPTY"}, HaloThreads);
  tuner.addConstraint(kernelId, {"LOCAL", "TBZ_XL", "TBZ", "WPTZ"}, HaloThreads);

  // Sets padding to zero in case local memory is not used
  auto padding = [](const std::vector<size_t> &v) { return (v[0] != 0 || v[1] == 0); };
  tuner.addConstraint(kernelId, {"LOCAL", "PADDING"}, padding);

  // GPUs have max. workgroup size
  auto maxWgSize = [](const std::vector<size_t> &v) {
    return v[0] * v[1] * v[2] <= MAX_WORK_GROUP_SIZE;
  };
  tuner.addConstraint(kernelId, {"TBX_XL", "TBY_XL", "TBZ_XL"}, maxWgSize);

  // GPUs have max. local memory size
  auto maxLocalMemSize = [](const std::vector<size_t> &v) {
    size_t haloXY = v[1] == 1 ? 2 * HFS : 0;
    size_t haloZ = v[0] == 2 || v[1] == 1 ? 2 * HFS : 0;
    return v[1] == 0 || (v[3] * v[4] + haloXY + v[2]) * (v[5] * v[6] + haloXY) *
                                (v[7] * v[8] + haloZ) * sizeof(float) <=
                            MAX_LOCAL_MEM_SIZE;
  };
  tuner.addConstraint(kernelId,
      {"ALGORITHM", "LOCAL", "PADDING", "TBX_XL", "WPTX", "TBY_XL", "WPTY", "TBZ_XL", "WPTZ"},
      maxLocalMemSize);

  auto reverseCacheLoopsOrder = [](const std::vector<size_t> &v) { return v[0] == 1 || v[1] == 0; };
  tuner.addConstraint(
      kernelId, {"CACHE_WORK_TO_REGS", "REVERSE_LOOP_ORDER3"}, reverseCacheLoopsOrder);

  // Sets the constrains on the vector size
  auto vectorConstraint = [](const std::vector<size_t> &v) {
    if (v[0] == 2)
      return IsMultiple(v[2], v[1]) && IsMultiple(2 * HFS, v[1]);
    else
      return IsMultiple(v[2], v[1]);
  };
  tuner.addConstraint(kernelId, {"LOCAL", "VECTOR", "WPTX"}, vectorConstraint);

  auto algorithm = [](const std::vector<size_t> &v) {
    // Don't tune any parameters for the reference kernel (ALGORITHM == 0)
    if (v[0] == 0)
      return v[1] == 8 && v[2] == 8 && v[3] == 1 && v[4] == 1 && v[5] == 1 && v[6] == 1 &&
             v[7] == 0 && v[8] == 1 && v[9] == 1 && v[10] == 1 && v[11] == 1 && v[12] == 1 &&
             v[13] == 1 && v[14] == 1;
    // Tune everything for Blocked kernel (ALGORITHM == 1)
    else if (v[0] == 1)
      return true;
    // Set TBZ to 1, WPTZ to 1, and LOCAL to 1/2 for Sliding plane kernel (ALGORITHM == 2)
    else // v[0] == 2
      return (v[3] == 1 && v[6] == 1 && v[7] != 0);
  };
  tuner.addConstraint(kernelId,
      {"ALGORITHM", "TBX", "TBY", "TBZ", "WPTX", "WPTY", "WPTZ", "LOCAL", "VECTOR", "UNROLL_FACTOR",
          "CONSTANT_COEFF", "CACHE_WORK_TO_REGS", "REVERSE_LOOP_ORDER", "REVERSE_LOOP_ORDER2",
          "REVERSE_LOOP_ORDER3"},
      algorithm);

  auto slidingPlane = [](const std::vector<size_t> &v) { return v[0] == 2 || v[1] == 16; };
  tuner.addConstraint(kernelId, {"ALGORITHM", "Z_ITERATIONS"}, slidingPlane);

  // Add all arguments utilized by kernels
  ktt::ArgumentId widthId = tuner.addArgumentScalar(WIDTH);
  ktt::ArgumentId heightId = tuner.addArgumentScalar(HEIGHT);
  ktt::ArgumentId depthId = tuner.addArgumentScalar(DEPTH);
  ktt::ArgumentId srcId = tuner.addArgumentVector(src, ktt::ArgumentAccessType::ReadOnly);
  ktt::ArgumentId coeffId = tuner.addArgumentVector(coeff, ktt::ArgumentAccessType::ReadOnly);
  ktt::ArgumentId destId = tuner.addArgumentVector(dest, ktt::ArgumentAccessType::WriteOnly);

  // Set kernel arguments for both tuned kernel and reference kernel
  tuner.setKernelArguments(
      kernelId, std::vector<ktt::ArgumentId>{widthId, heightId, srcId, coeffId, destId});
  // Rewrite arguments for Sliding plane kernel (needs depth as well)
  tuner.setCompositionKernelArguments(
      kernelId, slidingPlaneKernelId, {widthId, heightId, depthId, srcId, coeffId, destId});

  // Specify custom tolerance threshold for validation of floating point arguments. Default
  // threshold is 1e-4.
  tuner.setValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.001f);

  // Set tuning manipulator
  tuner.setTuningManipulator(kernelId, std::make_unique<ConvolutionManipulator>(blockedKernelId,
                                           referenceKernelId, slidingPlaneKernelId));

  // Set reference kernel which validates results provided by tuned kernel, provide list of
  // arguments which will be validated
  tuner.setReferenceClass(kernelId, std::make_unique<ConvolutionCpu>(src, coeff, dest),
      std::vector<ktt::ArgumentId>{destId});

  // Launch kernel tuning
  tuner.tuneKernel(kernelId);

  // Print tuning results to standard output and to output.csv file
  tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
  tuner.printResult(kernelId, "conv_3d_output.csv", ktt::PrintFormat::CSV);

  return 0;
};
