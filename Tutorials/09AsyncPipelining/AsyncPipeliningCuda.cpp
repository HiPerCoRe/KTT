// KTT tutorial demonstrating asynchronous kernel execution with double-buffered pipelining.
//
// Phase 1 (dynamic tuning): Uses TuneIteration to tune a reduction kernel while producing useful
// results on changing input data. The input vector is updated between iterations and the tuner
// re-uploads it each time.
//
// Phase 2 (async pipelining): After tuning completes, the best configuration is used with RunAsync
// to overlap data transfers and kernel execution on two CUDA streams. Two separate input/output
// buffer pairs (A and B) are alternated in a pipeline:
//   Stream 0:  update(A) -> reduce(A) -> download(resultA)
//   Stream 1:  update(B) -> reduce(B) -> download(resultB)

#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <Ktt.h>

#if defined(_MSC_VER)
const std::string kernelPrefix = "";
#else
const std::string kernelPrefix = "../";
#endif

int main(int argc, char** argv)
{
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = kernelPrefix + "../Tutorials/09AsyncPipelining/Reduction.cu";

    if (argc >= 2)
    {
        deviceIndex = std::stoul(std::string(argv[1]));

        if (argc >= 3)
        {
            kernelFile = std::string(argv[2]);
        }
    }

    // --- Problem setup ---
    const size_t n = 32 * 1024 * 1024;
    const int nInt = static_cast<int>(n);
    const size_t tuningIterations = 20;
    const size_t pipelineIterations = 40;

    // Two input vectors (double-buffering). Using referenced data so KTT sees our modifications.
    std::vector<float> vecA(n);
    std::vector<float> vecB(n);
    float resultA = 0.0f;
    float resultB = 0.0f;

    // Initialize with some data
    for (size_t i = 0; i < n; i++)
    {
        vecA[i] = 1.0f;
        vecB[i] = 2.0f;
    }

    // --- Create tuner with 2 compute queues (streams) ---
    ktt::Tuner tuner(0, deviceIndex, ktt::ComputeApi::CUDA, 2);
    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    // Number of blocks — will be controlled by tuning parameter via thread modifier
    const uint32_t initialBlocks = 256;
    const ktt::DimensionVector gridSize(initialBlocks);
    const ktt::DimensionVector blockSize; // will be set via thread modifier

    // --- Create TWO kernel definitions from the same source and function name ---
    // KTT allows multiple definitions with the same kernel function name. Each definition gets
    // a unique KernelDefinitionId and can have independent argument bindings. The compiled kernel
    // is shared via the kernel cache (same source + same configuration = same compiled code).
    const auto defA = tuner.AddKernelDefinitionFromFile("reduce", kernelFile, gridSize, blockSize);
    const auto defB = tuner.AddKernelDefinitionFromFile("reduce", kernelFile, gridSize, blockSize);

    // Arguments for kernel A: input vector A, output scalar, vector size
    // Using referenceUserData=true so KTT reads current content of vecA on each upload.
    const auto inputAId = tuner.AddArgumentVector(vecA, ktt::ArgumentAccessType::ReadWrite,
        ktt::ArgumentMemoryLocation::Device, ktt::ArgumentManagementType::Framework, true);
    const auto outputAId = tuner.AddArgumentVector(std::vector<float>{0.0f}, ktt::ArgumentAccessType::ReadWrite);
    const auto nAId = tuner.AddArgumentScalar(nInt);
    tuner.SetArguments(defA, {inputAId, outputAId, nAId});

    // Arguments for kernel B: separate buffers
    const auto inputBId = tuner.AddArgumentVector(vecB, ktt::ArgumentAccessType::ReadWrite,
        ktt::ArgumentMemoryLocation::Device, ktt::ArgumentManagementType::Framework, true);
    const auto outputBId = tuner.AddArgumentVector(std::vector<float>{0.0f}, ktt::ArgumentAccessType::ReadWrite);
    const auto nBId = tuner.AddArgumentScalar(nInt);
    tuner.SetArguments(defB, {inputBId, outputBId, nBId});

    // --- Create kernels ---
    const auto kernelA = tuner.CreateSimpleKernel("reduceA", defA);
    const auto kernelB = tuner.CreateSimpleKernel("reduceB", defB);

    // --- Add tuning parameters (same for both kernels) ---
    std::vector<uint64_t> blockSizes = {32, 64, 128, 256, 512};
    std::vector<uint64_t> gridBlocks = {64, 128, 256, 512, 1024};

    tuner.AddParameter(kernelA, "BLOCK_SIZE", blockSizes);
    tuner.AddParameter(kernelA, "GRID_BLOCKS", gridBlocks);
    tuner.AddParameter(kernelB, "BLOCK_SIZE", blockSizes);
    tuner.AddParameter(kernelB, "GRID_BLOCKS", gridBlocks);

    // Local size (block size) is controlled by BLOCK_SIZE parameter
    tuner.AddThreadModifier(kernelA, {defA}, ktt::ModifierType::Local, ktt::ModifierDimension::X,
        {"BLOCK_SIZE"}, [](const uint64_t, const std::vector<uint64_t>& params) -> uint64_t
        {
            return params[0];
        });
    tuner.AddThreadModifier(kernelA, {defA}, ktt::ModifierType::Global, ktt::ModifierDimension::X,
        {"GRID_BLOCKS"}, [](const uint64_t, const std::vector<uint64_t>& params) -> uint64_t
        {
            return params[0];
        });

    tuner.AddThreadModifier(kernelB, {defB}, ktt::ModifierType::Local, ktt::ModifierDimension::X,
        {"BLOCK_SIZE"}, [](const uint64_t, const std::vector<uint64_t>& params) -> uint64_t
        {
            return params[0];
        });
    tuner.AddThreadModifier(kernelB, {defB}, ktt::ModifierType::Global, ktt::ModifierDimension::X,
        {"GRID_BLOCKS"}, [](const uint64_t, const std::vector<uint64_t>& params) -> uint64_t
        {
            return params[0];
        });

    // ============================================================
    // Phase 1: Dynamic tuning with TuneIteration
    // ============================================================
    std::cout << "=== Phase 1: Dynamic tuning ===" << std::endl;

    tuner.SetSearcher(kernelA, std::make_unique<ktt::RandomSearcher>());

    for (size_t iter = 0; iter < tuningIterations; iter++)
    {
        // Fill input with new data each iteration (simulating a real workload)
        const float fillValue = static_cast<float>(iter + 1);

        for (size_t i = 0; i < n; i++)
        {
            vecA[i] = fillValue;
        }

        // Output buffer descriptor: download result after kernel execution
        ktt::BufferOutputDescriptor outputDescr(outputAId, &resultA, sizeof(float));

        // TuneIteration: tries a different configuration each call, uploads vecA (referenced data,
        // ReadWrite -> re-uploaded every time), runs the kernel, downloads the result.
        // recomputeReference=true because our input changes each iteration.
        const auto result = tuner.TuneIteration(kernelA, {outputDescr}, true);

        // Compute expected result for verification
        const float expected = fillValue * n;

        if (result.IsValid())
        {
            const float error = std::fabs(resultA - expected);
            std::cout << "  Iteration " << iter << ": result = " << resultA
                      << " (expected " << expected << ", error " << error << ")"
                      << ", duration = " << result.GetKernelDuration() << " us" << std::endl;
        }
        else
        {
            std::cout << "  Iteration " << iter << ": configuration failed" << std::endl;
        }

        // Reset result for next iteration
        resultA = 0.0f;
    }

    const auto bestConfig = tuner.GetBestConfiguration(kernelA);
    std::cout << "\nBest configuration: " << bestConfig.GetString() << std::endl;

    // ============================================================
    // Phase 2: Async pipelining with double-buffering
    // ============================================================
    std::cout << "\n=== Phase 2: Async pipelining ===" << std::endl;

    // Queue IDs correspond to the 2 streams created in the Tuner constructor.
    // Queue 0 is the default queue.
    const ktt::QueueId queue0 = 0;
    const ktt::QueueId queue1 = 1;

    // Initial upload of all buffers
    tuner.UploadBuffer(inputAId, queue0);
    tuner.UploadBuffer(inputBId, queue1);
    tuner.UploadBuffer(outputAId, queue0);
    tuner.UploadBuffer(outputBId, queue1);

    for (size_t iter = 0; iter < pipelineIterations; iter++)
    {
        const float fillA = static_cast<float>(iter * 2 + 1);
        const float fillB = static_cast<float>(iter * 2 + 2);

        // Update host vectors with new data
        for (size_t i = 0; i < n; i++)
        {
            vecA[i] = fillA;
            vecB[i] = fillB;
        }

        // Zero the output buffers
        resultA = 0.0f;
        resultB = 0.0f;

        // --- Submit work to both streams concurrently ---

        // Stream 0: update input A, zero output A, launch reduce(A)
        tuner.UpdateBufferAsync(inputAId, queue0, vecA.data(), n * sizeof(float));
        tuner.UpdateBufferAsync(outputAId, queue0, &resultA, sizeof(float));
        auto handleA = tuner.RunAsync(kernelA, bestConfig, queue0);

        // Stream 1: update input B, zero output B, launch reduce(B) — concurrent with stream 0
        tuner.UpdateBufferAsync(inputBId, queue1, vecB.data(), n * sizeof(float));
        tuner.UpdateBufferAsync(outputBId, queue1, &resultB, sizeof(float));
        auto handleB = tuner.RunAsync(kernelB, bestConfig, queue1);

        // --- Wait for results and download ---

        // Wait for reduce(A), then enqueue download on the same stream and wait for it
        tuner.WaitForRun(handleA);
        auto dlA = tuner.DownloadBufferAsync(outputAId, queue0, &resultA, sizeof(float));
        tuner.WaitForTransferAction(dlA);

        // Wait for reduce(B), then enqueue download on the same stream and wait for it
        tuner.WaitForRun(handleB);
        auto dlB = tuner.DownloadBufferAsync(outputBId, queue1, &resultB, sizeof(float));
        tuner.WaitForTransferAction(dlB);

        // Verify
        const float expectedA = fillA * n;
        const float expectedB = fillB * n;

        std::cout << "  Iteration " << iter
                  << ": A = " << resultA << " (expected " << expectedA
                  << ", error " << std::fabs(resultA - expectedA) << ")"
                  << ", B = " << resultB << " (expected " << expectedB
                  << ", error " << std::fabs(resultB - expectedB) << ")"
                  << std::endl;
    }

    std::cout << "\nDone." << std::endl;
    return 0;
}
