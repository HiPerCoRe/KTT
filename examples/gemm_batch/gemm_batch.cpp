#include <iostream>
#include <string>
#include <vector>
#include "tuner_api.h"

#define STRIDED 0

#if STRIDED == 1
    #define STRIDE_BLOCK 32
#endif

#define USE_CUDA 0

#if STRIDED == 0
    #if USE_CUDA == 0
        #if defined(_MSC_VER)
            #define KTT_KERNEL_FILE "../examples/gemm_batch/gemm_kernel.cl"
        #else
            #define KTT_KERNEL_FILE "../../examples/gemm_batch/gemm_kernel.cl"
        #endif
    #else /* USE_CUDA */
        #if defined(_MSC_VER)
            #define KTT_KERNEL_FILE "../examples/gemm_batch/gemm_kernel.cu"
        #else
            #define KTT_KERNEL_FILE "../../examples/gemm_batch/gemm_kernel.cu"
        #endif
    #endif /* USE_CUDA */
#else /* STRIDED */
    #if USE_CUDA == 0
        #if defined(_MSC_VER)
            #define KTT_KERNEL_FILE "../examples/gemm_batch/gemm_kernel_strided.cl"
        #else
            #define KTT_KERNEL_FILE "../../examples/gemm_batch/gemm_kernel_strided.cl"
        #endif
    #else /* USE_CUDA */
        #if defined(_MSC_VER)
            #define KTT_KERNEL_FILE "../examples/gemm_batch/gemm_kernel_strided.cu"
        #else
            #define KTT_KERNEL_FILE "../../examples/gemm_batch/gemm_kernel_strided.cu"
        #endif
    #endif /* USE_CUDA */
#endif /* STRIDED */

#define REAL float

class referenceGemm : public ktt::ReferenceClass
{
public:
    referenceGemm(const std::vector<REAL>& srcA, const std::vector<REAL>& srcB, const int a, const int b, const int c, const int n, const ktt::ArgumentId resultArgumentId) :
        srcA(srcA),
	    srcB(srcB),
    	a(a),
    	b(b),
	    c(c),
    	n(n),
        resultArgumentId(resultArgumentId)
    {}

    void computeResult() override {
        res.resize(n*c*b);
#if STRIDED == 0
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < c; j++)
                for (int k = 0; k < b; k++) {
                    REAL tmp = 0.0f;
                    for (int l = 0; l < a; l++)
                        tmp += srcA[i*a*b + k*a + l] * srcB[i*c*a + l*c + j];
                    res[i*c*b + k*c + j] = tmp;
                }
            /*if (i == 0) {
                for (int bl = 0; bl < b; bl++){
                    for (int al = 0; al < a; al++)
                        std::cout << srcA[i*a*b + bl*a + al] << " ";
                    std::cout << "\n";
                }
                for (int al = 0; al < a; al++){
                    for (int cl = 0; cl < c; cl++)
                        std::cout << srcB[i*c*b + al*c + cl] << " ";
                    std::cout << "\n";
                }
                for (int bl = 0; bl < b; bl++){
                    for (int cl = 0; cl < c; cl++)
                        std::cout << res[i*c*b + bl*c + cl] << " ";
                    std::cout << "\n";
                }
            }*/
	    }
#else
        for (int i = 0; i < n; i++) {
            int startBlock = (i / STRIDE_BLOCK) * STRIDE_BLOCK;
            int startLocal = i % STRIDE_BLOCK;
            for (int j = 0; j < c; j++)
                for (int k = 0; k < b; k++) {
                    REAL tmp = 0.0f;
                    for (int l = 0; l < a; l++)
                        tmp += srcA[startBlock*a*b + (k*a + l)*STRIDE_BLOCK + startLocal] * srcB[startBlock*c*a + (l*c + j)*STRIDE_BLOCK + startLocal];
                    res[startBlock*c*b + (k*c + j)*STRIDE_BLOCK + startLocal] = tmp;
                }
            if (i == 1) {
                for (int bl = 0; bl < b; bl++){
                    for (int al = 0; al < a; al++)
                        std::cout << srcA[startBlock*a*b + (bl*a + al)*STRIDE_BLOCK + startLocal] << " ";
                    std::cout << "\n";
                }
                for (int al = 0; al < a; al++){
                    for (int cl = 0; cl < c; cl++)
                        std::cout << srcB[startBlock*c*b + (al*c + cl)*STRIDE_BLOCK + startLocal] << " ";
                    std::cout << "\n";
                }
                for (int bl = 0; bl < b; bl++){
                    for (int cl = 0; cl < c; cl++)
                        std::cout << res[startBlock*c*b + (bl*c + cl)*STRIDE_BLOCK + startLocal] << " ";
                    std::cout << "\n";
                }
            }
        }
#endif
    }

    void* getData(const ktt::ArgumentId id) override {
        if (id == resultArgumentId) {
            return (void*)res.data();
        }
        return nullptr;
    }

    size_t getNumberOfElements(const ktt::ArgumentId argumentId) const override {
        return n*c*b;
    }

private:
    std::vector<REAL> res;
    std::vector<REAL> srcA;
    std::vector<REAL> srcB;
    int a, b, c, n;
    ktt::ArgumentId resultArgumentId;
};

class cTunableGemm : public ktt::TuningManipulator {
public:
    cTunableGemm(const int batch, const int a, const int b, const int c)  :
        batch(batch),
        a(a),
        b(b),
        c(c)
    {}
    void launchComputation(const ktt::KernelId kernelId) override {
        ktt::DimensionVector globalSize(1, 1, 1);
        ktt::DimensionVector localSize(1, 1, 1);
        std::vector<ktt::ParameterPair> parameterValues = getCurrentConfiguration();
        int gran = getParameterValue("GRANULARITY", parameterValues);
        int myBatch = batch / getParameterValue("MGCG_GROUP_SIZE_Y", parameterValues); // is always 1 when STRIDED == 0
        if (gran == 1) {
#if USE_CUDA == 0
            globalSize.setSizeX(myBatch);
    #if STRIDED == 1
            globalSize.setSizeY(getParameterValue("MGCG_GROUP_SIZE_Y", parameterValues));
    #endif
#else /* USE_CUDA */
    #if STRIDED == 0
            globalSize.setSizeX(myBatch/getParameterValue("GROUP_SIZE_X", parameterValues));
    #else
            globalSize.setSizeX(myBatch/STRIDE_BLOCK);
    #endif
#endif /* USE_CUDA */
#if STRIDED == 0
            localSize.setSizeX(getParameterValue("GROUP_SIZE_X", parameterValues));
#else
            localSize.setSizeX(STRIDE_BLOCK);
            localSize.setSizeY(getParameterValue("MGCG_GROUP_SIZE_Y", parameterValues));
#endif
        }
        if (gran == 2) {
            size_t y = getParameterValue("MGCG_GROUP_SIZE_Y", parameterValues);
#if USE_CUDA == 0
            globalSize.setSizeX(batch*c / y);
            globalSize.setSizeY(y);
#else
            globalSize.setSizeX(batch / y);
#endif
            localSize.setSizeX(c);
            localSize.setSizeY(y);
        }
        if (gran == 3) {
            size_t y = getParameterValue("MGCG_GROUP_SIZE_Y", parameterValues);
#if USE_CUDA == 0
            globalSize.setSizeX(batch*c);
            globalSize.setSizeY(y);
#else
            globalSize.setSizeX(batch);
#endif
            localSize.setSizeX(c);
            localSize.setSizeY(y);
        }

        runKernel(kernelId, globalSize, localSize);
    }
private:
    int batch, a, b, c;
};

int main(int argc, char** argv)
{
    // Initialize platform and device index
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = KTT_KERNEL_FILE;

    if (argc >= 2)
    {
        platformIndex = std::stoul(std::string(argv[1]));
        if (argc >= 3)
        {
            deviceIndex = std::stoul(std::string(argv[2]));
        }
    }

    // Declare and initialize data (m, n > 1)
    unsigned int a = 4;
    unsigned int b = 4;
    unsigned int c = 4;
    int batch = 1024*64;

    if (argc >= 7) {
        a = std::stoul(std::string(argv[3]));
        b = std::stoul(std::string(argv[4]));
        c = std::stoul(std::string(argv[5]));
        batch = std::stoul(std::string(argv[6]));
    }

    std::cout << "Computing C = AB using " << batch << " matrices of sizes"
        << std::endl
        << "A: " << a << " x " << b << std::endl
        << "B: " << c << " x " << a << std::endl
        << "C: " << c << " x " << b << std::endl;

    std::vector<REAL> srcA(a*b*batch, 0.0f);
    std::vector<REAL> srcB(c*a*batch, 0.0f);
    std::vector<REAL> dst(c*b*batch, 0.0f);

    // fill with random data
    for (size_t i = 0; i < a*b*batch; i++)
    {
        srcA[i] = 10.0f*((REAL)rand()) / ((REAL) RAND_MAX);
    }
    for (size_t i = 0; i < c*a*batch; i++)
    {
        srcB[i] = 10.0f*((REAL)rand()) / ((REAL) RAND_MAX);
    }
/*    // fill transposed B
    for (int i = 0; i < batch; i++)
        for (int j = 0; j < c; j++)
            for (int k = 0; k < a; k++)
            {
                srcBT[i*a*c + j*a + k] = srcB[i*a*c + k*c + j];
            }*/

#if USE_CUDA == 0
    ktt::Tuner tuner(platformIndex, deviceIndex);
#else
    ktt::Tuner tuner(0, deviceIndex, ktt::ComputeAPI::CUDA);
#endif

    // create kernel
    ktt::DimensionVector ndRangeDimensions(batch);
    ktt::DimensionVector workGroupDimensions;
    ktt::KernelId kernelId = tuner.addKernelFromFile(kernelFile, "gemm_batch", ndRangeDimensions, workGroupDimensions);

    // create input/output
    ktt::ArgumentId srcAId = tuner.addArgumentVector(srcA, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId srcBId = tuner.addArgumentVector(srcB, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId dstId = tuner.addArgumentVector(dst, ktt::ArgumentAccessType::WriteOnly);
    ktt::ArgumentId nId = tuner.addArgumentScalar(batch);
    tuner.setKernelArguments(kernelId, std::vector<ktt::ArgumentId>{srcAId, srcBId, dstId, nId});

    tuner.addParameter(kernelId, "SIZE_A", {(size_t)a});
    tuner.addParameter(kernelId, "SIZE_B", {(size_t)b});
    tuner.addParameter(kernelId, "SIZE_C", {(size_t)c});
#if STRIDED == 1
    tuner.addParameter(kernelId, "STRIDE_BLOCK", {(size_t)STRIDE_BLOCK});
#endif
#if STRIDED == 0
    tuner.addParameter(kernelId, "GRANULARITY", {1, 2, 3}); // 1 = fine (matrix per thread), 2 = medium (block of a), 3 = coarse (block of a*b)
#else
    tuner.addParameter(kernelId, "GRANULARITY", {1}); // other granularities not supported
#endif
#if STRIDED == 0
    tuner.addParameter(kernelId, "GROUP_SIZE_X", {1, 32, 64, 128, 256, 512});
#else
    tuner.addParameter(kernelId, "GROUP_SIZE_X", {1, STRIDE_BLOCK});
#endif
    tuner.addParameter(kernelId, "MGCG_GROUP_SIZE_X", {1, (size_t)c});
    tuner.addParameter(kernelId, "MGCG_GROUP_SIZE_Y", {1, 2, 4, 8, 16, 32});
    tuner.addParameter(kernelId, "CACHING_STRATEGY", {0, 1, 2}); /* 0 = implicit caching, 1 = local memory, 2 = private memory */

#if STRIDED == 0
    auto parallelismConstraint = [](std::vector<size_t> v) {return (v[0] == 1 && v[1] > 1 && v[2] == 1 && v[3] == 1) || (v[0] == 2 && v[1] == 1 && v[2] > 1) || (v[0] == 3 && v[1] == 1 && v[2] > 1);};
#else
    auto parallelismConstraint = [](std::vector<size_t> v) {return (v[0] == 1 && v[1] > 1 && v[2] == 1) || (v[0] == 2 && v[1] == 1 && v[2] > 1) || (v[0] == 3 && v[1] == 1 && v[2] > 1);};
#endif
    tuner.addConstraint(kernelId, parallelismConstraint, {"GRANULARITY", "GROUP_SIZE_X", "MGCG_GROUP_SIZE_X", "MGCG_GROUP_SIZE_Y"});
    auto tmpConstraint = [](std::vector<size_t> v) {return (v[0] < 3 || v[1] < 2);};
    tuner.addConstraint(kernelId, tmpConstraint, {"GRANULARITY", "CACHING_STRATEGY"});

    tuner.setReferenceClass(kernelId, std::make_unique<referenceGemm>(srcA, srcB, a, b, c, batch, dstId), std::vector<ktt::ArgumentId>{dstId});
    tuner.setValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.001f);
    tuner.setValidationRange(dstId, c*b*batch);

    tuner.setTuningManipulator(kernelId, std::make_unique<cTunableGemm>(batch, a, b, c));
    
    tuner.tuneKernel(kernelId);
    tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
    tuner.printResult(kernelId, "gemm_batch_output.csv", ktt::PrintFormat::CSV);

    std::pair<std::vector<ktt::ParameterPair>, double> bestConf = tuner.getBestConfiguration(kernelId);

    std::cout << "Performance: " << (double)(a*b*c*2)*(double)batch / std::get<1>(bestConf) << " GFlops" << std::endl;
    std::cout << "Memory BW: " << (double)(a*b+c*a+c*b)*(double)(batch)*(double)sizeof(REAL) / std::get<1>(bestConf) << " GB/s" << std::endl;

    return 0;
}
