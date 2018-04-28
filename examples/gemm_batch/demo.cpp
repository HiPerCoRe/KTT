#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include "tuner_api.h"

#if defined(_MSC_VER)
    #define KTT_CL_KERNEL_FILE "../examples/gemm_batch/gemm_kernel.cl"
#else
    #define KTT_CL_KERNEL_FILE "../../examples/gemm_batch/gemm_kernel.cl"
#endif
    #if defined(_MSC_VER)
        #define KTT_CU_KERNEL_FILE "../examples/gemm_batch/gemm_kernel.cu"
    #else
        #define KTT_CU_KERNEL_FILE "../../examples/gemm_batch/gemm_kernel.cu"
    #endif

#define REAL float
#define STEPS 1000
#define MAX_MEM 900000000

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
        size_t gran = getParameterValue("GRANULARITY", parameterValues);
        size_t myBatch = batch / getParameterValue("MGCG_GROUP_SIZE_Y", parameterValues); 
        std::cout << "batch: " << batch << std::endl;
        if (gran == 1) {
            globalSize.setSizeX(myBatch/getParameterValue("GROUP_SIZE_X", parameterValues));
            localSize.setSizeX(getParameterValue("GROUP_SIZE_X", parameterValues));
        }
        if (gran == 2) {
            size_t y = getParameterValue("MGCG_GROUP_SIZE_Y", parameterValues);
            globalSize.setSizeX(batch / y);
            localSize.setSizeX(c);
            localSize.setSizeY(y);
        }
        if (gran == 3) {
            size_t y = getParameterValue("MGCG_GROUP_SIZE_Y", parameterValues);
            globalSize.setSizeX(batch);
            localSize.setSizeX(c);
            localSize.setSizeY(y);
        }

        runKernel(kernelId, globalSize, localSize);
    }
private:
    int batch, a, b, c;
};

void tuneKernel(ktt::Tuner* tuner, std::string& kernelFile, ktt::ArgumentId& aID, ktt::ArgumentId& bID, ktt::ArgumentId &dstID, ktt::ArgumentId& nID, int a, int b, int c, int batch) {
    clock_t beginOverallTime = clock();

    // create kernel
    ktt::DimensionVector ndRangeDimensions(batch);
    ktt::DimensionVector workGroupDimensions;
    ktt::KernelId kernelId = tuner->addKernelFromFile(kernelFile, "gemm_batch", ndRangeDimensions, workGroupDimensions);

    // assign arguments
    tuner->setKernelArguments(kernelId, std::vector<ktt::ArgumentId>{aID, bID, dstID, nID});

    // create tuning space
    tuner->addParameter(kernelId, "SIZE_A", {(size_t)a});
    tuner->addParameter(kernelId, "SIZE_B", {(size_t)b});
    tuner->addParameter(kernelId, "SIZE_C", {(size_t)c});
    tuner->addParameter(kernelId, "GRANULARITY", {/*1, */2, 3}); // 1 = fine (matrix per thread), 2 = medium (block of a), 3 = coarse (block of a*b)
    tuner->addParameter(kernelId, "GROUP_SIZE_X", {1, 32, 64, 128, 256, 512});
    tuner->addParameter(kernelId, "MGCG_GROUP_SIZE_X", {1, (size_t)c});
    tuner->addParameter(kernelId, "MGCG_GROUP_SIZE_Y", {1, 2, 4, 8, 16, 32});
    tuner->addParameter(kernelId, "CACHING_STRATEGY", {0, 1, 2}); /* 0 = implicit caching, 1 = local memory, 2 = private memory */
    auto parallelismConstraint = [](const std::vector<size_t>& v) {return (v[0] == 1 && v[1] > 1 && v[2] == 1 && v[3] == 1) || (v[0] == 2 && v[1] == 1 && v[2] > 1) || (v[0] == 3 && v[1] == 1 && v[2] > 1);};
    tuner->addConstraint(kernelId, parallelismConstraint, {"GRANULARITY", "GROUP_SIZE_X", "MGCG_GROUP_SIZE_X", "MGCG_GROUP_SIZE_Y"});
    auto tmpConstraint = [](const std::vector<size_t>& v) {return (v[0] < 3 || v[1] < 2);};
    tuner->addConstraint(kernelId, tmpConstraint, {"GRANULARITY", "CACHING_STRATEGY"});
    auto smConstraint = [](const std::vector<size_t>& v) {return (v[0] != 2) || (v[1]*v[2] + v[3]*v[1] + v[3]*v[2])*v[4]*sizeof(REAL) < 48*1024;};
    tuner->addConstraint(kernelId, smConstraint, {"GRANULARITY", "SIZE_A", "SIZE_B", "SIZE_C", "MGCG_GROUP_SIZE_Y"});

    // assign manipulator
    tuner->setTuningManipulator(kernelId, std::make_unique<cTunableGemm>(batch, a, b, c));

    // tune kernel   
    std::vector<REAL> firstMatrix(32*32);
    ktt::OutputDescriptor output(dstID, (void*)firstMatrix.data(), 32*32*sizeof(REAL));
    for (int i = 0; i < STEPS; i++) {
        tuner->tuneKernelByStep(kernelId, {output});
        clock_t now = clock();
        double overallSec = double(now - beginOverallTime) / CLOCKS_PER_SEC;
        double perfActual = 0.0;
        double effActual = 0.0;
        double perfOverall = (double)((i+1)*a*b*c*2)*(double)batch / overallSec / 1000000000.0;
        std::cout << overallSec << std::endl;
        std::cout << "Actual perf. " << perfActual << "GFlops, "
            << "actual BW " << effActual << "GB/s, "
            << "perf. with overhead " << perfOverall << "GFlops" << std::endl;
    }

    // print best
    std::pair<std::vector<ktt::ParameterPair>, double> bestConf = tuner->getBestConfiguration(kernelId);
    std::cout << "Performance: " << (double)(a*b*c*2)*(double)batch / std::get<1>(bestConf) << " GFlops" << std::endl;
    std::cout << "Memory BW: " << (double)(a*b+c*a+c*b)*(double)(batch)*(double)sizeof(REAL) / std::get<1>(bestConf) << " GB/s" << std::endl;
}

int main(int argc, char** argv)
{
    // Initialize platform and device index
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile;

    if (argc >= 2)
    {
        platformIndex = std::stoul(std::string(argv[1]));
        if (argc >= 3)
        {
            deviceIndex = std::stoul(std::string(argv[2]));
        }
    }

    int interface = 0;
    std::cout << "Which interface to use (0 = CUDA, 1 = OpenCL)? ";
    std::cin >> interface;

    // create and configure tunner
    ktt::Tuner* tuner = NULL; 
    if (interface == 1) {
        tuner = new ktt::Tuner(platformIndex, deviceIndex);
        kernelFile = KTT_CL_KERNEL_FILE;
    }
    else {
        tuner = new ktt::Tuner(0, deviceIndex, ktt::ComputeAPI::CUDA);
        kernelFile = KTT_CU_KERNEL_FILE;
    }
    tuner->setGlobalSizeType(ktt::GlobalSizeType::CUDA);
    tuner->setLoggingTarget(std::string("/dev/null"));

    // tune kernels for different sizes
    for (int i = 0; i < 10; i++) {
        // generate input size
        int a = 2+(long long)(rand())*31 / RAND_MAX;
        int b = 2+(long long)(rand())*31 / RAND_MAX;
        int c = 2+(long long)(rand())*31 / RAND_MAX;
        int batch = ((MAX_MEM/(sizeof(REAL)*(a*b+c*a+c*b)))/512)*512;

        std::cout << "Tuning for matrices of size "
            << a << "x" << b << ", "
            << c << "x" << a << ", "
            << c << "x" << b << ", "
            << "batch size " << batch << std::endl;

        // create data in host memory
        std::vector<REAL> srcA(a*b*batch, 0.0f);
        std::vector<REAL> srcB(c*a*batch, 0.0f);
        std::vector<REAL> dst(c*b*batch, 0.0f);

        // fill with random values
        for (size_t i = 0; i < a*b*batch; i++)
            srcA[i] = 10.0f*((REAL)rand()) / ((REAL) RAND_MAX);
        for (size_t i = 0; i < c*a*batch; i++)
            srcB[i] = 10.0f*((REAL)rand()) / ((REAL) RAND_MAX);

        // create input/output
        ktt::ArgumentId srcAId = tuner->addArgumentVector(srcA, ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device, false);
        tuner->persistArgument(srcAId, true);
        ktt::ArgumentId srcBId = tuner->addArgumentVector(srcB, ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device, false);
        tuner->persistArgument(srcBId, true);
        ktt::ArgumentId dstId = tuner->addArgumentVector(dst, ktt::ArgumentAccessType::WriteOnly, ktt::ArgumentMemoryLocation::Device, false);
        tuner->persistArgument(dstId, true);
        ktt::ArgumentId nId = tuner->addArgumentScalar(batch);

        tuneKernel(tuner, kernelFile, srcAId, srcBId, dstId, nId, a, b, c, batch);

        tuner->persistArgument(srcAId, false);
        tuner->persistArgument(srcBId, false);
        tuner->persistArgument(dstId, false);
    }

    delete tuner;

    return 0;
}
