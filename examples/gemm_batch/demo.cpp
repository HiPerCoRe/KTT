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
#define TESTS 100
#define MAX_MEM 900000000

class NullBuffer : public std::streambuf
{
public:
    int overflow(int c)
    {
        return c;
    }
};

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
        size_t padd_c = getParameterValue("PADD_C", parameterValues);
        size_t y = getParameterValue("GROUP_SIZE_Y", parameterValues);
        size_t z = getParameterValue("GROUP_SIZE_Z", parameterValues);
/*#if USE_CUDA == 0
        globalSize.setSizeX(batch*(c+padd_c)/z);
        globalSize.setSizeY(y);
        globalSize.setSizeZ(z);
#else*/
        globalSize.setSizeX(batch/z);
//#endif
        localSize.setSizeX(c+padd_c);
        localSize.setSizeY(y);
        localSize.setSizeZ(z);

        runKernel(kernelId, globalSize, localSize);
    }
private:
    int batch, a, b, c;
};

void tuneKernel(ktt::Tuner* tuner, std::string& kernelFile, ktt::ArgumentId& aID, ktt::ArgumentId& bID, ktt::ArgumentId &dstID, ktt::ArgumentId& nID, int a, int b, int c, int batch, int stopBW) {
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
    tuner->addParameter(kernelId, "GROUP_SIZE_Y", {1, 2, 4, 8, 16, 32});
    tuner->addParameter(kernelId, "GROUP_SIZE_Z", {1, 2, 4, 8, 16, 32, 64});
    tuner->addParameter(kernelId, "CACHING_STRATEGY", {0, 1, 2}); /* 0 = implicit caching, 1 = local memory, 2 = private memory */
    tuner->addParameter(kernelId, "PADD_AA", {0, 1});
    tuner->addParameter(kernelId, "PADD_AB", {0, 1});
    if (c % 4 == 0)
        tuner->addParameter(kernelId, "PADD_C", {0});
    else
        tuner->addParameter(kernelId, "PADD_C", {0, static_cast<size_t>(c % 4)});
    tuner->addParameter(kernelId, "DIRECT_WRITE", {0, 1});
    tuner->addParameter(kernelId, "UNROLL_K", {0, 1});

    auto parallelismConstraint = [](const std::vector<size_t>& v) {return v[0] <= v[1];};
    tuner->addConstraint(kernelId, {"GROUP_SIZE_Y", "SIZE_B"}, parallelismConstraint);
    auto paddConstraint = [](const std::vector<size_t>& v) {return (v[0] == 0 && v[1] == 0 && v[2] == 0) || (v[3] > 0);};
    tuner->addConstraint(kernelId, {"PADD_AA", "PADD_AB", "PADD_C", "CACHING_STRATEGY"}, paddConstraint);
    auto dwConstraint = [](const std::vector<size_t>& v) {return (v[0] == 1) || (v[1] > 0);};
    tuner->addConstraint(kernelId, {"DIRECT_WRITE", "CACHING_STRATEGY"}, dwConstraint);
    auto unrollkConstraint = [](const std::vector<size_t>& v) {return (v[0] == 0) || (v[1] == 2);};
    tuner->addConstraint(kernelId, {"UNROLL_K", "CACHING_STRATEGY"}, unrollkConstraint);
#define SHARED_PER_BLOCK (49152/4)
    auto memConstraint = [](const std::vector<size_t>& v) {size_t a = v[1]; size_t b = v[2]; size_t c = v[3]; return (v[0] == 1 && ((a+v[7])*(b+v[8])+c*a+(1-v[4])*(c*b))*v[6] < SHARED_PER_BLOCK) || (v[0] == 2 && v[5] == 1 && ((a+v[7])*(b+v[8])+(1-v[4])*(c*b))*v[6] < SHARED_PER_BLOCK) || (v[0] == 2 && ((a+v[7])*(b+v[8])+c*a+(1-v[4])*(c*b))*v[6] < SHARED_PER_BLOCK);};
    tuner->addConstraint(kernelId, {"CACHING_STRATEGY", "SIZE_A", "SIZE_B", "SIZE_C", "DIRECT_WRITE", "GROUP_SIZE_Y", "GROUP_SIZE_Z", "PADD_AA", "PADD_AB"}, memConstraint);
#define MAX_BLOCK_SIZE 1024
    auto blockConstraint = [](const std::vector<size_t>&v) {return ((v[0]+v[2])*v[1]*v[3] < MAX_BLOCK_SIZE) && ((v[0]+v[2])*v[1]*v[3] >= 32);};
    tuner->addConstraint(kernelId, {"SIZE_C", "GROUP_SIZE_Y", "PADD_C", "GROUP_SIZE_Z"}, blockConstraint);

    // assign manipulator
    tuner->setTuningManipulator(kernelId, std::make_unique<cTunableGemm>(batch, a, b, c));

    tuner->setSearchMethod(ktt::SearchMethod::RandomSearch, std::vector<double>{});

    // tune kernel   
    std::vector<REAL> firstMatrix(32*32);
    ktt::ComputationResult res;
    ktt::OutputDescriptor output(dstID, (void*)firstMatrix.data(), 32*32*sizeof(REAL));
    bool tune = true;
    for (int i = 0; i < STEPS; i++) {
        if (tune)
            res = tuner->tuneKernelByStep(kernelId, {output});
        else {
            ktt::ComputationResult bestConf = tuner->getBestComputationResult(kernelId);
            res = tuner->runKernel(kernelId, bestConf.getConfiguration(), {output});
        }
        clock_t now = clock();
        double overallSec = double(now - beginOverallTime) / CLOCKS_PER_SEC;
        double perfActual = (double)(a*b*c*2)*(double)batch / res.getDuration();
        double effActual = (double)(a*b+c*a+c*b) * (double)batch * (double)sizeof(REAL) / res.getDuration();
        double perfOverall = (double)((i+1)*a*b*c*2)*(double)batch / overallSec / 1000000000.0;
        double bwOverall = (double)(i+1)*(double)((a*b+c*a+c*b)*sizeof(REAL))*(double)batch / overallSec / 1000000000.0;
        std::cout << "Actual perf. " << perfActual << "GFlops, "
            << "actual BW " << effActual << "GB/s, "
            << "perf. with overhead " << perfOverall << "GFlops, " 
            << "BW with overhead " << bwOverall << "GB/s" << std::endl;
        if (effActual > (double)stopBW)
            tune = false;
    }

    // print best
    ktt::ComputationResult bestConf = tuner->getBestComputationResult(kernelId);
    std::cout << "Performance: " << (double)(a*b*c*2)*(double)batch / (double)bestConf.getDuration() << " GFlops" << std::endl;
    std::cout << "Memory BW: " << (double)(a*b+c*a+c*b)*(double)(batch)*(double)sizeof(REAL) / (double)bestConf.getDuration() << " GB/s" << std::endl;
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

    int stopBW = 0;
    std::cout << "Enter bandwidth (GB/s) sufficient to stop tuning ";
    std::cin >> stopBW;

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

    NullBuffer nullBuffer;
    std::ostream nullStream(&nullBuffer);
    tuner->setLoggingTarget(nullStream);

    // tune kernels for different sizes
    for (int i = 0; i < TESTS; i++) {
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

        tuneKernel(tuner, kernelFile, srcAId, srcBId, dstId, nId, a, b, c, batch, stopBW);

        tuner->persistArgument(srcAId, false);
        tuner->persistArgument(srcBId, false);
        tuner->persistArgument(dstId, false);
    }

    delete tuner;

    return 0;
}
