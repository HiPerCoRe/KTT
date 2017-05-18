#include <iostream>
#include <string>
#include <vector>
#include <CL/opencl.h>

#include "../../include/ktt.h"

class referenceReduction : public ktt::ReferenceClass
{
    std::vector<float> res;
    std::vector<float> src;
    size_t resultArgumentId;
public:
    referenceReduction(const std::vector<float>& src, const size_t resultArgumentId) :
        src(src),
        resultArgumentId(resultArgumentId)
    {}

    // High precision of reduction
    virtual void computeResult() override {
        std::vector<double> resD(src.size());
        size_t resSize = src.size();
        for (int i = 0; i < resSize; i++)
            resD[i] = src[i];

        while (resSize > 1) {
            for (int i = 0; i < resSize/2; i++)
                resD[i] = resD[i*2] + resD[i*2+1];
            if (resSize%2) resD[resSize/2-1] += resD[resSize-1];
            resSize = resSize/2;
        }
        res.clear();
        res.push_back((float)resD[0]);
    }

    virtual const void* getData(const size_t argumentId) const override {
        if (argumentId == resultArgumentId) {
            return (void*)res.data();
        }
        throw std::runtime_error("No result available for specified argument id");
    }

    virtual ktt::ArgumentDataType getDataType(const size_t argumentId) const override {
        return ktt::ArgumentDataType::Float;
    }

    virtual size_t getNumberOfElements(const size_t argumentId) const override {
        return 1;
}

    virtual size_t getElementSizeInBytes(const size_t argumentId) const override {
        return sizeof(float);
    }
};

class tunableReduction : public ktt::TuningManipulator {
    ktt::Tuner *tuner;
    int n;
    std::vector<float> *src;
    std::vector<float> *dst;
    size_t srcId;
    size_t dstId;
    size_t nId;
    size_t inOffsetId;
    size_t outOffsetId;
    size_t kernelId;
public:
    tunableReduction(ktt::Tuner *tuner, std::vector<float> *src, std::vector<float> *dst, int n) : TuningManipulator() {
        this->tuner = tuner;

        // input is set in constructor in this example
        this->n = n;
        this->src = src;
        this->dst = dst;

        // create kernel
        int nUp = ((n+512-1)/512)*512; // maximal WG size used in tuning parameters
        ktt::DimensionVector ndRangeDimensions(nUp, 1, 1);
        ktt::DimensionVector workGroupDimensions(1, 1, 1);
        kernelId = tuner->addKernelFromFile("../examples/reduction/reduction_kernel.cl", std::string("reduce"), ndRangeDimensions, workGroupDimensions);

        // create input/output
        srcId = tuner->addArgument((void*)src->data(), n, ktt::ArgumentDataType::Float, ktt::ArgumentMemoryType::ReadOnly);
        dstId = tuner->addArgument((void*)dst->data(), n, ktt::ArgumentDataType::Float, ktt::ArgumentMemoryType::ReadWrite);
        nId = tuner->addArgument((void*)(&n), ktt::ArgumentDataType::Int);
        int offset = 0;
        inOffsetId = tuner->addArgument((void*)(&offset), ktt::ArgumentDataType::Int);
        outOffsetId = tuner->addArgument((void*)(&offset), ktt::ArgumentDataType::Int);
        tuner->setKernelArguments(kernelId, std::vector<size_t>{ srcId, dstId, nId, inOffsetId, outOffsetId } );

        // get number of compute units
        //TODO refactor to use KTT functions
        size_t cus = 30;

        // create parameter space
        tuner->addParameter(kernelId, "WORK_GROUP_SIZE_X", { /*1, 2, 4, 8, 16, 32, 64, */128/*, 256, 512 */},
            ktt::ThreadModifierType::Local, 
            ktt::ThreadModifierAction::Multiply, 
            ktt::Dimension::X);
        tuner->addParameter(kernelId, "UNBOUNDED_WG", { /*0,*/ 1 });
        tuner->addParameter(kernelId, "WG_NUM", { 0, cus, cus * 2, cus * 4, cus * 8, cus * 16 });
        tuner->addParameter(kernelId, "VECTOR_SIZE", { 1/*, 2, 4, 8, 16*/ },
            ktt::ThreadModifierType::Global,
            ktt::ThreadModifierAction::Divide,
            ktt::Dimension::X);
        tuner->addParameter(kernelId, "USE_ATOMICS", { /*0, */1 });
        auto persistConstraint = [](std::vector<size_t> v) { return (v[0] && v[1] == 0) || (!v[0] && v[1] > 0); };
        tuner->addConstraint(kernelId, persistConstraint, { "UNBOUNDED_WG", "WG_NUM" });
        auto persistentAtomic = [](std::vector<size_t> v) { return (v[0] == 1) || (v[0] == 0 && v[1] == 1); };
        tuner->addConstraint(kernelId, persistentAtomic, { "UNBOUNDED_WG", "USE_ATOMICS" } );

        tuner->setReferenceClass(kernelId, std::make_unique<referenceReduction>(*src, dstId), std::vector<size_t>{ dstId });
        tuner->setValidationMethod(ktt::ValidationMethod::SideBySideComparison, (float)n/100000.0f, 1);

        // set itself as a tuning manipulator
        tuner->setTuningManipulator(kernelId, std::unique_ptr<TuningManipulator>(this));
    }

    size_t getParameterValue(const std::vector<ktt::ParameterValue>& parameterValue, const std::string& name){
        for (auto parIt : parameterValue)
            if (std::get<0>(parIt) == name)
                return std::get<1>(parIt);

        return 0;
    }

    virtual void launchComputation(const size_t kernelId, const ktt::DimensionVector& globalSize, const ktt::DimensionVector& localSize, const std::vector<ktt::ParameterValue>& parameterValues) override {
        ktt::DimensionVector myGlobalSize = globalSize;
        // change global size for constant numners of work-groups
        //XXX this may be done also by thread modifier operators in constructor
        if (getParameterValue(parameterValues, std::string("UNBOUNDED_WG")) == 0) {
            myGlobalSize = std::make_tuple(
                getParameterValue(parameterValues, std::string("WG_NUM")) 
                * std::get<0>(localSize), 1, 1);
        }

        // execute reduction kernel
        runKernel(kernelId, myGlobalSize, localSize);

        // execute kernel log n times, when atomics are not used 
        if (getParameterValue(parameterValues, std::string("USE_ATOMICS")) == 0) {
            int n = std::max(std::get<0>(globalSize) / std::get<0>(localSize),
                std::get<0>(localSize));
            int inOffset = 0;
            int outOffset = n;
            int vectorSize = getParameterValue(parameterValues, std::string("VECTOR_SIZE"));
            
            // output array contains input now
            tuner->setKernelArguments(kernelId, std::vector<size_t>{ dstId, dstId, nId, inOffsetId, outOffsetId } );

            /*while (n > 1) {
                if (std::get<0>(globalSize) == std::get<0>(localSize))
                    outOffset = 0; // only one WG will be executed
                if (getParameterValue(parameterValues, std::string("UNBOUNDED_WG")) == 1)
                    std::get<0>(myGlobalSize) = (n+vectorSize-1) / vectorSize;
                updateArgumentScalar(nId, &n);
                updateArgumentScalar(outOffsetId, &outOffset);
                updateArgumentScalar(inOffsetId, &inOffset);
                std::cout << "XXX " << n << " " << inOffset << " " 
                    << outOffset << "\n";
                std::cout << "YYY " << std::get<0>(myGlobalSize) << " "
                    << std::get<0>(localSize) << "\n";
                runKernel(kernelId, myGlobalSize, localSize);
                n = (n+std::get<0>(localSize)-1)/std::get<0>(localSize);
                inOffset = outOffset;
                outOffset += n;
            }*/
            runKernel(kernelId, myGlobalSize, localSize);

            // reset kernel arguments
            tuner->setKernelArguments(kernelId, std::vector<size_t>{ srcId, dstId, nId, inOffsetId, outOffsetId } );
        }
    }

    void tune() {
        tuner->tuneKernel(kernelId);
        tuner->printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
    }
};

int main(int argc, char** argv)
{
    // Initialize platform and device index
    size_t platformIndex = 0;
    size_t deviceIndex = 0;

    if (argc >= 2)
    {
        platformIndex = std::stoul(std::string{ argv[1] });
        if (argc >= 3)
        {
            deviceIndex = std::stoul(std::string{ argv[2] });
        }
    }

    // Declare and initialize data
    const int n = 1025/**1024*32*/;
    std::vector<float> src(n);
    std::vector<float> dst(n);
    for (int i = 0; i < n; i++)
    {
        src[i] = 1.0f;//(float) rand() / (float) RAND_MAX + 1.0f;
        dst[i] = 0.0f;
    }

    ktt::Tuner tuner(platformIndex, deviceIndex);

    tunableReduction reduction(&tuner, &src, &dst, n);

    reduction.tune();

    return 0;
}
