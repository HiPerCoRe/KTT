#pragma once

#include "tuner_api.h"

#include "reduction_reference.h"

class tunableReduction : public ktt::TuningManipulator {
public:

/* 
    Constructor creates internal structures and setups tuning environment */
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
        srcId = tuner->addArgument(*src, ktt::ArgumentMemoryType::ReadWrite);
        dstId = tuner->addArgument(*dst, ktt::ArgumentMemoryType::ReadWrite);
        nId = tuner->addArgument(n);
        int offset = 0;
        inOffsetId = tuner->addArgument(offset);
        outOffsetId = tuner->addArgument(offset);
        tuner->setKernelArguments(kernelId, std::vector<size_t>{ srcId, dstId, nId, inOffsetId, outOffsetId } );

        // get number of compute units
        const ktt::DeviceInfo di = tuner->getCurrentDeviceInfo();
        std::cout << "Number of compute units: " << di.getMaxComputeUnits() << std::endl;
        size_t cus = di.getMaxComputeUnits();

        // create parameter space
        tuner->addParameter(kernelId, "WORK_GROUP_SIZE_X", { /*1, 2, 4, 8, 16, */32, 64, 128, 256, 512 },
            ktt::ThreadModifierType::Local,
            ktt::ThreadModifierAction::Multiply,
            ktt::Dimension::X);
        tuner->addParameter(kernelId, "UNBOUNDED_WG", { 0, 1 });
        tuner->addParameter(kernelId, "WG_NUM", { 0, cus, cus * 2, cus * 4, cus * 8, cus * 16 });
        tuner->addParameter(kernelId, "VECTOR_SIZE", { 1, 2, 4, 8, 16 },
            ktt::ThreadModifierType::Global,
            ktt::ThreadModifierAction::Divide,
            ktt::Dimension::X);
        tuner->addParameter(kernelId, "USE_ATOMICS", { 0, 1 });
        auto persistConstraint = [](std::vector<size_t> v) { return (v[0] && v[1] == 0) || (!v[0] && v[1] > 0); };
        tuner->addConstraint(kernelId, persistConstraint, { "UNBOUNDED_WG", "WG_NUM" });
        auto persistentAtomic = [](std::vector<size_t> v) { return (v[0] == 1) || (v[0] == 0 && v[1] == 1); };
        tuner->addConstraint(kernelId, persistentAtomic, { "UNBOUNDED_WG", "USE_ATOMICS" } );
        auto unboundedWG = [](std::vector<size_t> v) { return (!v[0] || v[1] >= 32); };
        tuner->addConstraint(kernelId, unboundedWG, { "UNBOUNDED_WG", "WORK_GROUP_SIZE_X" } );

        tuner->setReferenceClass(kernelId, std::make_unique<referenceReduction>(*src, dstId), std::vector<size_t>{ dstId });
        tuner->setValidationMethod(ktt::ValidationMethod::SideBySideComparison, (float)n*500.0f/10000000.0f);
        tuner->setValidationRange(dstId, 1);
    }

/*
    launchComputation is responsible for actual execution of tuned kernel */
    virtual void launchComputation(const size_t kernelId) override {
        ktt::DimensionVector globalSize = getCurrentGlobalSize(kernelId);
        ktt::DimensionVector localSize = getCurrentLocalSize(kernelId);
        std::vector<ktt::ParameterValue> parameterValues = getCurrentConfiguration();
        ktt::DimensionVector myGlobalSize = globalSize;
        
        // change global size for constant numners of work-groups
        //XXX this may be done also by thread modifier operators in constructor
        if (getParameterValue("UNBOUNDED_WG", parameterValues) == 0) {
            myGlobalSize = std::make_tuple(
                getParameterValue("WG_NUM", parameterValues)
                * std::get<0>(localSize), 1, 1);
        }

        // execute reduction kernel
        runKernel(kernelId, myGlobalSize, localSize);

        // execute kernel log n times, when atomics are not used 
        if (getParameterValue("USE_ATOMICS", parameterValues) == 0) {
            size_t n = std::get<0>(globalSize) / std::get<0>(localSize);
            size_t inOffset = 0;
            size_t outOffset = n;
            size_t vectorSize = getParameterValue("VECTOR_SIZE", parameterValues);
            size_t wgSize = std::get<0>(localSize);
            
            size_t iterations = 0; // make sure the end result is in the correct buffer
            while (n > 1 || iterations % 2 == 1) {
                swapKernelArguments(kernelId, srcId, dstId);
                std::get<0>(myGlobalSize) = (n+vectorSize-1) / vectorSize;
                std::get<0>(myGlobalSize) = ((std::get<0>(myGlobalSize)-1)/wgSize + 1) * wgSize;
                if (myGlobalSize == localSize)
                    outOffset = 0; // only one WG will be executed
                updateArgumentScalar(nId, &n);
                updateArgumentScalar(outOffsetId, &outOffset);
                updateArgumentScalar(inOffsetId, &inOffset);
                //std::cout << "n inOfs, outOfs " << n << " " << inOffset << " "
                //    << outOffset << "\n";
                //std::cout << "glob loc " << std::get<0>(myGlobalSize) << " "
                //    << std::get<0>(localSize) << "\n";
                runKernel(kernelId, myGlobalSize, localSize);
                n = (n+wgSize*vectorSize-1)/(wgSize*vectorSize);
                inOffset = outOffset/vectorSize; //XXX input is vectorized, output is scalar
                outOffset += n;
                iterations++;
            }
        }
    }

    void tune() {
        tuner->tuneKernel(kernelId);
        tuner->printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
        tuner->printResult(kernelId, std::string("output.csv"), ktt::PrintFormat::CSV);
    }

/*
    simple utility functions */
    size_t getKernelId() const {
        return kernelId;
    }

private:
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
};
