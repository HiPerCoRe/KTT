#pragma once

#include "tuner_api.h"

class TunableReduction : public ktt::TuningManipulator {
public:

/* 
    Constructor creates internal structures and setups tuning environment */
    TunableReduction(const int n, std::vector<float> *src, std::vector<float> *dst, const ktt::ArgumentId srcId, const ktt::ArgumentId dstId,
        const ktt::ArgumentId nId, const ktt::ArgumentId inOffsetId, const ktt::ArgumentId outOffsetId) :
        n(n),
        src(src),
        dst(dst),
        srcId(srcId),
        dstId(dstId),
        nId(nId),
        inOffsetId(inOffsetId),
        outOffsetId(outOffsetId)
    {}

/*
    launchComputation is responsible for actual execution of tuned kernel */
    void launchComputation(const ktt::KernelId kernelId) override {
        ktt::DimensionVector globalSize = getCurrentGlobalSize(kernelId);
        ktt::DimensionVector localSize = getCurrentLocalSize(kernelId);
        std::vector<ktt::ParameterPair> parameterValues = getCurrentConfiguration();
        ktt::DimensionVector myGlobalSize = globalSize;
        
        // change global size for constant numners of work-groups
        //XXX this may be done also by thread modifier operators in constructor
        if (getParameterValue("UNBOUNDED_WG", parameterValues) == 0) {
            myGlobalSize = ktt::DimensionVector(getParameterValue("WG_NUM", parameterValues) * localSize.getSizeX());
        }

        // execute reduction kernel
        runKernel(kernelId, myGlobalSize, localSize);

        // execute kernel log n times, when atomics are not used 
        if (getParameterValue("USE_ATOMICS", parameterValues) == 0) {
            size_t n = globalSize.getSizeX() / localSize.getSizeX();
            size_t inOffset = 0;
            size_t outOffset = n;
            size_t vectorSize = getParameterValue("VECTOR_SIZE", parameterValues);
            size_t wgSize = localSize.getSizeX();
            
            size_t iterations = 0; // make sure the end result is in the correct buffer
            while (n > 1 || iterations % 2 == 1) {
                swapKernelArguments(kernelId, srcId, dstId);
                myGlobalSize.setSizeX((n + vectorSize - 1) / vectorSize);
                myGlobalSize.setSizeX(((myGlobalSize.getSizeX() - 1) / wgSize + 1) * wgSize);
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

private:
    int n;
    std::vector<float> *src;
    std::vector<float> *dst;
    ktt::ArgumentId srcId;
    ktt::ArgumentId dstId;
    ktt::ArgumentId nId;
    ktt::ArgumentId inOffsetId;
    ktt::ArgumentId outOffsetId;
};
