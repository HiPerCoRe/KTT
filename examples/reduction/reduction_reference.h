#pragma once

#include <iomanip>

#include "tuner_api.h"

class referenceReduction : public ktt::ReferenceClass
{
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
        std::cout << "Reference in double: " << std::setprecision(10) << resD[0] << std::endl;
    }

    virtual const void* getData(const size_t argumentId) const override {
        if (argumentId == resultArgumentId) {
            return (void*)res.data();
        }
        throw std::runtime_error("No result available for specified argument id");
    }

    virtual size_t getNumberOfElements(const size_t argumentId) const override {
        return 1;
    }

private:
    std::vector<float> res;
    std::vector<float> src;
    size_t resultArgumentId;
};
