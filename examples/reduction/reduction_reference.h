#pragma once

#include <iomanip>
#include "tuner_api.h"

class referenceReduction : public ktt::ReferenceClass
{
public:
    referenceReduction(const std::vector<float>& src, const ktt::ArgumentId resultArgumentId) :
        src(src),
        resultArgumentId(resultArgumentId)
    {}

    // High precision of reduction
    void computeResult() override {
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

    const void* getData(const ktt::ArgumentId id) const override {
        if (id == resultArgumentId) {
            return (void*)res.data();
        }
        return nullptr;
    }

    size_t getNumberOfElements(const size_t argumentId) const override {
        return 1;
    }

private:
    std::vector<float> res;
    std::vector<float> src;
    ktt::ArgumentId resultArgumentId;
};
