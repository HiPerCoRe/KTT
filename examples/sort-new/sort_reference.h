#pragma once

#include "tuner_api.h"

class ReferenceSort : public ktt::ReferenceClass {
public:
    ReferenceSort(const std::vector<unsigned int>& data) :
        data(data)
    {}

    void computeResult() override {
        std::sort(data.begin(), data.end());
    }

    void* getData(const ktt::ArgumentId) override {
        return data.data();
    }

    size_t getNumberOfElements(const ktt::ArgumentId) const override {
      return data.size();
    }

private:
    std::vector<unsigned int> data;
};
