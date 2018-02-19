#pragma once

#include "tuner_api.h"

//the main code and reference kernel adopted from SHOC benchmark, example sort
class referenceSort : public ktt::ReferenceClass {
  public:
    // Constructor creates internal structures and setups the environment
    // it takes arguments from command line and generated input data
    referenceSort(const std::vector<unsigned int>& data) :
      data(data)
    {}

    //run the code with kernels
    void computeResult() override {
      std::sort(data.begin(), data.end());
    }

    void* getData(const ktt::ArgumentId id) override {
        return data.data();
    }

    size_t getNumberOfElements(const ktt::ArgumentId argumentId) const override {
      return data.size();
    }

  private:
      std::vector<unsigned int> data;
};
