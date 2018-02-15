#pragma once

#include "tuner_api.h"

//the main code and reference kernel adopted from SHOC benchmark, example sort
class referenceSort : public ktt::ReferenceClass {
  public:
    // Constructor creates internal structures and setups the environment
    // it takes arguments from command line and generated input data
    referenceSort(ktt::Tuner *tuner, std::vector<unsigned int> *in)
  {
    this->tuner = tuner;

    this->in = in;

  }

    //run the code with kernels
    void computeResult() override {
      std::sort(in->begin(), in->end());
    }
    void* getData(const ktt::ArgumentId id) override {
        return in->data();
    }

    size_t getNumberOfElements(const ktt::ArgumentId argumentId) const override {
      return in->size();
    }


  private:

    ktt::Tuner* tuner;
    std::vector<unsigned int> *in;
    std::vector<unsigned int> *out;
};
