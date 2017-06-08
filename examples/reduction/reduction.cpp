#include <iostream>
#include <string>
#include <vector>

#include "../../include/ktt.h"

#include "reduction_tunable.h"

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
    const int n = 32*1024*1024;
    const int nAlloc = ((n+16-1)/16)*16; // padd to longest vector size
    std::vector<float> src(nAlloc);
    std::vector<float> dst(nAlloc);
    for (int i = 0; i < n; i++)
    {
        src[i] = 1000.0f*((float)rand()) / ((float) RAND_MAX);
        dst[i] = 0.0f;
    }
    for (int i = n; i < nAlloc; i++)
    {
        src[i] = 0.0f;
        dst[i] = 0.0f;
    }

    ktt::Tuner tuner(platformIndex, deviceIndex);

    tunableReduction* reduction = new tunableReduction(&tuner, &src, &dst, nAlloc);
    tuner.setTuningManipulator(reduction->getKernelId(), std::unique_ptr<tunableReduction>(reduction));
    reduction->tune();

    return 0;
}
