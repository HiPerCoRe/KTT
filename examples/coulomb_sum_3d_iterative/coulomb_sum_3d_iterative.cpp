#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "tuner_api.h"
#include "coulomb_sum_3d_iterative_tunable.h"

int main(int argc, char** argv)
{
    // Initialize platform index, device index and paths to kernels
    size_t platformIndex = 0;
    size_t deviceIndex = 0;

    if (argc >= 2)
    {
        platformIndex = std::stoul(std::string{argv[1]});
        if (argc >= 3)
        {
            deviceIndex = std::stoul(std::string{argv[2]});
        }
    }

    // Set the problem size
    const int atoms = 4000;
    const int gridSize = 256;

    // Create tuner
    ktt::Tuner tuner(platformIndex, deviceIndex);

    // Create tunable coulomb and execute tuning
    TunableCoulomb* coulomb = new TunableCoulomb(&tuner, gridSize, atoms);
    tuner.setTuningManipulator(coulomb->getKernelId(), std::unique_ptr<TunableCoulomb>(coulomb));
    coulomb->tune();

    return 0;
}
