#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../../include/ktt.h"

#include "coulomb_sum_tunable.h"

int main(int argc, char** argv)
{
    // Initialize platform index, device index and paths to kernels
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

    // Set the problem size
    const int atoms = 32;
    const int gridSize = 128;

    // Create tuner
    ktt::Tuner tuner(platformIndex, deviceIndex);

    // Create tunable coulomb and execute tuning
    tunableCoulomb* coulomb = new tunableCoulomb(&tuner, gridSize, atoms);
    tuner.setTuningManipulator(coulomb->getKernelId(), std::unique_ptr<tunableCoulomb>(coulomb));
    coulomb->tune();

    return 0;
}
