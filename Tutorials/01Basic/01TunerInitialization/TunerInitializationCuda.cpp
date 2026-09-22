// KTT tutorial demonstrating how the tuner is initialized.
// Serves more as a sanity check that KTT is working correctly.
// Users are recommended to start with KTT Introductory guide
// at https://github.com/HiPerCoRe/KTT/blob/master/OnboardingGuide.md
// before reading the tutorial's code.

#include <iostream>
#include <vector>

#include <Ktt.h>

int main(int argc, char **argv)
{
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;

    if (argc >= 2)
    {
        platformIndex = std::stoul(std::string(argv[1]));

        if (argc >= 3)
        {
            deviceIndex = std::stoul(std::string(argv[2]));
        }
    }

    // Create new tuner which uses CUDA as compute API.
    ktt::Tuner tuner(platformIndex, deviceIndex, ktt::ComputeApi::CUDA);
    auto deviceInfo = tuner.GetCurrentDeviceInfo();
    std::cout << "Current device info: " << deviceInfo.GetString() << "\n";

    std::cout << "Tuner successfully initialized.\n";

    return 0;
}
