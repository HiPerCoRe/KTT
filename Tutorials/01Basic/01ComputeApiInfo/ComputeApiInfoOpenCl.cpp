// KTT tutorial demonstrating simple platform info print
// Users are recommended to start with KTT Introductory guide
// at https://github.com/HiPerCoRe/KTT/blob/master/OnboardingGuide.md
// before reading the tutorial's code.

#include <iostream>
#include <vector>

#include <Ktt.h>

int main()
{
    // Create new tuner which uses OpenCL as compute API.
    ktt::Tuner tuner(0, 0, ktt::ComputeApi::OpenCL);

    // Print information about platforms and devices to standard output.
    std::vector<ktt::PlatformInfo> platforms = tuner.GetPlatformInfo();

    for (size_t i = 0; i < platforms.size(); ++i)
    {
        std::cout << platforms[i].GetString() << std::endl;
        std::vector<ktt::DeviceInfo> devices = tuner.GetDeviceInfo(static_cast<ktt::PlatformIndex>(i));

        for (const auto& device : devices)
        {
            std::cout << device.GetString() << std::endl;
        }
    }

    return 0;
}
