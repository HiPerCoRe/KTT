#include <iostream>
#include <vector>
#include "tuner_api.h"

int main()
{
    // Create new tuner which uses OpenCL as compute API.
    ktt::Tuner tuner(0, 0);

    // Print basic information about available platforms and devices to standard output.
    tuner.printComputeAPIInfo(std::cout);

    // Print detailed information about platforms and devices to standard output.
    std::vector<ktt::PlatformInfo> platformInfo = tuner.getPlatformInfo();

    for (size_t i = 0; i < platformInfo.size(); i++)
    {
        std::cout << platformInfo.at(i) << std::endl;

        std::vector<ktt::DeviceInfo> deviceInfo = tuner.getDeviceInfo(static_cast<ktt::PlatformIndex>(i));
        for (const auto& currentDevice : deviceInfo)
        {
            std::cout << currentDevice << std::endl;
        }
    }

    return 0;
}
