#include <iostream>
#include <vector>
#include "tuner_api.h"

int main()
{
    // Create new tuner
    ktt::Tuner tuner(0, 0);

    // Print basic information about available platforms and devices to standard output
    tuner.printComputeApiInfo(std::cout);

    // Print detailed information about all platforms and devices to standard output
    std::vector<ktt::PlatformInfo> platformInfo = tuner.getPlatformInfo();
    for (size_t i = 0; i < platformInfo.size(); i++)
    {
        std::cout << platformInfo.at(i) << std::endl;

        std::vector<ktt::DeviceInfo> deviceInfo = tuner.getDeviceInfo(i);
        for (const auto& currentDevice : deviceInfo)
        {
            std::cout << currentDevice << std::endl;
        }
    }

    return 0;
}
