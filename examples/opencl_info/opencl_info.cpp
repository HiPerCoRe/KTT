#include <iostream>
#include <string>
#include <vector>

#include "../../include/ktt.h"

int main()
{
    // Print basic information about available platforms and devices to standard output
    ktt::Tuner::printComputeAPIInfo(std::cout);

    // Print detailed information about all platforms and devices to standard output
    std::vector<ktt::PlatformInfo> platformInfo = ktt::Tuner::getPlatformInfo();
    for (size_t i = 0; i < platformInfo.size(); i++)
    {
        std::cout << platformInfo.at(i) << std::endl;

        std::vector<ktt::DeviceInfo> deviceInfo = ktt::Tuner::getDeviceInfo(i);
        for (size_t j = 0; j < deviceInfo.size(); j++)
        {
            std::cout << deviceInfo.at(j) << std::endl;
        }
    }

    return 0;
}
