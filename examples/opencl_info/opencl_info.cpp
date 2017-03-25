#include <iostream>
#include <string>
#include <vector>

#include "../../include/ktt.h"

int main()
{
    ktt::Tuner::printComputeAPIInfo(std::cout);

    std::vector<ktt::PlatformInfo> platformInfo = ktt::Tuner::getPlatformInfoAll();
    for (size_t i = 0; i < platformInfo.size(); i++)
    {
        std::cout << platformInfo.at(i) << std::endl;

        std::vector<ktt::DeviceInfo> deviceInfo = ktt::Tuner::getDeviceInfoAll(i);
        for (size_t j = 0; j < deviceInfo.size(); j++)
        {
            std::cout << deviceInfo.at(j) << std::endl;
        }
    }

    return 0;
}
