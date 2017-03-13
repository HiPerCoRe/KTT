#include <iostream>
#include <string>
#include <vector>

#include "../../include/ktt.h"

std::string deviceTypeToString(const ktt::DeviceType& deviceType)
{
    switch (deviceType)
    {
    case ktt::DeviceType::ACCELERATOR:
        return std::string("ACCELERATOR");
    case ktt::DeviceType::CPU:
        return std::string("CPU");
    case ktt::DeviceType::CUSTOM:
        return std::string("CUSTOM");
    case ktt::DeviceType::DEFAULT:
        return std::string("DEFAULT");
    default:
        return std::string("GPU");
    }
}

int main()
{
    ktt::Tuner::printComputeAPIInfo(std::cout);

    std::vector<ktt::PlatformInfo> platformInfo = ktt::Tuner::getPlatformInfoAll();
    for (size_t i = 0; i < platformInfo.size(); i++)
    {
        std::cout << "Printing detailed platform info for platform " << i << std::endl;
        std::cout << platformInfo.at(i).getName() << platformInfo.at(i).getVendor() << platformInfo.at(i).getVersion() << std::endl;
        std::cout << "Extensions: " << platformInfo.at(i).getExtensions() << std::endl << std::endl;

        std::vector<ktt::DeviceInfo> deviceInfo = ktt::Tuner::getDeviceInfoAll(i);
        for (size_t j = 0; j < deviceInfo.size(); j++)
        {
            std::cout << "Printing detailed device info for device " << j << " on platform " << i << std::endl;
            std::cout << deviceInfo.at(j).getName() << deviceInfo.at(j).getVendor() << deviceInfo.at(j).getGlobalMemorySize() << " "
                << deviceInfo.at(j).getLocalMemorySize() << " " << deviceInfo.at(j).getMaxConstantBufferSize() << " "
                << deviceInfo.at(j).getMaxComputeUnits() << " " << deviceInfo.at(j).getMaxWorkGroupSize() << " "
                << deviceTypeToString(deviceInfo.at(j).getDeviceType()) << std::endl << std::endl;
            //std::cout << "Extensions: " << deviceInfo.at(j).getExtensions() << std::endl << std::endl;
        }
    }

    return 0;
}
