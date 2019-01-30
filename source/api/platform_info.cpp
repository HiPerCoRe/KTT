#include <api/platform_info.h>

namespace ktt
{

PlatformInfo::PlatformInfo(const PlatformIndex platform, const std::string& name) :
    id(platform),
    name(name)
{}

PlatformIndex PlatformInfo::getId() const
{
    return id;
}

const std::string& PlatformInfo::getName() const
{
    return name;
}

const std::string& PlatformInfo::getVendor() const
{
    return vendor;
}

const std::string& PlatformInfo::getVersion() const
{
    return version;
}

const std::string& PlatformInfo::getExtensions() const
{
    return extensions;
}

void PlatformInfo::setVendor(const std::string& vendor)
{
    this->vendor = vendor;
}

void PlatformInfo::setVersion(const std::string& version)
{
    this->version = version;
}

void PlatformInfo::setExtensions(const std::string& extensions)
{
    this->extensions = extensions;
}

std::ostream& operator<<(std::ostream& outputTarget, const PlatformInfo& platformInfo)
{
    outputTarget << "Printing detailed info for platform with index: " << platformInfo.getId() << std::endl;
    outputTarget << "Name: " << platformInfo.getName() << std::endl;
    outputTarget << "Vendor: " << platformInfo.getVendor() << std::endl;
    outputTarget << "Compute API version: " << platformInfo.getVersion() << std::endl;
    outputTarget << "Extensions: " << platformInfo.getExtensions() << std::endl;
    return outputTarget;
}

} // namespace ktt
