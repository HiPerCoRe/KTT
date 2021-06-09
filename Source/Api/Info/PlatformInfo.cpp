#include <Api/Info/PlatformInfo.h>

namespace ktt
{

PlatformInfo::PlatformInfo(const PlatformIndex index, const std::string& name) :
    m_Index(index),
    m_Name(name)
{}

PlatformIndex PlatformInfo::GetIndex() const
{
    return m_Index;
}

const std::string& PlatformInfo::GetName() const
{
    return m_Name;
}

const std::string& PlatformInfo::GetVendor() const
{
    return m_Vendor;
}

const std::string& PlatformInfo::GetVersion() const
{
    return m_Version;
}

const std::string& PlatformInfo::GetExtensions() const
{
    return m_Extensions;
}

std::string PlatformInfo::GetString() const
{
    std::string result;

    result += "Information about platform with index: " + std::to_string(m_Index) + "\n";
    result += "Name: " + m_Name + "\n";
    result += "Vendor: " + m_Vendor + "\n";
    result += "Compute API version: " + m_Version + "\n";
    result += "Extensions: " + m_Extensions + "\n";

    return result;
}

void PlatformInfo::SetVendor(const std::string& vendor)
{
    m_Vendor = vendor;
}

void PlatformInfo::SetVersion(const std::string& version)
{
    m_Version = version;
}

void PlatformInfo::SetExtensions(const std::string& extensions)
{
    m_Extensions = extensions;
}

} // namespace ktt
