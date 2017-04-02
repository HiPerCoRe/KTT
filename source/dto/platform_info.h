#pragma once

#include <iostream>
#include <string>

namespace ktt
{

class PlatformInfo
{
public:
    explicit PlatformInfo(const size_t id, const std::string& name);

    size_t getId() const;
    std::string getName() const;
    std::string getVendor() const;
    std::string getVersion() const;
    std::string getExtensions() const;

    void setVendor(const std::string& vendor);
    void setVersion(const std::string& version);
    void setExtensions(const std::string& extensions);

    friend std::ostream& operator<<(std::ostream& outputTarget, const PlatformInfo& platformInfo);

private:
    size_t id;
    std::string name;
    std::string vendor;
    std::string version;
    std::string extensions;
};

std::ostream& operator<<(std::ostream& outputTarget, const PlatformInfo& platformInfo);

} // namespace ktt
