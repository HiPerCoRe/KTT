#pragma once

#include <string>

namespace ktt
{

class Platform
{
public:
    explicit Platform(const size_t id, const std::string& name):
        id(id),
        name(name)
    {}

    size_t getId() const
    {
        return id;
    }

    std::string getName() const
    {
        return name;
    }

    std::string getVendor() const
    {
        return vendor;
    }

    std::string getVersion() const
    {
        return version;
    }

    std::string getExtensions() const
    {
        return extensions;
    }

    void setVendor(const std::string& vendor)
    {
        this->vendor = vendor;
    }

    void setVersion(const std::string& version)
    {
        this->version = version;
    }

    void setExtensions(const std::string& extensions)
    {
        this->extensions = extensions;
    }

private:
    size_t id;
    std::string name;
    std::string vendor;
    std::string version;
    std::string extensions;
};

} // namespace ktt
