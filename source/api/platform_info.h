#pragma once

#ifndef KTT_API
#if defined(_MSC_VER) && !defined(KTT_TESTS)
    #pragma warning(disable : 4251) // MSVC irrelevant warning (as long as there are no public attributes)
    #if defined(KTT_LIBRARY)
        #define KTT_API __declspec(dllexport)
    #else
        #define KTT_API __declspec(dllimport)
    #endif // KTT_LIBRARY
#else
    #define KTT_API
#endif // _MSC_VER
#endif // KTT_API

#include <iostream>
#include <string>

namespace ktt
{

class KTT_API PlatformInfo
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

    KTT_API friend std::ostream& operator<<(std::ostream&, const PlatformInfo&);

private:
    size_t id;
    std::string name;
    std::string vendor;
    std::string version;
    std::string extensions;
};

KTT_API std::ostream& operator<<(std::ostream& outputTarget, const PlatformInfo& platformInfo);

} // namespace ktt
