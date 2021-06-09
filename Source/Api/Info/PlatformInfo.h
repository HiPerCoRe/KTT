/** @file PlatformInfo.h
  * Information about compute API platforms.
  */
#pragma once

#include <string>

#include <KttPlatform.h>
#include <KttTypes.h>

namespace ktt
{

/** @class PlatformInfo
  * Class which holds information about a compute API platform.
  */
class KTT_API PlatformInfo
{
public:
    /** @fn explicit PlatformInfo(const PlatformIndex index, const std::string& name)
      * Constructor, which creates new platform info object.
      * @param index Index of platform assigned by KTT framework.
      * @param name Name of platform retrieved from compute API.
      */
    explicit PlatformInfo(const PlatformIndex index, const std::string& name);

    /** @fn PlatformIndex GetIndex() const
      * Getter for index of platform assigned by KTT framework.
      * @return Index of platform assigned by KTT framework.
      */
    PlatformIndex GetIndex() const;

    /** @fn const std::string& GetName() const
      * Getter for name of platform retrieved from compute API.
      * @return Name of platform retrieved from compute API.
      */
    const std::string& GetName() const;

    /** @fn const std::string& GetVendor() const
      * Getter for name of platform vendor retrieved from compute API.
      * @return Name of platform vendor retrieved from compute API.
      */
    const std::string& GetVendor() const;

    /** @fn const std::string& GetVersion() const
      * Getter for platform version retrieved from compute API.
      * @return Platform version retrieved from compute API.
      */
    const std::string& GetVersion() const;

    /** @fn const std::string& GetExtensions() const
      * Getter for list of supported platform extensions retrieved from compute API.
      * @return List of supported platform extensions retrieved from compute API.
      */
    const std::string& GetExtensions() const;

    /** @fn std::string GetString() const
      * Converts platform info to string.
      * @return String containing information about the platform.
      */
    std::string GetString() const;

    /** @fn void SetVendor(const std::string& vendor)
      * Setter for name of platform vendor.
      * @param vendor Name of platform vendor.
      */
    void SetVendor(const std::string& vendor);

    /** @fn void SetVersion(const std::string& version)
      * Setter for platform version.
      * @param version Platform version.
      */
    void SetVersion(const std::string& version);

    /** @fn void SetExtensions(const std::string& extensions)
      * Setter for list of supported platform extensions.
      * @param extensions List of supported platform extensions.
      */
    void SetExtensions(const std::string& extensions);

private:
    PlatformIndex m_Index;
    std::string m_Name;
    std::string m_Vendor;
    std::string m_Version;
    std::string m_Extensions;
};

} // namespace ktt
