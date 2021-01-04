/** @file platform_info.h
  * Functionality related to retrieving information about compute API platforms.
  */
#pragma once

#include <iostream>
#include <string>
#include <ktt_platform.h>
#include <ktt_types.h>

namespace ktt
{

/** @class PlatformInfo
  * Class which holds information about a compute API platform.
  */
class KTT_API PlatformInfo
{
public:
    /** @fn explicit PlatformInfo(const PlatformIndex platform, const std::string& name)
      * Constructor, which creates new platform info object.
      * @param platform Index of platform assigned by KTT framework.
      * @param name Name of platform retrieved from compute API.
      */
    explicit PlatformInfo(const PlatformIndex platform, const std::string& name);

    /** @fn PlatformIndex getId() const
      * Getter for index of platform assigned by KTT framework.
      * @return Index of platform assigned by KTT framework.
      */
    PlatformIndex getId() const;

    /** @fn const std::string& getName() const
      * Getter for name of platform retrieved from compute API.
      * @return Name of platform retrieved from compute API.
      */
    const std::string& getName() const;

    /** @fn const std::string& getVendor() const
      * Getter for name of platform vendor retrieved from compute API.
      * @return Name of platform vendor retrieved from compute API.
      */
    const std::string& getVendor() const;

    /** @fn const std::string& getVersion() const
      * Getter for platform version retrieved from compute API.
      * @return Platform version retrieved from compute API.
      */
    const std::string& getVersion() const;

    /** @fn const std::string& getExtensions() const
      * Getter for list of supported platform extensions retrieved from compute API.
      * @return List of supported platform extensions retrieved from compute API.
      */
    const std::string& getExtensions() const;

    /** @fn void setVendor(const std::string& vendor)
      * Setter for name of platform vendor.
      * @param vendor Name of platform vendor.
      */
    void setVendor(const std::string& vendor);

    /** @fn void setVersion(const std::string& version)
      * Setter for platform version.
      * @param version Platform version.
      */
    void setVersion(const std::string& version);

    /** @fn void setExtensions(const std::string& extensions)
      * Setter for list of supported platform extensions.
      * @param extensions List of supported platform extensions.
      */
    void setExtensions(const std::string& extensions);

private:
    PlatformIndex id;
    std::string name;
    std::string vendor;
    std::string version;
    std::string extensions;
};

/** @fn std::ostream& operator<<(std::ostream& outputTarget, const PlatformInfo& platformInfo)
  * @brief Output operator for platform info class.
  * @param outputTarget Location where information about platform will be printed.
  * @param platformInfo Platform info object that will be printed.
  * @return Output target to support chaining of output operations.
  */
KTT_API std::ostream& operator<<(std::ostream& outputTarget, const PlatformInfo& platformInfo);

} // namespace ktt
