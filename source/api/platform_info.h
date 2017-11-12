/** @file platform_info.h
  * @brief File containing functionality related to retrieving information about compute API platforms.
  */
#pragma once

#include <iostream>
#include <string>
#include "ktt_platform.h"

namespace ktt
{

/** @class PlatformInfo
  * @brief Class which holds information about a compute API platform.
  */
class KTT_API PlatformInfo
{
public:
    /** @fn explicit PlatformInfo(const size_t id, const std::string& name)
      * @brief Constructor, which creates new platform info object.
      * @param id Id of platform assigned by KTT library.
      * @param name Name of platform retrieved from compute API.
      */
    explicit PlatformInfo(const size_t id, const std::string& name);

    /** @fn size_t getId() const
      * @brief Getter for id of platform assigned by KTT library.
      * @return Id of platform assigned by KTT library.
      */
    size_t getId() const;

    /** @fn std::string getName() const
      * @brief Getter for name of platform retrieved from compute API.
      * @return Name of platform retrieved from compute API.
      */
    std::string getName() const;

    /** @fn std::string getVendor() const
      * @brief Getter for name of platform vendor retrieved from compute API.
      * @return Name of platform vendor retrieved from compute API.
      */
    std::string getVendor() const;

    /** @fn std::string getVersion() const
      * @brief Getter for platform version retrieved from compute API.
      * @return Platform version retrieved from compute API.
      */
    std::string getVersion() const;

    /** @fn std::string getExtensions() const
      * @brief Getter for list of supported platform extensions retrieved from compute API.
      * @return List of supported platform extensions retrieved from compute API.
      */
    std::string getExtensions() const;

    /** @fn void setVendor(const std::string& vendor)
      * @brief Setter for name of platform vendor.
      * @param vendor Name of platform vendor.
      */
    void setVendor(const std::string& vendor);

    /** @fn void setVersion(const std::string& version)
      * @brief Setter for platform version.
      * @param version Platform version.
      */
    void setVersion(const std::string& version);

    /** @fn void setExtensions(const std::string& extensions)
      * @brief Setter for list of supported platform extensions.
      * @param extensions List of supported platform extensions.
      */
    void setExtensions(const std::string& extensions);

    KTT_API friend std::ostream& operator<<(std::ostream&, const PlatformInfo&);

private:
    size_t id;
    std::string name;
    std::string vendor;
    std::string version;
    std::string extensions;
};

/** @fn std::ostream& operator<<(std::ostream& outputTarget, const PlatformInfo& platformInfo)
  * @brief Output operator for platform info class.
  * @param outputTarget Location where information about platform is printed.
  * @param platformInfo Platform info object that is printed.
  * @return Output target to support chain of output operations.
  */
KTT_API std::ostream& operator<<(std::ostream& outputTarget, const PlatformInfo& platformInfo);

} // namespace ktt
