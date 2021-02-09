/** @file KernelConfiguration.h
  * Definition of kernel tuning configuration.
  */
#pragma once

#include <string>
#include <vector>

#include <Api/Configuration/ParameterPair.h>
#include <KttPlatform.h>

namespace ktt
{

/** @class KernelConfiguration
  * Class which describes single kernel tuning configuration.
  */
class KTT_API KernelConfiguration
{
public:
    /** @fn KernelConfiguration()
      * Default constructor, creates invalid kernel configuration.
      */
    KernelConfiguration();

    /** @fn explicit KernelConfiguration(const std::vector<ParameterPair>& pairs)
      * Constructs kernel configuration with the specified parameter pairs.
      * @param pairs Values of tuning parameters for the configuration.
      */
    explicit KernelConfiguration(const std::vector<ParameterPair>& pairs);

    /** @fn const std::vector<ParameterPair>& GetPairs() const
      * Returns values of tuning parameters for kernel configuration.
      * @return Values of tuning parameters. See ParameterPair for more information.
      */
    const std::vector<ParameterPair>& GetPairs() const;

    /** @fn bool IsValid() const
      * Checks whether kernel configuration is valid.
      * @return True if kernel configuration is valid, false otherwise.
      */
    bool IsValid() const;

    /** @fn std::string GeneratePrefix() const
      * Generates kernel source preprocessor definitions from configuration.
      * @return Kernel source preprocessor definitions.
      */
    std::string GeneratePrefix() const;

    /** @fn std::string GetString() const
      * Converts configuration to string.
      * @return String in format "parameter1String, parameter2String, ...".
      */
    std::string GetString() const;

private:
    std::vector<ParameterPair> m_Pairs;
};

} // namespace ktt
