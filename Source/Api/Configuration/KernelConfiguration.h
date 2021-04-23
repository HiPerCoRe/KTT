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

    /** @fn void Merge(const KernelConfiguration& other)
      * Merges two configurations, adding paramater values from the specified configuration into this one. If the configurations
      * share some parameters, the values of parameters from this configuration are preserved.
      * @param other Configuration that will be merged into this one.
      */
    void Merge(const KernelConfiguration& other);
    
    /** @fn std::vector<KernelConfiguration> GenerateNeighbours(const std::string& parameter, const std::vector<ParameterPair>& pairs) const
      * Generates neighbour configurations which differ in the specified parameter.
      * @param parameter Parameter which will be different in the generated configurations. All other parameters will remain identical.
      * @param pairs All valid pairs for the previously specified parameter.
      * @return Generated neighbour configurations.
      */
    std::vector<KernelConfiguration> GenerateNeighbours(const std::string& parameter, const std::vector<ParameterPair>& pairs) const;

    /** @fn bool operator==(const KernelConfiguration& other) const
      * Checks whether kernel configuration is equal to other. I.e., it has the same parameter pairs with the same values.
      * @param other Comparison target.
      * @return True if the configurations are equal. False otherwise.
      */
    bool operator==(const KernelConfiguration& other) const;

    /** @fn bool operator!=(const KernelConfiguration& other) const
      * Checks whether kernel configuration is equal to other. I.e., it has different parameter pairs or the pairs have different
      * values.
      * @param other Comparison target.
      * @return True if the configurations are not equal. False otherwise.
      */
    bool operator!=(const KernelConfiguration& other) const;

private:
    std::vector<ParameterPair> m_Pairs;
};

} // namespace ktt
