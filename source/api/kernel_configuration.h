/** @file kernel_configuration.h
  * Class which describes single kernel tuning configuration.
  */
#pragma once

#include <map>
#include <ostream>
#include <vector>
#include <api/dimension_vector.h>
#include <api/parameter_pair.h>
#include <ktt_platform.h>
#include <ktt_types.h>

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

    /** @fn explicit KernelConfiguration(const DimensionVector& globalSize, const DimensionVector& localSize,
      * const std::vector<ParameterPair>& parameterPairs)
      * Constructs kernel configuration for a single kernel.
      * @param globalSize Global kernel thread size for the configuration.
      * @param localSize Local kernel thread size for the configuration.
      * @param parameterPairs Values of tuning parameters for the configuration.
      */
    explicit KernelConfiguration(const DimensionVector& globalSize, const DimensionVector& localSize,
        const std::vector<ParameterPair>& parameterPairs);

    /** @fn explicit KernelConfiguration(const std::map<KernelId, DimensionVector>& compositionGlobalSizes,
      * const std::map<KernelId, DimensionVector>& compositionLocalSizes, const std::vector<ParameterPair>& parameterPairs)
      * Constructs kernel configuration for a kernel composition.
      * @param compositionGlobalSizes Global kernel thread sizes of the kernels inside composition for the configuration.
      * @param compositionLocalSizes Local kernel thread sizes of the kernels inside composition for the configuration.
      * @param parameterPairs Values of tuning parameters for the configuration.
      */
    explicit KernelConfiguration(const std::map<KernelId, DimensionVector>& compositionGlobalSizes,
        const std::map<KernelId, DimensionVector>& compositionLocalSizes, const std::vector<ParameterPair>& parameterPairs);

    /** @fn const DimensionVector& getGlobalSize() const
      * Retrieves global kernel thread size for the configuration. Valid value is returned only for single kernel configurations.
      * @return Global kernel thread size for the configuration.
      */
    const DimensionVector& getGlobalSize() const;

    /** @fn const DimensionVector& getLocalSize() const
      * Retrieves local kernel thread size for the configuration. Valid value is returned only for single kernel configurations.
      * @return Local kernel thread size for the configuration.
      */
    const DimensionVector& getLocalSize() const;

    /** @fn const DimensionVector& getCompositionKernelGlobalSize(const KernelId id) const
      * Retrieves global kernel thread size of the specified kernel inside composition for the configuration. Valid value is returned only for
      * kernel composition configurations.
      * @param id Kernel whose thread size will be returned.
      * @return Global kernel thread size of the specified kernel.
      */
    const DimensionVector& getCompositionKernelGlobalSize(const KernelId id) const;

    /** @fn const DimensionVector& getCompositionKernelLocalSize(const KernelId id) const
      * Retrieves local kernel thread size of the specified kernel inside composition for the configuration. Valid value is returned only for
      * kernel composition configurations.
      * @param id Kernel whose thread size will be returned.
      * @return Local kernel thread size of the specified kernel.
      */
    const DimensionVector& getCompositionKernelLocalSize(const KernelId id) const;

    /** @fn std::vector<DimensionVector> getGlobalSizes() const
      * Returns global kernel thread sizes of all the kernels inside the configuration. If called on a single kernel configuration,
      * the size of returned vector will be 1.
      * @return Global kernel thread sizes of all the kernels inside the configuration.
      */
    std::vector<DimensionVector> getGlobalSizes() const;

    /** @fn std::vector<DimensionVector> getLocalSizes() const
      * Returns local kernel thread sizes of all the kernels inside the configuration. If called on a single kernel configuration,
      * the size of returned vector will be 1.
      * @return Local kernel thread sizes of all the kernels inside the configuration.
      */
    std::vector<DimensionVector> getLocalSizes() const;

    /** @fn const std::vector<ParameterPair>& getParameterPairs() const
      * Returns values of tuning parameters for kernel configuration.
      * @return Values of tuning parameters. See ParameterPair for more information.
      */
    const std::vector<ParameterPair>& getParameterPairs() const;

    /** @fn bool isValid() const
      * Checks whether kernel configuration is valid.
      * @return True if kernel configuration is valid, false otherwise.
      */
    bool isValid() const;

private:
    DimensionVector globalSize;
    DimensionVector localSize;
    std::map<KernelId, DimensionVector> compositionGlobalSizes;
    std::map<KernelId, DimensionVector> compositionLocalSizes;
    std::vector<ParameterPair> parameterPairs;
    bool validConfiguration;
};

/** @fn std::ostream& operator<<(std::ostream& outputTarget, const KernelConfiguration& configuration)
  * @brief Output operator for kernel configuration.
  * @param outputTarget Location where information about kernel configuration will be printed.
  * @param configuration Kernel configuration which will be printed.
  * @return Output target to support chaining of output operations.
  */
KTT_API std::ostream& operator<<(std::ostream& outputTarget, const KernelConfiguration& configuration);

} // namespace ktt
