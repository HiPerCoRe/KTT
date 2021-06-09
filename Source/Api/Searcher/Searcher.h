/** @file Searcher.h
  * Interface for implementing kernel configuration searchers.
  */
#pragma once

#include <cstdint>
#include <set>
#include <vector>

#include <Api/Configuration/KernelConfiguration.h>
#include <Api/Output/KernelResult.h>
#include <KttPlatform.h>

namespace ktt
{

class ConfigurationData;

/** @class Searcher
  * Class which is used to decide which kernel configuration will be run next during the kernel tuning process.
  */
class KTT_API Searcher
{
public:
    /** @fn  virtual ~Searcher() = default
      * Searcher destructor. Inheriting class can override destructor with custom implementation. Default implementation is
      * provided by KTT framework.
      */
    virtual ~Searcher() = default;

    /** @fn virtual void OnInitialize()
      * Called after searcher is initialized with kernel configurations. The first kernel configuration as well as custom searcher
      * parameters should be initialized here.
      */
    virtual void OnInitialize();

    /** @fn virtual void OnReset()
      * Called before searcher is reset to initial state and configurations are removed. Custom searcher parameters
      * should be reset here.
      */
    virtual void OnReset();

    /** @fn virtual bool CalculateNextConfiguration(const KernelResult& previousResult) = 0
      * Calculates the configuration which will be run next. Called after processing the current configuration if there are any
      * remaining unexplored configurations.
      * @param previousResult Result from the last tested configuration. See KernelResult for more information.
      * @return True if the next configuration was successfully calculated, false otherwise. If false is returned, configuration
      * space exploration will be stopped.
      */
    virtual bool CalculateNextConfiguration(const KernelResult& previousResult) = 0;

    /** @fn virtual KernelConfiguration GetCurrentConfiguration() const = 0
      * Returns current kernel configuration. Note that this may be called repeatedly before calculating next configuration. In
      * that case, the returned configuration must always be the same.
      * @return Current configuration.
      */
    virtual KernelConfiguration GetCurrentConfiguration() const = 0;

    /** @fn Searcher()
      * Default searcher constructor. Should be called from inheriting searcher's constructor.
      */
    Searcher();

    /** @fn KernelConfiguration GetConfiguration(const uint64_t index) const
      * Returns configuration with the specified index.
      * @param index Index of the configuration that should be retrieved. The index must be less than the count returned by
      * GetConfigurationsCount method.
      * @return Configuration with the specified index.
      */
    KernelConfiguration GetConfiguration(const uint64_t index) const;

    /** @fn uint64_t GetIndex(const KernelConfiguration& configuration) const
      * Returns index of the specified configuration.
      * @param configuration Configuration for which the index will be retrieved.
      * @return Index of the specified configuration.
      */
    uint64_t GetIndex(const KernelConfiguration& configuration) const;

    /** @fn KernelConfiguration GetRandomConfiguration() const
      * Returns random unexplored configuration.
      * @return Random unexplored configuration.
      */
    KernelConfiguration GetRandomConfiguration() const;

    /** @fn std::vector<KernelConfiguration> GetNeighbourConfigurations(const KernelConfiguration& configuration,
      * const uint64_t maxDifferences, const size_t maxNeighbours = 3) const
      * Retrieves unexplored neighbour configurations of the specified configuration.
      * @param configuration Configuration whose neighbours will be retrieved.
      * @param maxDifferences Maximum number of parameters in neighbour configurations whose values differ from the original
      * configuration.
      * @param maxNeighbours Maximum number of retrieved neighbour configurations.
      * @return Neighbours of the specified configuration. Note that the result might be empty in case no suitable configurations
      * were found.
      */
    std::vector<KernelConfiguration> GetNeighbourConfigurations(const KernelConfiguration& configuration,
        const uint64_t maxDifferences, const size_t maxNeighbours = 3) const;

    /** @fn uint64_t GetConfigurationsCount() const
      * Returns total number of valid kernel configurations.
      * @return Number of valid kernel configurations.
      */
    uint64_t GetConfigurationsCount() const;

    /** @fn const std::set<uint64_t>& GetExploredIndices() const
      * Returns indices of already explored configurations.
      * @return Indices of already explored configurations.
      */
    const std::set<uint64_t>& GetExploredIndices() const;

    /** @fn bool IsInitialized() const
      * Returns whether searcher is initialized.
      * @return True if searcher is initialized, false otherwise.
      */
    bool IsInitialized() const;

    /** @fn void Initialize(const ConfigurationTree& tree)
      * Initializes searcher with the tree of configurations which can be explored for corresponding kernel.
      * @param data Internal structure containing configurations which can be explored
      */
    void Initialize(const ConfigurationData& data);

    /** @fn void Reset()
      * Resets searcher to initial state and clears configuration tree.
      */
    void Reset();

private:
    const ConfigurationData* m_Data;
};

} // namespace ktt
