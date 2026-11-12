/** @file Searcher.h
  * Interface for implementing kernel configuration searchers.
  */
#pragma once

#include <cstdint>
#include <optional>
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

    /** @fn virtual void OnSeed(const uint64_t seed)
      * Called before OnInitialize() whenever a seed was assigned to the searcher through its constructor or the SetSeed
      * method. Searchers which own a random number generator should seed it here, so that the generator is reinitialized
      * every time the configuration space is (re)generated. The default implementation does nothing.
      * @param seed Seed assigned to the searcher.
      */
    virtual void OnSeed(const uint64_t seed);

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
      * Default searcher constructor. Should be called from inheriting searcher's constructor. Searcher created this way
      * uses a randomly initialized source of randomness, so repeated tuning runs explore configurations in a different
      * order.
      */
    Searcher();

    /** @fn explicit Searcher(const uint64_t seed)
      * Searcher constructor which assigns a seed to the searcher. Should be called from inheriting searcher's constructor.
      * See SetSeed() method for details about the effect of the seed.
      * @param seed Seed for the source of randomness used by the searcher.
      */
    explicit Searcher(const uint64_t seed);

    /** @fn void SetSeed(const uint64_t seed)
      * Assigns a seed to the searcher. The seed initializes both the random number generator used by the framework to
      * pick random configurations (see GetRandomConfiguration method) and the generator owned by the searcher itself, if
      * it has one. The generators are seeded every time the configuration space is generated for the corresponding
      * kernel, so a searcher with a seed assigned draws the same sequence of random numbers in every tuning run.
      *
      * Note that a seeded sequence of random numbers does not imply a fully deterministic tuning process. Determinism is
      * guaranteed only for RandomSearcher and for searchers whose decisions do not depend on measured kernel durations,
      * and for any searcher during simulated tuning (see Tuner::SimulateTuning method), where kernel durations are
      * replayed from previously collected results. McmcSearcher and ProfileBasedSearcher select configurations based on
      * durations measured during tuning, which fluctuate between runs, so these searchers may explore different
      * configurations in different runs even when the seed is set.
      *
      * The seed must be assigned before the configuration space is generated for the corresponding kernel, i.e., before
      * the first tuning method is called or before Tuner::InitializeConfigurationData method is called.
      * @param seed Seed for the source of randomness used by the searcher.
      */
    void SetSeed(const uint64_t seed);

    /** @fn bool HasSeed() const
      * Returns whether a seed was assigned to the searcher.
      * @return True if a seed was assigned, false otherwise.
      */
    bool HasSeed() const;

    /** @fn uint64_t GetSeed() const
      * Returns the seed assigned to the searcher. Can only be called when HasSeed() method returns true.
      * @return Seed assigned to the searcher.
      */
    uint64_t GetSeed() const;

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

    /**
     * @fn uint64_t GetUnexploredConfigurationsCount() const
     * Return number of unexplored kernel configurations
     * @return Number of unexplored kernel configurations
     */
    uint64_t GetUnexploredConfigurationsCount() const;

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
    std::optional<uint64_t> m_Seed;
};

} // namespace ktt
