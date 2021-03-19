/** @file Searcher.h
  * Interface for implementing kernel configuration searchers.
  */
#pragma once

#include <Api/Configuration/KernelConfiguration.h>
#include <Api/Output/KernelResult.h>
#include <KttPlatform.h>

namespace ktt
{

class ConfigurationTree;

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
      * Called after searcher is initialized with kernel configurations. Custom searcher parameters should be initialized here.
      */
    virtual void OnInitialize();

    /** @fn virtual void OnReset()
      * Called before searcher is reset to initial state and configurations are removed. Custom searcher parameters
      * should be reset here.
      */
    virtual void OnReset();

    /** @fn virtual void CalculateNextConfiguration(const KernelResult& previousResult) = 0
      * Decides which configuration will be run next. Called after GetCurrentConfiguration method if there are remaining
      * unexplored configurations.
      * @param previousResult Result from the last tested configuration. See KernelResult for more information.
      */
    virtual void CalculateNextConfiguration(const KernelResult& previousResult) = 0;

    /** @fn virtual KernelConfiguration GetCurrentConfiguration() const = 0
      * Returns current kernel configuration.
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

    /** @fn uint64_t GetConfigurationsCount() const
      * Returns total number of valid kernel configurations.
      * @return Number of valid kernel configurations.
      */
    uint64_t GetConfigurationsCount() const;

    /** @fn bool IsInitialized() const
      * Returns whether searcher is initialized.
      * @return True if searcher is initialized, false otherwise.
      */
    bool IsInitialized() const;

    /** @fn void Initialize(const ConfigurationTree& tree)
      * Initializes searcher with the tree of configurations which can be explored for corresponding kernel.
      * @param tree Tree of configurations which can be explored
      */
    void Initialize(const ConfigurationTree& tree);

    /** @fn void Reset()
      * Resets searcher to initial state and clears configuration tree.
      */
    void Reset();

private:
    const ConfigurationTree* m_Tree;
};

} // namespace ktt
