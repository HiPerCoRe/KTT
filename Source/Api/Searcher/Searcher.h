/** @file Searcher.h
  * Interface for implementing kernel configuration searchers.
  */
#pragma once

#include <Api/Output/KernelResult.h>
#include <Api/KernelConfiguration.h>
#include <KttPlatform.h>

namespace ktt
{

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
      * Decides which configuration will be run next. Called after GetCurrentConfiguration method if unexplored count
      * is greater than zero.
      * @param previousResult Result from the last tested configuration. See KernelResult for more information.
      */
    virtual void CalculateNextConfiguration(const KernelResult& previousResult) = 0;

    /** @fn virtual const KernelConfiguration& GetCurrentConfiguration() const = 0
      * Returns current kernel configuration.
      * @return Current configuration.
      */
    virtual const KernelConfiguration& GetCurrentConfiguration() const = 0;

    /** @fn virtual uint64_t GetUnexploredConfigurationCount() const = 0
      * Returns number of unexplored kernel configurations.
      * @return Number of unexplored configurations.
      */
    virtual uint64_t GetUnexploredConfigurationCount() const = 0;

    /** @fn Searcher()
      * Default searcher constructor. Should be called from inheriting searcher's constructor.
      */
    Searcher();

    /** @fn const std::vector<KernelConfiguration>& GetConfigurations() const
      * Returns all configurations which can be explored for corresponding kernel.
      * @return All configurations which can be explored.
      */
    const std::vector<KernelConfiguration>& GetConfigurations() const;

    /** @fn bool IsInitialized() const
      * Returns whether searcher is initialized.
      * @return True if searcher is initialized, false otherwise.
      */
    bool IsInitialized() const;

    /** @fn void Initialize(const std::vector<KernelConfiguration>& configurations)
      * Initializes searcher with all configurations which can be explored for corresponding kernel.
      * @param configurations Vector of all valid kernel configurations. See KernelConfiguration for more information.
      */
    void Initialize(const std::vector<KernelConfiguration>& configurations);

    /** @fn void Reset()
      * Resets searcher to initial state and clears configurations.
      */
    void Reset();

private:
    const std::vector<KernelConfiguration>* m_Configurations;
};

} // namespace ktt
