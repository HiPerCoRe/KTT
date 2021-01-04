/** @file searcher.h
  * Interface for implementing kernel configuration searchers.
  */
#pragma once

#include <api/computation_result.h>
#include <api/kernel_configuration.h>
#include <ktt_platform.h>

namespace ktt
{

/** @class Searcher
  * Class which is used to decide which kernel configuration will be run next during the kernel tuning process.
  */
class KTT_API Searcher
{
public:
    /** @fn  virtual ~Searcher() = default
      * Searcher destructor. Inheriting class can override destructor with custom implementation. Default implementation is provided by KTT framework.
      */
    virtual ~Searcher() = default;

    /** @fn virtual void onInitialize()
      * Called after searcher is initialized with kernel configurations. Custom searcher parameters should be initialized here.
      */
    virtual void onInitialize();

    /** @fn virtual void onReset()
      * Called before searcher is reset to initial state and configurations are removed. Custom searcher parameters should be reset here.
      */
    virtual void onReset();

    /** @fn virtual void calculateNextConfiguration(const ComputationResult& previousResult) = 0
      * Decides which configuration will be run next. Called each time after getNextConfiguration method if unexplored count is greater than zero.
      * @param previousResult Computation result from last tested configuration. See ComputationResult for more information.
      */
    virtual void calculateNextConfiguration(const ComputationResult& previousResult) = 0;

    /** @fn virtual const KernelConfiguration& getNextConfiguration() const = 0
      * Returns kernel configuration which will be run next.
      * @return Configuration which will be run next.
      */
    virtual const KernelConfiguration& getNextConfiguration() const = 0;

    /** @fn virtual size_t getUnexploredConfigurationCount() const = 0
      * Returns number of unexplored kernel configurations.
      * @return Number of unexplored configurations.
      */
    virtual size_t getUnexploredConfigurationCount() const = 0;

    /** @fn Searcher()
      * Default searcher constructor. Should be called from inheriting searcher's constructor.
      */
    Searcher();

    /** @fn const std::vector<KernelConfiguration>& getConfigurations() const
      * Returns all configurations which can be explored for corresponding kernel.
      * @return All configurations which can be explored.
      */
    const std::vector<KernelConfiguration>& getConfigurations() const;

    /** @fn bool isInitialized() const
      * Returns whether searcher is initialized.
      * @return True if searcher is initialized, false otherwise.
      */
    bool isInitialized() const;

    /** @fn void initialize(const std::vector<KernelConfiguration>& configurations)
      * Initializes searcher with all configurations which can be explored for corresponding kernel.
      * @param configurations Vector of all valid kernel configurations. See KernelConfiguration for more information.
      */
    void initialize(const std::vector<KernelConfiguration>& configurations);

    /** @fn void reset()
      * Resets searcher to initial state and clears configurations.
      */
    void reset();

private:
    const std::vector<KernelConfiguration>* configurations;
};

} // namespace ktt
