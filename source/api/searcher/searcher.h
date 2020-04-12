/** @file searcher.h
  * Interface for implementing kernel configuration searchers.
  */
#pragma once

#include <api/computation_result.h>
#include <api/kernel_configuration.h>

namespace ktt
{

/** @class Searcher
  * Class which is used to decide which kernel configuration will be run next during the the tuning process.
  */
class Searcher
{
public:
    /** @fn  virtual ~Searcher() = default
      * Searcher destructor. Inheriting class can override destructor with custom implementation. Default implementation is provided by KTT framework.
      */
    virtual ~Searcher() = default;

    /** @fn virtual void initializeConfigurations(const std::vector<KernelConfiguration>& configurations) = 0
      * Initializes searcher with all configurations which can be explored for corresponding kernel.
      * @param configurations Vector of all valid kernel configurations. See KernelConfiguration for more information.
      */
    virtual void initializeConfigurations(const std::vector<KernelConfiguration>& configurations) = 0;

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

    /** @fn virtual bool isInitialized() const = 0
      * Returns whether searcher is initialized.
      * @return True if searcher is initialized, false otherwise.
      */
    virtual bool isInitialized() const = 0;

    /** @fn virtual void reset() = 0
      * Resets searcher to initial state and clears configurations.
      */
    virtual void reset() = 0;
};

} // namespace ktt
