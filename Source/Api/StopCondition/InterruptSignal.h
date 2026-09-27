#pragma once

#include <atomic>
#include <cstdint>
#include "Api/StopCondition/StopCondition.h"

namespace ktt
{

/** @class InterruptSignal
  * Class which implements a stop condition based on listening for
  * SIGINT. */
class InterruptSignal : public StopCondition
{
public:
    InterruptSignal() = default;
    ~InterruptSignal();

    virtual void Initialize(const uint64_t = 0) override;
    virtual bool IsFulfilled() const override;
    virtual void Update(const KernelResult &) override;
    virtual std::string GetStatusString() const override;

    /** @fn static void ResetShouldInterrupt()
      * Allows resetting of the interrupt flag without destroying
      * the object. */
    static void ResetShouldInterrupt();

private:
    void Unregister();
    static void HandleInterrupt(int signal);

    bool m_registered = false;
    void (*m_oldHandler)(int) = nullptr;
    bool m_oldShouldInterrupt = false;
    static std::atomic<bool> m_shouldInterrupt;
};

}