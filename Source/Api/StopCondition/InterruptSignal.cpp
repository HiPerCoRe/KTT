#include "Api/StopCondition/InterruptSignal.h"
#include "Api/Output/KernelResult.h"
#include "Utility/Logger/Logger.h"
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <string>
#include <csignal>

using namespace std;
using namespace ktt;

atomic<bool> InterruptSignal::m_shouldInterrupt = false;

InterruptSignal::~InterruptSignal()
{
    Unregister();
}

void InterruptSignal::Initialize(const uint64_t)
{
    if (m_registered)
    {
        ResetShouldInterrupt();
        Logger::LogWarning("Attempted double InterruptSignal initialization. Resetting instead.");
        return;
    }

    m_oldHandler = signal(SIGINT, HandleInterrupt);
    Logger::LogInfo("SIGINT handler registered");
    if (m_oldHandler == SIG_ERR)
    {
        Logger::LogError("Could not install SIGINT handler, tuning will not save on Ctrl-C.");
        Logger::LogError(string("Message: ") + strerror(errno));
        return;
    }
    if (m_oldHandler == SIG_IGN)
    {
        signal(SIGINT, SIG_IGN);  // Conventionally, if parent ignored signal, it should stay ignored
        Logger::LogWarning("SIGINT had been ignored before this call, not installing handler.");
        return;
    }
    m_oldShouldInterrupt = m_shouldInterrupt;
    m_shouldInterrupt = false;

    m_registered = true;
}

void InterruptSignal::Unregister()
{
    if (!m_registered || m_oldHandler == SIG_ERR)
    {
        return;
    }
    signal(SIGINT, m_oldHandler);
    m_registered = false;
    m_shouldInterrupt = m_oldShouldInterrupt;
}

bool InterruptSignal::IsFulfilled() const 
{
    return m_shouldInterrupt;
}

void InterruptSignal::ResetShouldInterrupt() 
{
    m_shouldInterrupt = false;
}

void InterruptSignal::HandleInterrupt(int)
{
    m_shouldInterrupt = true;
}

void InterruptSignal::Update(const KernelResult &) {}

string InterruptSignal::GetStatusString() const 
{
    return "Waiting for SIGINT...";
}
