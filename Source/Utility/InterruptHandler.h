#pragma once

#include <atomic>

class InterruptHandler
{
public:
    InterruptHandler() = default;
    ~InterruptHandler();

    void Register();
    static bool GetShouldInterrupt();
    static void ResetShouldInterrupt();

private:
    void Unregister();
    static void HandleInterrupt(int signal);

    bool m_registered = false;
    void (*m_oldHandler)(int) = nullptr;
    bool m_oldShouldInterrupt = false;
    static std::atomic<bool> m_shouldInterrupt;
};
