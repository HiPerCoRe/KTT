#include <atomic>

class InterruptHandler 
{
public:
    static bool GetShouldInterrupt();
    static void RegisterHandler();
    static void UnregisterHandler();
private:
    static void HandleInterrupt(int signal);
    static void (*m_oldHandler)(int);
    static std::atomic<bool> m_shouldInterrupt;
};