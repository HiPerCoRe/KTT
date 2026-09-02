#include "InterruptHandler.h"
#include "Utility/Logger/Logger.h"
#include <cassert>
#include <cerrno>
#include <cstring>
#include <string>
#include <csignal>

using namespace std;
using namespace ktt;

atomic<bool> InterruptHandler::m_shouldInterrupt = false;
void (*InterruptHandler::m_oldHandler)(int) = nullptr;

bool InterruptHandler::GetShouldInterrupt()
{
    return m_shouldInterrupt;
}

void InterruptHandler::RegisterHandler()
{
    m_oldHandler = signal(SIGINT, HandleInterrupt);
    Logger::LogInfo("SIGINT handler registered");
    if (m_oldHandler == SIG_ERR)
    {
        Logger::LogError("Could not install SIGINT handler, tuning will not save on Ctrl-C.");
        Logger::LogError(string("Message: ") + strerror(errno));
    }
    if (m_oldHandler == SIG_IGN)
    {
        signal(SIGINT, SIG_IGN);  // Conventionally, if parent ignored signal, it should stay ignored
        Logger::LogWarning("SIGINT was ignored before this call, not installing handler.");
    }
}

void InterruptHandler::UnregisterHandler()
{
    assert(m_oldHandler != nullptr || m_oldHandler == SIG_DFL);
    if (m_oldHandler != SIG_ERR)
    {
        signal(SIGINT, m_oldHandler);
    }
}

void InterruptHandler::HandleInterrupt(int)
{
    m_shouldInterrupt = true;
}