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

InterruptHandler::~InterruptHandler()
{
    Unregister();
}

void InterruptHandler::Register()
{
    if (m_registered)
    {
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
        Logger::LogWarning("SIGINT was ignored before this call, not installing handler.");
        return;
    }
    m_oldShouldInterrupt = m_shouldInterrupt;
    m_shouldInterrupt = false;

    m_registered = true;
}

void InterruptHandler::Unregister()
{
    if (!m_registered || m_oldHandler == SIG_ERR)
    {
        return;
    }
    signal(SIGINT, m_oldHandler);
    m_registered = false;
    m_shouldInterrupt = m_oldShouldInterrupt;
}

bool InterruptHandler::GetShouldInterrupt()
{
    return m_shouldInterrupt;
}

void InterruptHandler::ResetShouldInterrupt() 
{
    m_shouldInterrupt = false;
}

void InterruptHandler::HandleInterrupt(int)
{
    m_shouldInterrupt = true;
}
