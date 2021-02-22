#include <Api/KttException.h>

namespace ktt
{

KttException::KttException(const std::string& message) :
    m_Message(message)
{}

const char* KttException::what() const
{
    return m_Message.c_str();
}

} // namespace ktt
