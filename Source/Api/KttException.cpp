#include <Api/KttException.h>

namespace ktt
{

KttException::KttException(const std::string& message, const ExceptionReason reason) :
    m_Message(message),
    m_Reason(reason)
{}

const char* KttException::what() const noexcept
{
    return m_Message.c_str();
}

ExceptionReason KttException::GetReason() const
{
    return m_Reason;
}

} // namespace ktt
