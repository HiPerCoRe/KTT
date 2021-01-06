#pragma once

#include <exception>
#include <string>

namespace ktt
{

class KttException : public std::exception
{
public:
    KttException(const std::string& message);

    const char* what() const override;

private:
    std::string m_Message;
};

} // namespace ktt
