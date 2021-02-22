/** @file KttException.h
* Error handling for KTT framework.
*/
#pragma once

#include <exception>
#include <string>

#include <KttPlatform.h>

namespace ktt
{

class KttException : public std::exception
{
public:
    KTT_API KttException(const std::string& message);

    KTT_API const char* what() const override;

private:
    std::string m_Message;
};

} // namespace ktt
