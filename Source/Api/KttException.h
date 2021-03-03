/** @file KttException.h
* Error handling for KTT framework.
*/
#pragma once

#include <exception>
#include <string>

#include <KttPlatform.h>

namespace ktt
{

/** @class KttException
  * Exception thrown when invalid usage of KTT API is detected.
  */
class KttException : public std::exception
{
public:
    /** @fn KttException(const std::string& message)
      * Creates new exception with the specified error message.
      * @param message Holds reason why the exception was thrown.
      */
    KTT_API KttException(const std::string& message);

    /** @fn const char* what() const
      * Returns reason why the exception was thrown.
      * @return Reason why the exception was thrown.
      */
    KTT_API const char* what() const override;

private:
    std::string m_Message;
};

} // namespace ktt
