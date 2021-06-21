/** @file KttException.h
* Error handling for KTT framework.
*/
#pragma once

#include <exception>
#include <string>

#include <Api/ExceptionReason.h>
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
      * @param message Holds message describing why the exception was thrown.
      * @param reason Reason why the exception was thrown.
      */
    KTT_API KttException(const std::string& message, const ExceptionReason reason = ExceptionReason::General);

    /** @fn const char* what() const noexcept
      * Returns message which describes why the exception was thrown.
      * @return Message which describes why the exception was thrown.
      */
    KTT_API const char* what() const noexcept override;

    /** @fn ExceptionReason GetReason() const
      * Returns reason why the exception was thrown.
      * @return Reason why the exception was thrown.
      */
    KTT_API ExceptionReason GetReason() const;

private:
    std::string m_Message;
    ExceptionReason m_Reason;
};

} // namespace ktt
