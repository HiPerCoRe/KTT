#pragma once

// Shared fixture for the default database location tests. Only included from within the
// #if defined(KTT_DATABASE) && !defined(_MSC_VER) guard of each test file: the tests change environment variables
// through the POSIX setenv/unsetenv API.

#include <cstdlib>
#include <optional>
#include <string>

namespace ktt::db::test
{

// Sets (or unsets) an environment variable for the lifetime of the object and restores the original value afterwards.
class ScopedEnvironmentVariable
{
public:
    ScopedEnvironmentVariable(const std::string& name, const std::optional<std::string>& value) :
        m_Name(name)
    {
        if (const char* original = std::getenv(name.c_str()); original != nullptr)
            m_Original = original;

        Apply(value);
    }

    ~ScopedEnvironmentVariable()
    {
        Apply(m_Original);
    }

    ScopedEnvironmentVariable(const ScopedEnvironmentVariable&) = delete;
    ScopedEnvironmentVariable& operator=(const ScopedEnvironmentVariable&) = delete;

private:
    std::string m_Name;
    std::optional<std::string> m_Original;

    void Apply(const std::optional<std::string>& value) const
    {
        if (value)
            setenv(m_Name.c_str(), value->c_str(), 1);
        else
            unsetenv(m_Name.c_str());
    }
};

} // namespace ktt::db::test
