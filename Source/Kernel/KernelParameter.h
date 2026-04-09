#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include <Api/Configuration/ParameterPair.h>
#include <KttTypes.h>

namespace ktt
{

class KernelParameter
{
public:
    explicit KernelParameter(const std::string& name, const std::vector<ParameterValue>& values, const std::string& group,
        const bool isCompilerParameter);
    explicit KernelParameter(const std::string& name, const ParameterValueType valueType, const std::string& valueScript,
        const std::string& group, const bool isCompilerParameter);

    const std::string& GetName() const;
    const std::string& GetGroup() const;
    size_t GetValuesCount() const;
    const std::vector<ParameterValue>& GetValues() const;
    ParameterValueType GetValueType() const;
    ParameterPair GeneratePair(const size_t valueIndex) const;
    std::vector<ParameterPair> GeneratePairs() const;
    bool IsCompilerParameter() const;
    bool IsTuned() const;

    void SetTuning(const bool isTuned) const;

    bool operator==(const KernelParameter& other) const;
    bool operator!=(const KernelParameter& other) const;
    bool operator<(const KernelParameter& other) const;

private:
    std::string m_Name;
    std::string m_Group;
    std::vector<ParameterValue> m_Values;
    bool m_IsCompilerParameter;

    // If the parameter is not tuned with the others, only its first value will be used in the configurations.
    // Used for compiler options that will be tuned separately after standard tuning.
    mutable bool m_IsTuned;

    static std::vector<ParameterValue> GetValuesFromScript(const ParameterValueType valueType, const std::string& valueScript);
};

} // namespace ktt
