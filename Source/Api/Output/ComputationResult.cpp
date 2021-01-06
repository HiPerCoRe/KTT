//#include <limits>
//
//#include <Api/Output/ComputationResult.h>
//
//namespace ktt
//{
//
//ComputationResult::ComputationResult() :
//    status(false),
//    duration(std::numeric_limits<uint64_t>::max()),
//    kernelName(""),
//    errorMessage("")
//{}
//
//ComputationResult::ComputationResult(const std::string& kernelName, const std::vector<ParameterPair>& configuration, const uint64_t duration,
//    const KernelCompilationData& compilationData, const KernelProfilingData& profilingData) :
//    status(true),
//    duration(duration),
//    kernelName(kernelName),
//    errorMessage(""),
//    configuration(configuration),
//    compilationData(compilationData),
//    profilingData(profilingData)
//{}
//
//ComputationResult::ComputationResult(const std::string& compositionName, const std::vector<ParameterPair>& configuration, const uint64_t duration,
//    const std::map<KernelId, KernelCompilationData>& compilationData, const std::map<KernelId, KernelProfilingData>& profilingData) :
//    status(true),
//    duration(duration),
//    kernelName(compositionName),
//    errorMessage(""),
//    configuration(configuration),
//    compositionCompilationData(compilationData),
//    compositionProfilingData(profilingData)
//{}
//
//ComputationResult::ComputationResult(const std::string& kernelName, const std::vector<ParameterPair>& configuration,
//    const std::string& errorMessage) :
//    status(false),
//    duration(std::numeric_limits<uint64_t>::max()),
//    kernelName(kernelName),
//    errorMessage(errorMessage),
//    configuration(configuration)
//{}
//
//bool ComputationResult::getStatus() const
//{
//    return status;
//}
//
//uint64_t ComputationResult::getDuration() const
//{
//    return duration;
//}
//
//const std::string& ComputationResult::getKernelName() const
//{
//    return kernelName;
//}
//
//const std::string& ComputationResult::getErrorMessage() const
//{
//    return errorMessage;
//}
//
//const std::vector<ParameterPair>& ComputationResult::getConfiguration() const
//{
//    return configuration;
//}
//
//const KernelCompilationData& ComputationResult::getCompilationData() const
//{
//    return compilationData;
//}
//
//const std::map<KernelId, KernelCompilationData>& ComputationResult::getCompositionCompilationData() const
//{
//    return compositionCompilationData;
//}
//
//const KernelProfilingData& ComputationResult::getProfilingData() const
//{
//    return profilingData;
//}
//
//const std::map<KernelId, KernelProfilingData>& ComputationResult::getCompositionProfilingData() const
//{
//    return compositionProfilingData;
//}
//
//} // namespace ktt
