#pragma once

#include <vector>
#include <gpu_perf_api/GPUPerfAPI.h>
#include <api/kernel_profiling_data.h>
#include <compute_engine/opencl/gpa_profiling_context.h>
#include <compute_engine/opencl/opencl_utility.h>

namespace ktt
{

class GPAProfilingInstance
{
public:
    explicit GPAProfilingInstance(GPAFunctionTable& gpaFunctions, GPAProfilingContext& context) :
        gpaFunctions(gpaFunctions),
        context(context.getContext()),
        counters(context.getCounters()),
        sampleId(context.generateNewSampleId()),
        currentPassIndex(0)
    {
        checkGPAError(gpaFunctions.GPA_CreateSession(this->context, GPA_SESSION_SAMPLE_TYPE_DISCRETE_COUNTER, &session), "GPA_CreateSession");
        activateCounters();

        checkGPAError(gpaFunctions.GPA_BeginSession(session), "GPA_BeginSession");
        checkGPAError(gpaFunctions.GPA_GetPassCount(session, &totalPassCount), "GPA_GetPassCount");
    }

    ~GPAProfilingInstance()
    {
        checkGPAError(gpaFunctions.GPA_EndSession(session), "GPA_EndSession");
        checkGPAError(gpaFunctions.GPA_DeleteSession(session), "GPA_DeleteSession");
    }

    void updateState()
    {
        ++currentPassIndex;
    }

    GPA_SessionId getSession() const
    {
        return session;
    }

    gpa_uint32 getSampleId() const
    {
        return sampleId;
    }

    gpa_uint32 getCurrentPassIndex() const
    {
        return currentPassIndex;
    }

    gpa_uint32 getRemainingPassCount() const
    {
        if (currentPassIndex > totalPassCount)
        {
            return 0;
        }

        return totalPassCount - currentPassIndex;
    }

    KernelProfilingData generateProfilingData()
    {
        if (getRemainingPassCount() > 0)
        {
            throw std::runtime_error("GPA profiling API error: Insufficient number of completed kernel runs for profiling counter collection");
        }

        KernelProfilingData result;
        size_t sampleSizeInBytes;
        checkGPAError(gpaFunctions.GPA_GetSampleResultSize(session, sampleId, &sampleSizeInBytes), "GPA_GetSampleResultSize");

        std::vector<uint64_t> sampleData(sampleSizeInBytes / sizeof(uint64_t));
        checkGPAError(gpaFunctions.GPA_GetSampleResult(session, sampleId, sampleSizeInBytes, sampleData.data()), "GPA_GetSampleResult");

        gpa_uint32 enabledCount;
        checkGPAError(gpaFunctions.GPA_GetNumEnabledCounters(session, &enabledCount), "GPA_GetNumEnabledCounters");

        for (gpa_uint32 i = 0; i < enabledCount; ++i)
        {
            gpa_uint32 counterIndex;
            checkGPAError(gpaFunctions.GPA_GetEnabledIndex(session, i, &counterIndex), "GPA_GetEnabledIndex");

            GPA_Data_Type counterType;
            checkGPAError(gpaFunctions.GPA_GetCounterDataType(context, counterIndex, &counterType), "GPA_GetCounterDataType");

            const char* counterName;
            checkGPAError(gpaFunctions.GPA_GetCounterName(context, counterIndex, &counterName), "GPA_GetCounterName");

            ProfilingCounterValue value;
            ProfilingCounterType type;

            switch (counterType)
            {
            case GPA_DATA_TYPE_FLOAT64:
                value.doubleValue = static_cast<double>(sampleData[i]);
                type = ProfilingCounterType::Double;
                break;
            case GPA_DATA_TYPE_UINT64:
                value.uintValue = sampleData[i];
                type = ProfilingCounterType::UnsignedInt;
                break;
            default:
                throw std::runtime_error("GPA profiling API error: Unknown counter type");
            }

            result.addCounter(KernelProfilingCounter(counterName, value, type));
        }

        return result;
    }

private:
    GPAFunctionTable& gpaFunctions;
    GPA_ContextId context;
    GPA_SessionId session;
    std::vector<std::string> counters;
    gpa_uint32 sampleId;
    gpa_uint32 currentPassIndex;
    gpa_uint32 totalPassCount;

    void activateCounters()
    {
        for (const auto& counter : counters)
        {
            checkGPAError(gpaFunctions.GPA_EnableCounterByName(session, counter.c_str()), "GPA_EnableCounterByName");
        }
    }
};

} // namespace ktt
