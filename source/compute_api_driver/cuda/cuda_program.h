#pragma once

#include <regex>
#include <string>
#include <vector>

#include "cuda.h"
#include "nvrtc.h"
#include "cuda_utility.h"

namespace ktt
{

class CudaProgram
{
public:
    explicit CudaProgram(const std::string source) :
        source(source)
    {
        auto sourcePointer = &source[0];
        checkCudaError(nvrtcCreateProgram(&program, sourcePointer, nullptr, 0, nullptr, nullptr), std::string("nvrtcCreateProgram"));
    }

    ~CudaProgram()
    {
        checkCudaError(nvrtcDestroyProgram(&program), std::string("nvrtcDestroyProgram"));
    }

    void build(const std::string& compilerOptions)
    {
        std::vector<const char*> individualOptionsChar;

        if (compilerOptions != std::string(""))
        {
            std::vector<std::string> individualOptions;
            std::regex separator(" ");
            std::sregex_token_iterator iterator(compilerOptions.begin(), compilerOptions.end(), separator, -1);
            std::copy(iterator, std::sregex_token_iterator(), std::back_inserter(individualOptions));

            for (const auto& option : individualOptions)
            {
                individualOptionsChar.push_back(&option[0]);
            }
        }

        nvrtcResult result = nvrtcCompileProgram(program, static_cast<int>(individualOptionsChar.size()), individualOptionsChar.data());
        std::string buildInfo = getBuildInfo();
        checkCudaError(result, buildInfo);
    }

    std::string getPtxSource() const
    {
        size_t size;
        checkCudaError(nvrtcGetPTXSize(program, &size), std::string("nvrtcGetPTXSize"));
        
        std::string result(size, ' ');
        checkCudaError(nvrtcGetPTX(program, &result[0]), std::string("nvrtcGetPTX"));
        return result;
    }

    std::string getBuildInfo() const
    {
        size_t infoSize;
        checkCudaError(nvrtcGetProgramLogSize(program, &infoSize), std::string("nvrtcGetProgramLogSize"));
        std::string infoString(infoSize, ' ');
        checkCudaError(nvrtcGetProgramLog(program, &infoString[0]), std::string("nvrtcGetProgramLog"));

        return infoString;
    }

    std::string getSource() const
    {
        return source;
    }

    nvrtcProgram getProgram() const
    {
        return program;
    }

private:
    std::string source;
    nvrtcProgram program;
};

} // namespace ktt
