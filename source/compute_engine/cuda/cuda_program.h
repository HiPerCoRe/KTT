#pragma once

#include <algorithm>
#include <cstring>
#include <regex>
#include <string>
#include <vector>
#include "cuda.h"
#include "nvrtc.h"
#include "cuda_utility.h"

namespace ktt
{

class CUDAProgram
{
public:
    explicit CUDAProgram(const std::string& source) :
        source(source)
    {
        auto sourcePointer = &source[0];
        checkCUDAError(nvrtcCreateProgram(&program, sourcePointer, nullptr, 0, nullptr, nullptr), "nvrtcCreateProgram");
    }

    ~CUDAProgram()
    {
        checkCUDAError(nvrtcDestroyProgram(&program), "nvrtcDestroyProgram");
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

            std::transform(individualOptions.begin(), individualOptions.end(), std::back_inserter(individualOptionsChar),
                [](const std::string& sourceString)
                {
                    // making a copy is necessary, because nvrtc expects the options to be in continuous memory block
                    char* result = new char[sourceString.size() + 1];
                    std::memcpy(result, sourceString.c_str(), sourceString.size() + 1);
                    return result; 
                });
        }

        nvrtcResult result = nvrtcCompileProgram(program, static_cast<int>(individualOptionsChar.size()), individualOptionsChar.data());

        for (size_t i = 0; i < individualOptionsChar.size(); i++)
        {
            delete[] individualOptionsChar.at(i);
        }

        std::string buildInfo = getBuildInfo();
        checkCUDAError(result, buildInfo);
    }

    std::string getPtxSource() const
    {
        size_t size;
        checkCUDAError(nvrtcGetPTXSize(program, &size), "nvrtcGetPTXSize");
        
        char result[size+1];
        checkCUDAError(nvrtcGetPTX(program, result), "nvrtcGetPTX");
        return std::string(result);
    }

    std::string getBuildInfo() const
    {
        size_t infoSize;
        checkCUDAError(nvrtcGetProgramLogSize(program, &infoSize), "nvrtcGetProgramLogSize");
        char infoCstr[infoSize+1];
        checkCUDAError(nvrtcGetProgramLog(program, infoCstr), "nvrtcGetProgramLog");

        return std::string(infoCstr);
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
