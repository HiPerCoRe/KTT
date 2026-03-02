#pragma once

#ifdef KTT_API_CPP

#include <string>
#include <memory>
#include <functional>

namespace ktt
{

class CppCompiler
{
public:
    CppCompiler();
    ~CppCompiler();

    using KernelFunction = std::function<void(const std::vector<void*>&, const std::vector<size_t>&)>;

    // Compile kernel source and return a callable function.
    // kernelName: name of the kernel function in the source.
    // source: full kernel source code.
    // compilerOptions: additional options for the compiler.
    KernelFunction CompileKernel(const std::string& kernelName, const std::string& source, const std::string& compilerOptions);

    // Set the compiler executable to use (e.g., "g++", "clang++").
    // Default is "g++".
    void SetCompiler(const std::string& compiler);

private:
    class Impl;
    std::unique_ptr<Impl> m_Impl;
};

} // namespace ktt

#endif // KTT_API_CPP
