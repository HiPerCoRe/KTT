#ifdef KTT_API_CPP

#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

#include <Api/KttException.h>
#include <ComputeEngine/Cpp/CppCompiler.h>
#include <Utility/Logger/Logger.h>
#include <Utility/StringUtility.h>

namespace ktt
{

namespace fs = std::filesystem;

class CppCompiler::Impl
{
public:
    Impl()
    {
        // Determine temporary directory
        const char* tmpDir = std::getenv("TMPDIR");
        if (tmpDir == nullptr)
        {
            tmpDir = "/tmp";
        }
        m_TempDir = fs::path(tmpDir) / "ktt_cpp_jit";
        fs::create_directories(m_TempDir);
    }

    ~Impl()
    {
        // Cleanup temporary files? Could keep them for caching.
        // For now, we leave them.
    }

    KernelFunction CompileKernel(const std::string& kernelName, const std::string& source, const std::string& compilerOptions)
    {
        // Create a unique filename based on kernel name and source hash
        std::string hash = std::to_string(std::hash<std::string>{}(source));
        fs::path sourcePath = m_TempDir / (kernelName + "_" + hash + ".cpp");
        fs::path libraryPath = m_TempDir / (kernelName + "_" + hash + ".so");

        // Check if library already exists (caching)
        if (fs::exists(libraryPath))
        {
            Logger::LogDebug("Loading cached kernel library: " + libraryPath.string());
            return LoadLibrary(libraryPath, kernelName);
        }

        // Write source to file
        std::ofstream sourceFile(sourcePath);
        if (!sourceFile)
        {
            throw KttException("Failed to create temporary source file: " + sourcePath.string());
        }
        sourceFile << source;
        sourceFile.close();

        // Compile with g++
        // Only use essential flags here. Optimization flags (-O2, etc.) should be passed via compilerOptions.
        // Note: compilerOptions may include flags like -fopenmp that need to be passed to both compiler and linker
        std::string command = "g++ -shared -fPIC -std=c++11 ";
        command += compilerOptions;
        command += " -o ";
        command += libraryPath.string();
        command += " ";
        command += sourcePath.string();
        // Link with OpenMP if the flag is present in compilerOptions
        if (compilerOptions.find("-fopenmp") != std::string::npos)
        {
            command += " -fopenmp";
        }
        command += " 2>&1";

        Logger::LogDebug("Compiling kernel with command: " + command);

        FILE* pipe = popen(command.c_str(), "r");
        if (!pipe)
        {
            throw KttException("Failed to invoke compiler");
        }

        char buffer[128];
        std::string output;
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
        {
            output += buffer;
        }
        int status = pclose(pipe);

        if (status != 0)
        {
            Logger::LogError("Compilation failed:\n" + output);
            throw KttException("Kernel compilation failed: " + output, ExceptionReason::CompilerError);
        }

        Logger::LogDebug("Compilation succeeded");

        // Load library
        return LoadLibrary(libraryPath, kernelName);
    }

private:
    fs::path m_TempDir;

    KernelFunction LoadLibrary(const fs::path& libraryPath, const std::string& kernelName)
    {
        void* handle = dlopen(libraryPath.c_str(), RTLD_LAZY);
        if (!handle)
        {
            throw KttException("Failed to load shared library: " + std::string(dlerror()));
        }

        // The kernel function signature: void (*)(void**, size_t*)
        using RawFunc = void (*)(void**, size_t*);
        RawFunc rawFunc = reinterpret_cast<RawFunc>(dlsym(handle, kernelName.c_str()));
        if (!rawFunc)
        {
            dlclose(handle);
            throw KttException("Failed to find kernel symbol: " + kernelName);
        }

        // Wrap raw function into a std::function
        // Note: we need to keep handle alive for the lifetime of the function.
        // We'll capture handle in a shared_ptr to close later.
        auto handlePtr = std::shared_ptr<void>(handle, [](void* h) { dlclose(h); });

        return [rawFunc, handlePtr](const std::vector<void*>& buffers, const std::vector<size_t>& sizes) {
            // Convert vectors to raw arrays (temporary)
            // The kernel signature is: void (*)(void** buffers, size_t* sizes)
            // buffers contains pointers to vector arguments
            // sizes contains buffer sizes followed by scalar argument values
            rawFunc(const_cast<void**>(buffers.data()), const_cast<size_t*>(sizes.data()));
        };
    }
};

CppCompiler::CppCompiler() :
    m_Impl(std::make_unique<Impl>())
{}

CppCompiler::~CppCompiler() = default;

CppCompiler::KernelFunction CppCompiler::CompileKernel(const std::string& kernelName, const std::string& source,
    const std::string& compilerOptions)
{
    return m_Impl->CompileKernel(kernelName, source, compilerOptions);
}

} // namespace ktt

#endif // KTT_API_CPP
