#ifdef KTT_API_CPP

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>

#if defined(_MSC_VER)
#include <windows.h>
#include <io.h>
#define popen _popen
#define pclose _pclose
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

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
#if defined(_MSC_VER)
        : m_Compiler("cl")
#else
        : m_Compiler("g++")
#endif
    {
        // Determine temporary directory
#if defined(_MSC_VER)
        char tempPath[MAX_PATH];
        DWORD result = GetTempPathA(MAX_PATH, tempPath);
        if (result == 0 || result > MAX_PATH)
        {
            m_TempDir = fs::path("C:\\temp") / "ktt_cpp_jit";
        }
        else
        {
            m_TempDir = fs::path(tempPath) / "ktt_cpp_jit";
        }
#else
        const char* tmpDir = std::getenv("TMPDIR");
        if (tmpDir == nullptr)
        {
            tmpDir = "/tmp";
        }
        m_TempDir = fs::path(tmpDir) / "ktt_cpp_jit";
#endif
        fs::create_directories(m_TempDir);
    }

    void SetCompiler(const std::string& compiler)
    {
        m_Compiler = compiler;
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
#if defined(_MSC_VER)
        fs::path libraryPath = m_TempDir / (kernelName + "_" + hash + ".dll");
#else
        fs::path libraryPath = m_TempDir / (kernelName + "_" + hash + ".so");
#endif

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

        // Compile with configured compiler
#if defined(_MSC_VER)
        // MSVC compiler command
        // /LD - create DLL
        // /MD - use multi-threaded DLL runtime
        // /std:c++17 - C++17 standard
        // /EHsc - exception handling
        // /Fe - output executable/library name
        std::string command = m_Compiler + " /LD /MD /std:c++17 /EHsc ";
        command += compilerOptions;
        command += " /Fe:";
        command += libraryPath.string();
        command += " ";
        command += sourcePath.string();
        // Link with OpenMP if the flag is present in compilerOptions
        if (compilerOptions.find("/openmp") != std::string::npos || compilerOptions.find("-fopenmp") != std::string::npos)
        {
            command += " /openmp";
        }
        command += " 2>&1";
#else
        // GCC/Clang compiler command
        // Only use essential flags here. Optimization flags (-O2, etc.) should be passed via compilerOptions.
        // Note: compilerOptions may include flags like -fopenmp that need to be passed to both compiler and linker
        std::string command = m_Compiler + " -shared -fPIC -std=c++11 ";
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
#endif

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
    std::string m_Compiler;

    KernelFunction LoadLibrary(const fs::path& libraryPath, const std::string& kernelName)
    {
#if defined(_MSC_VER)
        HMODULE handle = LoadLibraryA(libraryPath.string().c_str());
        if (!handle)
        {
            DWORD error = GetLastError();
            throw KttException("Failed to load DLL: " + std::to_string(error));
        }

        // The kernel function signature: void (*)(void**, size_t*)
        using RawFunc = void (*)(void**, size_t*);
        RawFunc rawFunc = reinterpret_cast<RawFunc>(GetProcAddress(handle, kernelName.c_str()));
        if (!rawFunc)
        {
            DWORD error = GetLastError();
            FreeLibrary(handle);
            throw KttException("Failed to find kernel symbol: " + kernelName + ", error: " + std::to_string(error));
        }

        // Wrap raw function into a std::function
        // Note: we need to keep handle alive for the lifetime of the function.
        // We'll capture handle in a shared_ptr to close later.
        auto handlePtr = std::shared_ptr<HMODULE>(new HMODULE(handle), [](HMODULE* h) { FreeLibrary(*h); delete h; });

        return [rawFunc, handlePtr](const std::vector<void*>& buffers, const std::vector<size_t>& sizes) {
            // Convert vectors to raw arrays (temporary)
            // The kernel signature is: void (*)(void** buffers, size_t* sizes)
            // buffers contains pointers to vector arguments
            // sizes contains buffer sizes followed by scalar argument values
            rawFunc(const_cast<void**>(buffers.data()), const_cast<size_t*>(sizes.data()));
        };
#else
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
#endif
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

void CppCompiler::SetCompiler(const std::string& compiler)
{
    m_Impl->SetCompiler(compiler);
}

} // namespace ktt

#endif // KTT_API_CPP
