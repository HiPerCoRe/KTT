#ifdef KTT_API_CPP

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <system_error>
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
        // Determine base temporary directory.
        fs::path baseDir;
#if defined(_MSC_VER)
        char tempPath[MAX_PATH];
        DWORD result = GetTempPathA(MAX_PATH, tempPath);
        baseDir = (result == 0 || result > MAX_PATH) ? fs::path("C:\\temp") : fs::path(tempPath);
        const auto processId = GetCurrentProcessId();
#else
        const char* tmpDir = std::getenv("TMPDIR");
        if (tmpDir == nullptr)
        {
            tmpDir = "/tmp";
        }
        baseDir = fs::path(tmpDir);
        const auto processId = getpid();
#endif
        // The compiled-kernel cache is deliberately process-private and does not survive a KTT
        // instance: a fresh process always recompiles, so toolchain updates, changed include files
        // and other external changes are always picked up. Using a per-process directory (instead of
        // wiping a shared one at startup) also avoids clobbering the cache of another KTT process
        // running concurrently on the same machine.
        m_TempDir = baseDir / ("ktt_cpp_jit_" + std::to_string(processId));
        std::error_code ignored;
        fs::remove_all(m_TempDir, ignored);
        fs::create_directories(m_TempDir);
    }

    void SetCompiler(const std::string& compiler)
    {
        m_Compiler = compiler;
    }

    void ClearCache()
    {
        std::error_code ignored;
        fs::remove_all(m_TempDir, ignored);
        fs::create_directories(m_TempDir);
    }

    ~Impl()
    {
        // The cache is process-private and must not outlive the instance.
        std::error_code ignored;
        fs::remove_all(m_TempDir, ignored);
    }

    KernelFunction CompileKernel(const std::string& kernelName, const std::string& source,
        const std::string& compilerOptions, const std::string& cacheKey)
    {
        // The cache key is the kernel's canonical identity (KernelComputeData::GetUniqueIdentifier) -
        // the same key used by the in-memory kernel cache. Because this on-disk cache is process-private
        // and wiped per instance, that identity is sufficient: within a single instance the kernel
        // source, compiler executable and static compiler options are fixed, so they need not be
        // re-encoded into the cache key here.
        std::string hash = std::to_string(std::hash<std::string>{}(cacheKey));
        fs::path sourcePath = m_TempDir / (kernelName + "_" + hash + ".cpp");
#if defined(_MSC_VER)
        fs::path libraryPath = m_TempDir / (kernelName + "_" + hash + ".dll");
#else
        fs::path libraryPath = m_TempDir / (kernelName + "_" + hash + ".so");
#endif

        // Reuse the cached library if present. The atomic publish below guarantees that any file at
        // libraryPath is a complete, fully-written library.
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

        // Compile to a unique temporary path and atomically move it into place on success. This
        // guarantees that libraryPath only ever appears as a complete, fully-written library; an
        // interrupted, killed or failed compile then leaves an orphan temp file instead of a
        // truncated .so/.dll that a later run would mistake for a valid cache entry.
#if defined(_MSC_VER)
        const auto processId = GetCurrentProcessId();
#else
        const auto processId = getpid();
#endif
        fs::path tempLibraryPath = libraryPath;
        tempLibraryPath += ".tmp." + std::to_string(processId);

        // Compile with configured compiler
        std::string command = m_Compiler + " " + compilerOptions;

#if defined(_MSC_VER)
        // /LD - create DLL
        // /MD - use multi-threaded DLL runtime
        // /EHsc - exception handling
        command += " /LD /MD /EHsc";

        // /Fe - output name
        command += " /Fe:" + tempLibraryPath.string();

        // Link with OpenMP if the flag is present in compilerOptions
        if (compilerOptions.find("/openmp") != std::string::npos)
        {
            command += " /openmp";
        }
#else
        command += " -shared -fPIC";
        command += " -o " + tempLibraryPath.string();
        // Note: compilerOptions may include flags like -fopenmp that need to be passed to both compiler and linker
        // Link with OpenMP if the flag is present in compilerOptions
        if (compilerOptions.find("-fopenmp") != std::string::npos)
        {
            command += " -fopenmp";
        }
#endif

        command += " " + sourcePath.string();
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
            // Remove any partial output so it cannot be mistaken for a valid cache entry later.
            std::error_code ignored;
            fs::remove(tempLibraryPath, ignored);
            Logger::LogError("Compilation failed:\n" + output);
            throw KttException("Kernel compilation failed: " + output, ExceptionReason::CompilerError);
        }

        // Atomically publish the freshly compiled library so concurrent or future runs never observe
        // a partially-written file at libraryPath.
        std::error_code renameError;
        fs::rename(tempLibraryPath, libraryPath, renameError);
        if (renameError)
        {
            std::error_code ignored;
            fs::remove(tempLibraryPath, ignored);
            throw KttException("Failed to finalize compiled kernel library " + libraryPath.string() + ": "
                + renameError.message());
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
    const std::string& compilerOptions, const std::string& cacheKey)
{
    return m_Impl->CompileKernel(kernelName, source, compilerOptions, cacheKey);
}

void CppCompiler::SetCompiler(const std::string& compiler)
{
    m_Impl->SetCompiler(compiler);
}

void CppCompiler::ClearCache()
{
    m_Impl->ClearCache();
}

} // namespace ktt

#endif // KTT_API_CPP
