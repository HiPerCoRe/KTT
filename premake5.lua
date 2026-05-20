-- Global configuration variables
cudaProjects = false
openClProjects = false
vulkanProjects = false
cppProjects = false

-- Helper functions to find and link compute API headers and libraries
function linkLibrariesAmd()
    local path = os.getenv("OCL_ROOT")
    
    if not path then
        return false
    end
    
    defines {"KTT_PLATFORM_AMD"}
    includedirs {"$(OCL_ROOT)/include"}
        
    if os.target() == "linux" then
        libdirs {"$(OCL_ROOT)/lib64"}
    else
        libdirs {"$(OCL_ROOT)/lib/x86_64"}
    end
    
    if _OPTIONS["no-opencl"] then
        return true
    end
    
    defines {"KTT_API_OPENCL"}
    defines {"CL_TARGET_OPENCL_VERSION=300"}
    links {"OpenCL"}
    openClProjects = true
    
    if _OPTIONS["profiling"] == "gpa" then
        defines {"KTT_PROFILING_GPA"}
        includedirs {"Libraries/GpuPerfApi-3.6/Include"}
        
        if os.target() == "linux" then
            libdirs {"Libraries/GpuPerfApi-3.6/Lib/Linux"}
        else
            -- One of the GPA headers includes Windows.h with evil min/max macros
            defines {"NOMINMAX"}
            libdirs {"Libraries/GpuPerfApi-3.6/Lib/Windows"}
        end
    elseif _OPTIONS["profiling"] == "gpa-legacy" then
        defines {"KTT_PROFILING_GPA_LEGACY"}
        includedirs {"Libraries/GpuPerfApi-3.3/Include"}
        
        if os.target() == "linux" then
            libdirs {"Libraries/GpuPerfApi-3.3/Lib/Linux"}
        else
            defines {"NOMINMAX"}
            libdirs {"Libraries/GpuPerfApi-3.3/Lib/Windows"}
        end
    end
    
    return true
end

function linkLibrariesIntel()
    local path = os.getenv("INTELOCLSDKROOT")
    
    if not path then
        return false
    end
    
    defines {"KTT_PLATFORM_INTEL"}
    includedirs {"$(INTELOCLSDKROOT)/include"}
        
    if os.target() == "linux" then
        libdirs {"$(INTELOCLSDKROOT)/lib64"}
    else
        libdirs {"$(INTELOCLSDKROOT)/lib/x64"}
    end
    
    if _OPTIONS["no-opencl"] then
        return true
    end
    
    defines {"KTT_API_OPENCL"}
    defines {"CL_TARGET_OPENCL_VERSION=300"}
    links {"OpenCL"}
    openClProjects = true
    
    return true
end

function linkLibrariesNvidia()
    local path = os.getenv("CUDA_PATH")
    
    if not path then
        return false
    end
    
    defines {"KTT_PLATFORM_NVIDIA"}
    includedirs {"$(CUDA_PATH)/include"}
        
    if os.target() == "linux" then
        libdirs {"$(CUDA_PATH)/lib64", "$(CUDA_PATH)/lib64/stubs"}
    else
        libdirs {"$(CUDA_PATH)/lib/x64", "$(CUDA_PATH)/lib/x64/stubs"}
    end
    
    if not _OPTIONS["no-opencl"] then
        defines {"KTT_API_OPENCL"}
        defines {"CL_TARGET_OPENCL_VERSION=300"}
        links {"OpenCL"}
        openClProjects = true
    end
        
    if not _OPTIONS["no-cuda"] then
        defines {"KTT_API_CUDA"}
        links {"cuda", "nvrtc"}
        cudaProjects = true
        
        if _OPTIONS["power-usage"] then
            defines {"KTT_POWER_USAGE_NVML"}
            links {"nvidia-ml"}
        end
        
        if _OPTIONS["profiling"] == "cupti-legacy" or _OPTIONS["profiling"] == "cupti" or _OPTIONS["power-usage"] then
            includedirs {"$(CUDA_PATH)/extras/CUPTI/include"}
            libdirs {"$(CUDA_PATH)/extras/CUPTI/lib64"}
            links {"cupti"}
        end
        
        if _OPTIONS["profiling"] == "cupti-legacy" then
            defines {"KTT_PROFILING_CUPTI_LEGACY"}
            
            if os.target() == "windows" then
                libdirs {"$(CUDA_PATH)/extras/CUPTI/libx64"}
            end
        elseif _OPTIONS["profiling"] == "cupti" then
            defines {"KTT_PROFILING_CUPTI"}
            links {"nvperf_host", "nvperf_target"}
        end
    end
        
    return true
end

function linkCpp()
    defines {"KTT_API_CPP"}
    cppProjects = true
    return true
end

-- Helper function to enable OpenMP support in a cross-platform manner
function enableOpenMP()
    filter "system:linux or system:macosx"
        buildoptions {"-fopenmp"}
        linkoptions {"-fopenmp"}
    filter "system:windows"
        buildoptions {"/openmp"}
    filter {}
end

function linkComputeLibraries()
    if _OPTIONS["platform"] then
        if _OPTIONS["platform"] == "amd" then
            return linkLibrariesAmd()
        elseif _OPTIONS["platform"] == "intel" then
            return linkLibrariesIntel()
        elseif _OPTIONS["platform"] == "nvidia" then
            return linkLibrariesNvidia()
        else
            error("The specified platform is unknown.")
        end
    end

    local retVal = false

    if _OPTIONS["cpp"] then
        linkCpp()
        retVal = true
    end

    if linkLibrariesAmd() then
        retVal = true
    end
    
    if linkLibrariesIntel() then
        retVal = true
    end
    
    if linkLibrariesNvidia() then
        retVal = true
    end
    
    return retVal
end

function linkVulkan()
    local path = os.getenv("VULKAN_SDK")
    
    if not path then
        return false
    end
    
    defines {"KTT_API_VULKAN"}
    includedirs {"Libraries/VulkanMemoryAllocator-2.3.0"}
    files {"Libraries/VulkanMemoryAllocator-2.3.0/**"}
    
    if os.target() == "linux" then
        includedirs {"$(VULKAN_SDK)/include"}
        libdirs {"$(VULKAN_SDK)/lib"}
    else
        includedirs {"$(VULKAN_SDK)/Include"}
        libdirs {"$(VULKAN_SDK)/Lib"}
    end
    
    links {"shaderc_shared"}
    
    if os.target() == "linux" then
        links {"vulkan"}
    else
        links {"vulkan-1"}
    end
    
    vulkanProjects = true
    return true
end

function linkPython()
    local pythonHeaders = os.getenv("PYTHON_HEADERS")
    local pythonLibrary = os.getenv("PYTHON_LIB")
    
    if not pythonHeaders or not pythonLibrary then
        return false
    end
    
    defines {"KTT_PYTHON"}
    includedirs {pythonHeaders, "Libraries/pybind11-3.0.1"}
    files {"Libraries/pybind11-3.0.1/**"}
    
    if os.target() == "windows" then
        pythonLibrary = pythonLibrary:gsub("\\", "/")
    end
    
    local libraryPath = path.getdirectory(pythonLibrary)
    libdirs {libraryPath}
    
    local libraryName = path.getbasename(pythonLibrary)
    
    if os.target() == "linux" and string.startswith(libraryName, "lib") then
        libraryName = libraryName:sub(4)
    end
    
    links {libraryName}
    return true
end

function linkAllLibraries()
    local librariesFound = linkComputeLibraries()
    
    -- Allow usage of KTT with only Vulkan or C++ if no other compute API was explicitly specified by user
    if not librariesFound and (not _OPTIONS["vulkan"] or _OPTIONS["platform"]) and not _OPTIONS["cpp"] then
        error("Compute API libraries were not found. Please ensure that path to the SDK is correctly set in the environment variables:\nOCL_ROOT for AMD\nINTELOCLSDKROOT for Intel\nCUDA_PATH for Nvidia\nor add parameter --cpp to build C++ backend")
    end
    
    if _OPTIONS["vulkan"] then
        local vulkanFound = linkVulkan()
        
        if not vulkanFound then
            error("Vulkan SDK was not found. Please ensure that path to the SDK is correctly set in the environment variables under VULKAN_SDK.")
        end
    end
    
    if _OPTIONS["python"] then
        local pythonFound = linkPython()
        
        if not pythonFound then
            error("Python installation was not found. Please ensure that paths to Python headers and Python library (including library name) are correctly set in the environment variables under PYTHON_HEADERS and PYTHON_LIB.")
        end
    end
end

-- Command line arguments definition
newoption
{
    trigger = "platform",
    value = "vendor",
    description = "Specifies platform for KTT library compilation",
    allowed =
    {
        {"amd", "AMD"},
        {"intel", "Intel"},
        {"nvidia", "Nvidia"}
    }
}

newoption
{
    trigger = "vulkan",
    description = "Enables compilation of Vulkan backend"
}

newoption
{
    trigger = "cpp",
    description = "Enables compilation of C++ backend (CPU execution)"
}

newoption
{
    trigger = "profiling",
    value = "library",
    description = "Enables compilation of kernel profiling functionality using specified library",
    allowed =
    {
        {"cupti", "Nvidia CUPTI for Volta, Turing and Ampere"},
        {"cupti-legacy", "Nvidia CUPTI for legacy GPUs (Volta and older)"},
        {"gpa", "AMD GPA for GCN 3.0 GPUs and newer"},
        {"gpa-legacy", "AMD GPA for GCN 5.0 GPUs and older"}
    }
}

newoption
{
    trigger = "power-usage",
    description = "Enables compilation of device power usage collection functionality"
}

newoption
{       
    trigger = "power-usage-mintime",
    description = "Sets how long is the kernel repeated to obtain reliable power measurement, in miliseconds. THIS IS AN EXPERIMENTAL FEATURE, enforcing repeating the kernel without cleaning or changing input/output."
}        

newoption
{
    trigger = "python",
    description = "Enables compilation of Python bindings"
}

newoption
{
    trigger = "tuning-loader",
    description = "Enables compilation of tuning loader and tuning launcher"
}

newoption
{
    trigger = "outdir",
    value = "path",
    description = "Specifies output directory for generated project files"
}

newoption
{
    trigger = "tests",
    description = "Enables compilation of unit tests"
}

newoption
{
    trigger = "no-cuda",
    description = "Disables compilation of CUDA backend (Nvidia platform only)"
}

newoption
{
    trigger = "no-opencl",
    description = "Disables compilation of OpenCL backend"
}

newoption
{
    trigger = "no-examples",
    description = "Disables compilation of examples"
}

newoption
{
    trigger = "reference-versions",
    description = "Enables compilation of reference versions of examples"
}

newoption
{
    trigger = "no-tutorials",
    description = "Disables compilation of tutorials"
}

-- Helper function to add example projects
function addExampleProject(name, kernelExt, apiDefine, useRefVersions, shouldEnableOpenMP)
    local projectName = name .. (useRefVersions and "Reference" or "") .. kernelExt

    local cppFiles
    if useRefVersions then
        cppFiles = {"Examples/ReferenceVersions/" .. name .. "/*.cpp"}
    else
        cppFiles = {"Examples/*.cpp", "Examples/" .. name .. "/*.cpp"}
    end

    project(projectName)
        kind "ConsoleApp"
        files {table.unpack(cppFiles)}
        includedirs {"Source"}
        defines {apiDefine}
        links {"ktt"}
        if shouldEnableOpenMP then
            enableOpenMP()
        end
end

-- Helper function to add OpenCL example with optional reference version
function addOpenClExample(name, enableOpenMP, noReference)
    addExampleProject(name, "OpenCl", "KTT_OPENCL_EXAMPLE", false, enableOpenMP)
    if _OPTIONS["reference-versions"] and not noReference then
        addExampleProject(name, "OpenCl", "KTT_OPENCL_EXAMPLE", true, enableOpenMP)
    end
end

-- Helper function to add CUDA example with optional reference version
function addCudaExample(name, enableOpenMP, noReference)
    addExampleProject(name, "Cuda", "KTT_CUDA_EXAMPLE", false, enableOpenMP)
    if _OPTIONS["reference-versions"] and not noReference then
        addExampleProject(name, "Cuda", "KTT_CUDA_EXAMPLE", true, enableOpenMP)
    end
end

-- Helper function to add C++ example with optional reference version
function addCppExample(name, enableOpenMP, noReference)
    addExampleProject(name, "Cpp", "KTT_CPP_EXAMPLE", false, enableOpenMP)
    if _OPTIONS["reference-versions"] and not noReference then
        addExampleProject(name, "Cpp", "KTT_CPP_EXAMPLE", true, enableOpenMP)
    end
end

-- Base example list (examples available for both OpenCL and CUDA)
baseExamples = {
    {"AtfCCSD"},
    {"AtfConvolution"},
    {"AtfGEMM"},
    {"AtfPRL"},
    {"Bicg"},
    {"ClTuneConvolution"},
    {"ClTuneGemm"},
    {"CoulombSum3d", true},      -- requires OpenMP
    {"Nbody"},
    {"Reduction"},
    {"Sort"},
    {"Sort2"},
    {"Transpose"},
    {"Dummy"},
    {"RodiniaHotspot", false, true},
    {"GemmBatch", false, true}
}

-- OpenCL-only examples
openClOnlyExamples = {
    {"Convolution3d"},
    {"CoulombSum2d"},
    {"Covariance"}
}

-- CUDA-only examples
cudaOnlyExamples = {
    {"KernelTunerConvolution"},
    {"KernelTunerPnpoly"},
    {"Microbenchmarks"}
}

-- C++ examples
cppExamples = {
    {"CoulombSum3d", true}   -- requires OpenMP
}

-- Project configuration
workspace "Ktt"
    local buildPath = "Build"
    
    if _OPTIONS["outdir"] then
        buildPath = _OPTIONS["outdir"]
    end
    
    configurations {"Release", "Debug"}
    platforms {"x86_64"}
    architecture "x86_64"
    
    location(buildPath)
    language "C++"
    cppdialect "C++17"
    warnings "Extra"
    
    filter "configurations:Debug"
        defines {"KTT_CONFIGURATION_DEBUG"}
        optimize "Off"
        symbols "On"
    
    filter "configurations:Release"
        defines {"KTT_CONFIGURATION_RELEASE"}
        optimize "Full"
        symbols "Off"
    
    filter "action:vs*"
        conformancemode "On"
        buildoptions {"/Zc:__cplusplus"}
    
    filter {}
    
    targetdir(buildPath .. "/%{cfg.platform}_%{cfg.buildcfg}")
    objdir(buildPath .. "/%{cfg.platform}_%{cfg.buildcfg}/obj")
    
-- Library configuration
project "Ktt"
    kind "SharedLib"
    
    files
    {
        "Source/**",
        "Libraries/CTPL-Ahajha/**",
        "Libraries/date-3/**",
        "Libraries/Json-3.9.1/**",
        "Libraries/pugixml-1.11.4/**"
    }
    
    includedirs
    {
        "Source",
        "Libraries/CTPL-Ahajha",
        "Libraries/date-3",
        "Libraries/Json-3.9.1",
        "Libraries/pugixml-1.11.4"
    }
    
    if _OPTIONS["python"] then
        if os.target() == "linux" then
            postbuildcommands {"{COPYFILE} %{cfg.targetdir}/libktt.so %{cfg.targetdir}/pyktt.so"}
        else
            postbuildcommands {"{COPYFILE} %{cfg.targetdir}/ktt.dll %{cfg.targetdir}/pyktt.pyd"}
        end     
    end
    
    defines {"KTT_LIBRARY"}
    targetname("ktt")
    linkAllLibraries()

-- Tuning loader and launcher
if _OPTIONS["tuning-loader"] then
    if not _OPTIONS["python"] then
        error("Tuning loader depends on KTT Python integration (specified with --python option).")
    end

project "KttTuningLoader"
    kind "SharedLib"
    
    files
    {
        "TuningLoader/**",
        "Libraries/Json-3.9.1/**",
        "Libraries/JsonSchemaValidator-2.1.0/**"
    }
    
    removefiles {"TuningLoader/TuningLauncher.cpp"}
    
    includedirs
    {
        "TuningLoader",
        "Libraries/Json-3.9.1",
        "Libraries/JsonSchemaValidator-2.1.0",
        "Source"
    }
    
    defines {"KTT_LOADER_LIBRARY"}
    links {"ktt"}

project "KttTuningLauncher"
    kind "ConsoleApp"
    files {"TuningLoader/TuningLauncher.cpp"}
    includedirs {"TuningLoader"}
    links {"KttTuningLoader"}
    
end -- _OPTIONS["tuning-loader"]

-- Tutorials configuration 
if not _OPTIONS["no-tutorials"] then

if openClProjects then

project "01InfoOpenCl"
    kind "ConsoleApp"
    files {"Tutorials/01ComputeApiInfo/ComputeApiInfoOpenCl.cpp"}
    includedirs {"Source"}
    links {"ktt"}

project "02KernelRunningOpenCl"
    kind "ConsoleApp"
    files {"Tutorials/02KernelRunning/KernelRunningOpenCl.cpp", "Tutorials/02KernelRunning/OpenClKernel.cl"}
    includedirs {"Source"}
    links {"ktt"}

project "03KernelTuningOpenCl"
    kind "ConsoleApp"
    files {"Tutorials/03KernelTuning/KernelTuningOpenCl.cpp", "Tutorials/03KernelTuning/OpenClKernel.cl"}
    includedirs {"Source"}
    links {"ktt"}

project "04CustomArgumentTypesOpenCl"
    kind "ConsoleApp"
    files {"Tutorials/04CustomArgumentTypes/CustomArgumentTypesOpenCl.cpp", "Tutorials/04CustomArgumentTypes/OpenClKernel.cl"}
    includedirs {"Source"}
    links {"ktt"}

project "05ComputeApiInitializerOpenCl"
    kind "ConsoleApp"
    files {"Tutorials/05ComputeApiInitializer/ComputeApiInitializerOpenCl.cpp", "Tutorials/05ComputeApiInitializer/OpenClKernel.cl"}
    includedirs {"Source"}
    links {"ktt"}
    linkComputeLibraries()

project "06VectorArgumentCustomizationOpenCl"
    kind "ConsoleApp"
    files {"Tutorials/06VectorArgumentCustomization/VectorArgumentCustomizationOpenCl.cpp", "Tutorials/06VectorArgumentCustomization/OpenClKernel.cl"}
    includedirs {"Source"}
    links {"ktt"}
    
end -- openClProjects

if cudaProjects then

project "01InfoCuda"
    kind "ConsoleApp"
    files {"Tutorials/01ComputeApiInfo/ComputeApiInfoCuda.cpp"}
    includedirs {"Source"}
    links {"ktt"}
    
project "02KernelRunningCuda"
    kind "ConsoleApp"
    files {"Tutorials/02KernelRunning/KernelRunningCuda.cpp", "Tutorials/02KernelRunning/CudaKernel.cu"}
    includedirs {"Source"}
    links {"ktt"}
    
project "03KernelTuningCuda"
    kind "ConsoleApp"
    files {"Tutorials/03KernelTuning/KernelTuningCuda.cpp", "Tutorials/03KernelTuning/CudaKernel.cu"}
    includedirs {"Source"}
    links {"ktt"}
    
project "04CustomArgumentTypesCuda"
    kind "ConsoleApp"
    files {"Tutorials/04CustomArgumentTypes/CustomArgumentTypesCuda.cpp", "Tutorials/04CustomArgumentTypes/CudaKernel.cu"}
    includedirs {"Source"}
    links {"ktt"}

project "05ComputeApiInitializerCuda"
    kind "ConsoleApp"
    files {"Tutorials/05ComputeApiInitializer/ComputeApiInitializerCuda.cpp", "Tutorials/05ComputeApiInitializer/CudaKernel.cu"}
    includedirs {"Source"}
    links {"ktt"}
    linkComputeLibraries()

project "06VectorArgumentCustomizationCuda"
    kind "ConsoleApp"
    files {"Tutorials/06VectorArgumentCustomization/VectorArgumentCustomizationCuda.cpp", "Tutorials/06VectorArgumentCustomization/CudaKernel.cu"}
    includedirs {"Source"}
    links {"ktt"}

end -- cudaProjects

if vulkanProjects then

project "01InfoVulkan"
    kind "ConsoleApp"
    files {"Tutorials/01ComputeApiInfo/ComputeApiInfoVulkan.cpp"}
    includedirs {"Source"}
    links {"ktt"}
    
project "02KernelRunningVulkan"
    kind "ConsoleApp"
    files {"Tutorials/02KernelRunning/KernelRunningVulkan.cpp", "Tutorials/02KernelRunning/VulkanKernel.glsl"}
    includedirs {"Source"}
    links {"ktt"}
    
project "03KernelTuningVulkan"
    kind "ConsoleApp"
    files {"Tutorials/03KernelTuning/KernelTuningVulkan.cpp", "Tutorials/03KernelTuning/VulkanKernel.glsl"}
    includedirs {"Source"}
    links {"ktt"}
    
end -- vulkanProjects

end -- _OPTIONS["no-tutorials"]

-- Examples configuration 
if not _OPTIONS["no-examples"] then

if openClProjects then

    for _, ex in ipairs(baseExamples) do
        addOpenClExample(ex[1], ex[2], ex[3])
    end

    for _, ex in ipairs(openClOnlyExamples) do
        addOpenClExample(ex[1], ex[2], ex[3])
    end

end -- openClProjects
    
if cudaProjects then

    for _, ex in ipairs(baseExamples) do
        addCudaExample(ex[1], ex[2], ex[3])
    end

    for _, ex in ipairs(cudaOnlyExamples) do
        addCudaExample(ex[1], ex[2], ex[3])
    end

end -- cudaProjects

if cppProjects then

    for _, ex in ipairs(cppExamples) do
        addCppExample(ex[1], ex[2], ex[3])
    end

end -- cppProjects
    
end -- _OPTIONS["no-examples"]

-- Unit tests configuration   
if _OPTIONS["tests"] then

project "Tests"
    kind "ConsoleApp"
    
    files
    {
        "Tests/**",
        "Source/**",
        "Libraries/Catch-2.13.8/**",
        "Libraries/CTPL-Ahajha/**",
        "Libraries/date-3/**",
        "Libraries/Json-3.9.1/**",
        "Libraries/pugixml-1.11.4/**"
    }
    
    includedirs
    {
        "Source",
        "Libraries/Catch-2.13.8",
        "Libraries/CTPL-Ahajha",
        "Libraries/date-3",
        "Libraries/Json-3.9.1",
        "Libraries/pugixml-1.11.4"
    }
    
    if _OPTIONS["no-opencl"] then
        removefiles {"Tests/OpenClEngineTests.cpp", "Tests/Kernels/SimpleOpenClKernel.cl"}
    end
    
    filter "action:gmake*"
        buildoptions {"-pthread"}
        linkoptions {"-pthread"}
        
    filter {}
    
    defines {"KTT_LIBRARY", "KTT_TESTS"}
    linkAllLibraries()
    
end -- _OPTIONS["tests"]
