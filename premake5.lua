-- Global configuration variables
cudaProjects = false
openClProjects = false
vulkanProjects = false

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
        libdirs {"$(CUDA_PATH)/lib64"}
    else
        libdirs {"$(CUDA_PATH)/lib/x64"}
    end
    
    if not _OPTIONS["no-opencl"] then
        defines {"KTT_API_OPENCL"}
        links {"OpenCL"}
        openClProjects = true
    end
        
    if not _OPTIONS["no-cuda"] then
        defines {"KTT_API_CUDA"}
        links {"cuda", "nvrtc"}
        cudaProjects = true
        
        if _OPTIONS["profiling"] == "cupti-legacy" or _OPTIONS["profiling"] == "cupti" then
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

    if linkLibrariesAmd() then
        return true
    end
    
    if linkLibrariesIntel() then
        return true
    end
    
    if linkLibrariesNvidia() then
        return true
    end
    
    return false
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
    
    defines {"KTT_PYTHON", "PYBIND11_USE_SMART_HOLDER_AS_DEFAULT"}
    includedirs {pythonHeaders, "Libraries/pybind11-2.8.1-smart_holder"}
    files {"Libraries/pybind11-2.8.1-smart_holder/**"}
    
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
    
    -- Allow usage of KTT with only Vulkan if no other compute API was explicitly specified by user
    if not librariesFound and (not _OPTIONS["vulkan"] or _OPTIONS["platform"]) then
        error("Compute API libraries were not found. Please ensure that path to the SDK is correctly set in the environment variables:\nOCL_ROOT for AMD\nINTELOCLSDKROOT for Intel\nCUDA_PATH for Nvidia")
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
    trigger = "python",
    description = "Enables compilation of Python bindings"
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
    trigger = "no-tutorials",
    description = "Disables compilation of tutorials"
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
        buildoptions {"/Zc:__cplusplus", "/permissive-"}
        
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

project "AtfSamplesOpenCl"
    kind "ConsoleApp"
    files {"Examples/AtfSamples/*.cpp", "Examples/AtfSamples/*.cl"}
    includedirs {"Source"}
    defines {"KTT_OPENCL_EXAMPLE"}
    links {"ktt"}

project "BicgOpenCl"
    kind "ConsoleApp"
    files {"Examples/Bicg/*.cpp", "Examples/Bicg/*.cl"}
    includedirs {"Source"}
    defines {"KTT_OPENCL_EXAMPLE"}
    links {"ktt"}

project "ClTuneConvolutionOpenCl"
    kind "ConsoleApp"
    files {"Examples/ClTuneConvolution/*.cpp", "Examples/ClTuneConvolution/*.cl"}
    includedirs {"Source"}
    defines {"KTT_OPENCL_EXAMPLE"}
    links {"ktt"}

project "ClTuneGemmOpenCl"
    kind "ConsoleApp"
    files {"Examples/ClTuneGemm/*.cpp", "Examples/ClTuneGemm/*.cl"}
    includedirs {"Source"}
    defines {"KTT_OPENCL_EXAMPLE"}
    links {"ktt"}

project "Convolution3dOpenCl"
    kind "ConsoleApp"
    files {"Examples/Convolution3d/*.cpp", "Examples/Convolution3d/*.cl"}
    includedirs {"Source"}
    defines {"KTT_OPENCL_EXAMPLE"}
    links {"ktt"}

project "CoulombSum2dOpenCl"
    kind "ConsoleApp"
    files {"Examples/CoulombSum2d/*.cpp", "Examples/CoulombSum2d/*.cl"}
    includedirs {"Source"}
    defines {"KTT_OPENCL_EXAMPLE"}
    links {"ktt"}

project "CoulombSum3dOpenCl"
    kind "ConsoleApp"
    files {"Examples/CoulombSum3d/*.cpp", "Examples/CoulombSum3d/*.cl"}
    includedirs {"Source"}
    defines {"KTT_OPENCL_EXAMPLE"}
    links {"ktt"}

project "CoulombSum3dIterativeOpenCl"
    kind "ConsoleApp"
    files {"Examples/CoulombSum3dIterative/*.cpp", "Examples/CoulombSum3dIterative/*.cl"}
    includedirs {"Source"}
    defines {"KTT_OPENCL_EXAMPLE"}
    links {"ktt"}

project "CovarianceOpenCl"
    kind "ConsoleApp"
    files {"Examples/Covariance/*.cpp", "Examples/Covariance/*.cl"}
    includedirs {"Source"}
    defines {"KTT_OPENCL_EXAMPLE"}
    links {"ktt"}

project "NbodyOpenCl"
    kind "ConsoleApp"
    files {"Examples/Nbody/*.cpp", "Examples/Nbody/*.cl"}
    includedirs {"Source"}
    defines {"KTT_OPENCL_EXAMPLE"}
    links {"ktt"}

project "ReductionOpenCl"
    kind "ConsoleApp"
    files {"Examples/Reduction/*.cpp", "Examples/Reduction/*.cl"}
    includedirs {"Source"}
    defines {"KTT_OPENCL_EXAMPLE"}
    links {"ktt"}

project "SortOpenCl"
    kind "ConsoleApp"
    files {"Examples/Sort/*.cpp", "Examples/Sort/*.cl"}
    includedirs {"Source"}
    defines {"KTT_OPENCL_EXAMPLE"}
    links {"ktt"}

project "Sort2OpenCl"
    kind "ConsoleApp"
    files {"Examples/Sort2/*.cpp", "Examples/Sort2/*.cl"}
    includedirs {"Source"}
    defines {"KTT_OPENCL_EXAMPLE"}
    links {"ktt"}

project "TransposeOpenCl"
    kind "ConsoleApp"
    files {"Examples/Transpose/*.cpp", "Examples/Transpose/*.cl"}
    includedirs {"Source"}
    defines {"KTT_OPENCL_EXAMPLE"}
    links {"ktt"}

end -- openClProjects
    
if cudaProjects then

project "AtfSamplesCuda"
    kind "ConsoleApp"
    files {"Examples/AtfSamples/*.cpp", "Examples/AtfSamples/*.cu"}
    includedirs {"Source"}
    defines {"KTT_CUDA_EXAMPLE"}
    links {"ktt"}

project "BicgCuda"
    kind "ConsoleApp"
    files {"Examples/Bicg/*.cpp", "Examples/Bicg/*.cu"}
    includedirs {"Source"}
    defines {"KTT_CUDA_EXAMPLE"}
    links {"ktt"}

project "ClTuneConvolutionCuda"
    kind "ConsoleApp"
    files {"Examples/ClTuneConvolution/*.cpp", "Examples/ClTuneConvolution/*.cu"}
    includedirs {"Source"}
    defines {"KTT_CUDA_EXAMPLE"}
    links {"ktt"}

project "ClTuneGemmCuda"
    kind "ConsoleApp"
    files {"Examples/ClTuneGemm/*.cpp", "Examples/ClTuneGemm/*.cu"}
    includedirs {"Source"}
    defines {"KTT_CUDA_EXAMPLE"}
    links {"ktt"}

project "CoulombSum3dCuda"
    kind "ConsoleApp"
    files {"Examples/CoulombSum3d/*.cpp", "Examples/CoulombSum3d/*.cu"}
    includedirs {"Source"}
    defines {"KTT_CUDA_EXAMPLE"}
    links {"ktt"}

project "NbodyCuda"
    kind "ConsoleApp"
    files {"Examples/Nbody/*.cpp", "Examples/Nbody/*.cu"}
    includedirs {"Source"}
    defines {"KTT_CUDA_EXAMPLE"}
    links {"ktt"}

project "ReductionCuda"
    kind "ConsoleApp"
    files {"Examples/Reduction/*.cpp", "Examples/Reduction/*.cu"}
    includedirs {"Source"}
    defines {"KTT_CUDA_EXAMPLE"}
    links {"ktt"}

project "SortCuda"
    kind "ConsoleApp"
    files {"Examples/Sort/*.cpp", "Examples/Sort/*.cu"}
    includedirs {"Source"}
    defines {"KTT_CUDA_EXAMPLE"}
    links {"ktt"}

project "Sort2Cuda"
    kind "ConsoleApp"
    files {"Examples/Sort2/*.cpp", "Examples/Sort2/*.cu"}
    includedirs {"Source"}
    defines {"KTT_CUDA_EXAMPLE"}
    links {"ktt"}

project "TransposeCuda"
    kind "ConsoleApp"
    files {"Examples/Transpose/*.cpp", "Examples/Transpose/*.cu"}
    includedirs {"Source"}
    defines {"KTT_CUDA_EXAMPLE"}
    links {"ktt"}

end -- cudaProjects
    
end -- _OPTIONS["no-examples"]
