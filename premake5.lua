-- Global configuration variables
cudaProjects = false
openClProjects = false
vulkanProjects = false

-- Helper functions to find compute API headers and libraries
function findLibrariesAmd()
    local path = os.getenv("AMDAPPSDKROOT")
    
    if not path then
        return false
    end
    
    defines {"KTT_PLATFORM_AMD"}
    includedirs {"$(AMDAPPSDKROOT)/include"}
        
    if os.target() == "linux" then
        libdirs {"$(AMDAPPSDKROOT)/lib64"}
    else
        libdirs {"$(AMDAPPSDKROOT)/lib/x86_64"}
    end
    
    if _OPTIONS["no-opencl"] then
        return true
    end
    
    defines {"KTT_API_OPENCL"}
    links {"OpenCL"}
    openClProjects = true
    
    if _OPTIONS["profiling"] == "gpa" then
        defines {"KTT_PROFILING_GPA"}
        includedirs {"Libraries/GpuPerfApi3.6/Include"}
        
        if os.target() == "linux" then
            libdirs {"Libraries/GpuPerfApi3.6/Lib/Linux"}
        else
            -- One of the GPA headers includes Windows.h with evil min/max macros
            defines {"NOMINMAX"}
            libdirs {"Libraries/GpuPerfApi3.6/Lib/Windows"}
        end
    elseif _OPTIONS["profiling"] == "gpa-legacy" then
        defines {"KTT_PROFILING_GPA_LEGACY"}
        includedirs {"Libraries/GpuPerfApi3.3/Include"}
        
        if os.target() == "linux" then
            libdirs {"Libraries/GpuPerfApi3.3/Lib/Linux"}
        else
            defines {"NOMINMAX"}
            libdirs {"Libraries/GpuPerfApi3.3/Lib/Windows"}
        end
    end
    
    return true
end

function findLibrariesIntel()
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

function findLibrariesNvidia()
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

function findLibraries()
    if findLibrariesAmd() then
        return true
    end
    
    if findLibrariesIntel() then
        return true
    end
    
    if findLibrariesNvidia() then
        return true
    end
    
    return false
end

function findVulkan()
    local path = os.getenv("VULKAN_SDK")
    
    if not path then
        return false
    end
    
    defines {"KTT_API_VULKAN"}
    includedirs {"Libraries/Shaderc1.1.101/Include"}
    
    if os.target() == "linux" then
        includedirs {"$(VULKAN_SDK)/include"}
        libdirs {"$(VULKAN_SDK)/lib", "Libraries/Shaderc1.1.101/Lib/Linux"}
    else
        includedirs {"$(VULKAN_SDK)/Include"}
        libdirs {"$(VULKAN_SDK)/Lib", "Libraries/Shaderc1.1.101/Lib/Windows"}
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

function linkLibraries()
    local librariesFound = false
    
    if _OPTIONS["platform"] then
        if _OPTIONS["platform"] == "amd" then
            librariesFound = findLibrariesAmd()
        elseif _OPTIONS["platform"] == "intel" then
            librariesFound = findLibrariesIntel()
        elseif _OPTIONS["platform"] == "nvidia" then
            librariesFound = findLibrariesNvidia()
        else
            error("The specified platform is unknown.")
        end
    else
        librariesFound = findLibraries()
    end
    
    if not librariesFound and (not _OPTIONS["vulkan"] or _OPTIONS["platform"]) then
        error("Compute API libraries were not found. Please ensure that path to the SDK is correctly set in the environment variables:\nAMDAPPSDKROOT for AMD\nINTELOCLSDKROOT for Intel\nCUDA_PATH for Nvidia")
    end
    
    if _OPTIONS["vulkan"] then
        vulkanFound = findVulkan()
        
        if not vulkanFound then
            error("Vulkan SDK was not found. Please ensure that path to the SDK is correctly set in the environment variables under VULKAN_SDK.")
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
        {"cupti", "Nvidia CUPTI for Volta and Turing"},
        {"cupti-legacy", "Nvidia CUPTI for legacy GPUs (Volta and older)"},
        {"gpa", "AMD GPA for GCN 3.0 GPUs and newer"},
        {"gpa-legacy", "AMD GPA for GCN 5.0 GPUs and older"}
    }
}

newoption
{
    trigger = "outdir",
    value = "path",
    description = "Specifies output directory for generated project files"
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
    trigger = "tests",
    description = "Enables compilation of unit tests"
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
    
    filter {}
    
    targetdir(buildPath .. "/%{cfg.platform}_%{cfg.buildcfg}")
    objdir(buildPath .. "/%{cfg.platform}_%{cfg.buildcfg}/obj")
    
-- Library configuration
project "Ktt"
    kind "SharedLib"
    files {"Source/**"}
    includedirs {"Source", "Libraries/Half2.1.0/Include"}
    defines {"KTT_LIBRARY"}
    targetname("ktt")
    linkLibraries()

-- Tutorials configuration 
if not _OPTIONS["no-tutorials"] then

if openClProjects then
project "01InfoOpenCl"
    kind "ConsoleApp"
    files {"Tutorials/01ComputeApiInfo/ComputeApiInfoOpenCl.cpp"}
    includedirs {"Source"}
    links {"ktt"}
end -- openClProjects
    
if cudaProjects then
project "01InfoCuda"
    kind "ConsoleApp"
    files {"Tutorials/01ComputeApiInfo/ComputeApiInfoCuda.cpp"}
    includedirs {"Source"}
    links {"ktt"}
end -- cudaProjects

if vulkanProjects then
project "01InfoVulkan"
    kind "ConsoleApp"
    files {"Tutorials/01ComputeApiInfo/ComputeApiInfoVulkan.cpp"}
    includedirs {"Source"}
    links {"ktt"}
end -- vulkanProjects

end -- _OPTIONS["no-tutorials"]
