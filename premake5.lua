-- Configuration variables
cuda_examples = false
vulkan_examples = false

-- Helper functions to find compute API headers and libraries
function findLibrariesAmd()
    local path = os.getenv("AMDAPPSDKROOT")
    
    if not path then
        return false
    end
    
    defines { "PLATFORM_AMD" }
    includedirs { "$(AMDAPPSDKROOT)/include" }
        
    filter "platforms:x86"
        if os.get() == "linux" then
            libdirs { "$(AMDAPPSDKROOT)/lib" }
        else
            libdirs { "$(AMDAPPSDKROOT)/lib/x86" }
        end
        
    filter "platforms:x86_64"
        if os.get() == "linux" then
            libdirs { "$(AMDAPPSDKROOT)/lib64" }
        else
            libdirs { "$(AMDAPPSDKROOT)/lib/x86_64" }
        end
        
    filter {}
    links { "OpenCL" }
    return true
end

function findLibrariesIntel()
    local path = os.getenv("INTELOCLSDKROOT")
    
    if not path then
        return false
    end
    
    defines { "PLATFORM_INTEL" }
    includedirs { "$(INTELOCLSDKROOT)/include" }
        
    filter "platforms:x86"
        if os.get() == "linux" then
            libdirs { "$(INTELOCLSDKROOT)/lib" }
        else
            libdirs { "$(INTELOCLSDKROOT)/lib/x86" }
        end
        
    filter "platforms:x86_64"
        if os.get() == "linux" then
            libdirs { "$(INTELOCLSDKROOT)/lib64" }
        else
            libdirs { "$(INTELOCLSDKROOT)/lib/x64" }
        end
        
    filter {}
    links {"OpenCL"}
    return true
end

function findLibrariesNvidia()
    local path = os.getenv("CUDA_PATH")
    
    if not path then
        return false
    end
    
    defines { "PLATFORM_NVIDIA" }
    includedirs { "$(CUDA_PATH)/include" }
        
    filter "platforms:x86"
        if os.get() == "linux" then
            libdirs { "$(CUDA_PATH)/lib" }
        else
            libdirs { "$(CUDA_PATH)/lib/Win32" }
        end
        
    filter "platforms:x86_64"
        if os.get() == "linux" then
            libdirs { "$(CUDA_PATH)/lib64" }
        else
            libdirs { "$(CUDA_PATH)/lib/x64" }
        end
        
    filter {}
    links { "OpenCL" }
        
    if not _OPTIONS["no-cuda"] then
        cuda_examples = true
        defines { "PLATFORM_CUDA" }
        links { "cuda", "nvrtc" }
    end
        
    return true
end

function findVulkan()
    local path = os.getenv("VULKAN_SDK")
    
    if not path then
        return false
    end
    
    includedirs { "$(VULKAN_SDK)/Include" }
    
    filter "platforms:x86"
        libdirs { "$(VULKAN_SDK)/Lib32" }

    filter "platforms:x86_64"
        libdirs { "$(VULKAN_SDK)/Lib" }
    
    filter {}
    vulkan_examples = true
    defines { "PLATFORM_VULKAN" }
    links { "vulkan-1" }
    
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

-- Command line arguments definition
newoption
{
    trigger = "platform",
    value = "vendor",
    description = "Specifies platform for KTT library compilation",
    allowed =
    {
        { "amd", "AMD" },
        { "intel", "Intel" },
        { "nvidia", "Nvidia" }
    }
}

newoption
{
    trigger = "outdir",
    value = "path",
    description = "Specifies output directory for generated files"
}

newoption
{
    trigger = "vulkan",
    description = "Enables compilation of Vulkan back-end"
}

newoption
{
    trigger = "no-cuda",
    description = "Disables compilation of CUDA back-end (Nvidia platform only)"
}

newoption
{
    trigger = "tests",
    description = "Enables compilation of supplied unit tests"
}

newoption
{
    trigger = "no-examples",
    description = "Disables compilation of supplied examples"
}

-- Project configuration
workspace "ktt"
    local buildPath = "build"
    if _OPTIONS["outdir"] then
        buildPath = _OPTIONS["outdir"]
    end
    
    configurations { "Debug", "Release" }
    platforms { "x86", "x86_64" }
    location (buildPath)
    language "C++"
    flags { "C++14" }
    
    filter "platforms:x86"
        architecture "x86"
    
    filter "platforms:x86_64"
        architecture "x86_64"
    
    filter "configurations:Debug"
        defines { "DEBUG" }
        symbols "On"
    
    filter "configurations:Release"
        defines { "NDEBUG" }
        optimize "On"
    
    filter {}
    
    targetdir(buildPath .. "/%{cfg.platform}_%{cfg.buildcfg}")
    objdir(buildPath .. "/%{cfg.platform}_%{cfg.buildcfg}/obj")

-- Library configuration
project "ktt"
    kind "SharedLib"
    files { "source/**.h", "source/**.hpp", "source/**.cpp" }
    includedirs { "source" }
    defines { "KTT_LIBRARY" }
    
    local libraries = false
    
    if _OPTIONS["platform"] then
        if _OPTIONS["platform"] == "amd" then
            libraries = findLibrariesAmd()
        elseif _OPTIONS["platform"] == "intel" then
            libraries = findLibrariesIntel()
        else
            libraries = findLibrariesNvidia()
        end
    else
        libraries = findLibraries()
    end
    
    if _OPTIONS["vulkan"] then
        vulkan = findVulkan()
        
        if not vulkan then
            error("Vulkan SDK was not found")
        end
    end
    
    if not libraries then
        error("Compute API libraries were not found")
    end
    
-- Examples configuration 
if not _OPTIONS["no-examples"] then

project "simple_opencl"
    kind "ConsoleApp"
    files { "examples/simple/simple_opencl.cpp", "examples/simple/simple_opencl_kernel.cl" }
    includedirs { "source" }
    links { "ktt" }

project "info_opencl"
    kind "ConsoleApp"
    files { "examples/compute_api_info/compute_api_info_opencl.cpp" }
    includedirs { "source" }
    links { "ktt" }

project "nbody_opencl"
    kind "ConsoleApp"
    files { "examples/nbody/*.cpp", "examples/nbody/*.cl" }
    includedirs { "source" }
    links { "ktt" }

project "coulomb_sum_2d_opencl"
    kind "ConsoleApp"
    files { "examples/coulomb_sum_2d/*.cpp", "examples/coulomb_sum_2d/*.cl" }
    includedirs { "source" }
    links { "ktt" }

project "coulomb_sum_3d_opencl"
    kind "ConsoleApp"
    files { "examples/coulomb_sum_3d/*.cpp", "examples/coulomb_sum_3d/*.cl" }
    includedirs { "source" }
    links { "ktt" }

project "coulomb_sum_3d_iterative_opencl"
    kind "ConsoleApp"
    files { "examples/coulomb_sum_3d_iterative/*.h", "examples/coulomb_sum_3d_iterative/*.cpp", "examples/coulomb_sum_3d_iterative/*.cl" }
    includedirs { "source" }
    links { "ktt" }
    
project "reduction_opencl"
    kind "ConsoleApp"
    files { "examples/reduction/*.h", "examples/reduction/*.cpp", "examples/reduction/*.cl" }
    includedirs { "source" }
    links { "ktt" }

if cuda_examples then

project "simple_cuda"
    kind "ConsoleApp"
    files { "examples/simple/simple_cuda.cpp", "examples/simple/simple_cuda_kernel.cu" }
    includedirs { "source" }
    links { "ktt" }
    
project "info_cuda"
    kind "ConsoleApp"
    files { "examples/compute_api_info/compute_api_info_cuda.cpp" }
    includedirs { "source" }
    links { "ktt" }

end -- cuda_examples

if vulkan_examples then

project "info_vulkan"
    kind "ConsoleApp"
    files { "examples/compute_api_info/compute_api_info_vulkan.cpp" }
    includedirs { "source" }
    links { "ktt" }

end -- vulkan_examples

end -- _OPTIONS["no-examples"]
    
-- Unit tests configuration   
if _OPTIONS["tests"] then

project "tests"
    kind "ConsoleApp"
    files { "tests/**.hpp", "tests/**.cpp", "tests/**.cl", "source/**.h", "source/**.hpp", "source/**.cpp" }
    includedirs { "tests", "source" }
    defines { "KTT_TESTS", "DO_NOT_USE_WMAIN" }
    
    if _OPTIONS["platform"] then
        if _OPTIONS["platform"] == "amd" then
            findLibrariesAmd()
        elseif _OPTIONS["platform"] == "intel" then
            findLibrariesIntel()
        else
            findLibrariesNvidia()
        end
    else
        findLibraries()
    end
    
    if _OPTIONS["vulkan"] then
        findVulkan()
    end
    
end -- _OPTIONS["tests"]
