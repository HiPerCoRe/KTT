-- Helper functions to find compute API headers and libraries

function findLibraries()
    local path = os.getenv("INTELOCLSDKROOT")
    if (path) then
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
    
    path = os.getenv("AMDAPPSDKROOT")
    if (path) then
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
    
    path = os.getenv("CUDA_PATH")
    if (path) then
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
        
        if _OPTIONS["cuda"] then
            defines { "PLATFORM_CUDA" }
            links { "cuda" }
        end
        
        return true
	end
    
    return false
end

-- Command line arguments definition

newoption
{
   trigger = "outdir",
   value = "path",
   description = "Specifies output directory for generated files"
}

newoption
{
   trigger = "cuda",
   description = "Enables usage of CUDA API in addition to OpenCL (Nvidia platform only)"
}

-- Project configuration

workspace "KernelTuningToolkit"
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
    
project "KernelTuningToolkit"
    kind "StaticLib"
    
    files { "source/**.h", "source/**.cpp" }
    includedirs { "source/**" }
    
    targetdir(buildPath .. "/ktt/%{cfg.platform}_%{cfg.buildcfg}")
    objdir(buildPath .. "/ktt/%{cfg.platform}_%{cfg.buildcfg}/obj")
    
    local libraries = findLibraries()
    if not libraries then
        printf("Warning: Compute API libraries were not found.")
    end

-- Examples configuration 

project "ExampleSimple"
    kind "ConsoleApp"
    
    files { "examples/simple/*.cpp", "examples/simple/*.cl" }
    includedirs { "include/**" }
    
    links { "KernelTuningToolkit" }
    
    targetdir(buildPath .. "/examples/simple/%{cfg.platform}_%{cfg.buildcfg}")
    objdir(buildPath .. "/examples/simple/%{cfg.platform}_%{cfg.buildcfg}/obj")

project "ExampleOpenCLInfo"
    kind "ConsoleApp"
    
    files { "examples/opencl_info/*.cpp" }
    includedirs { "include/**" }
    
    links { "KernelTuningToolkit" }
    
    targetdir(buildPath .. "/examples/opencl_info/%{cfg.platform}_%{cfg.buildcfg}")
    objdir(buildPath .. "/examples/opencl_info/%{cfg.platform}_%{cfg.buildcfg}/obj")

project "ExampleCoulombSum"
    kind "ConsoleApp"
    
    files { "examples/coulomb_sum/*.cpp", "examples/coulomb_sum/*.cl" }
    includedirs { "include/**" }
    
    links { "KernelTuningToolkit" }
    
    targetdir(buildPath .. "/examples/coulomb_sum/%{cfg.platform}_%{cfg.buildcfg}")
    objdir(buildPath .. "/examples/coulomb_sum/%{cfg.platform}_%{cfg.buildcfg}/obj")

project "ExampleCoulombSum3D"
    kind "ConsoleApp"

    files { "examples/coulomb_sum_3d/*.cpp", "examples/coulomb_sum_3d/*.cl" }
    includedirs { "include/**" }

    links { "KernelTuningToolkit" }

    targetdir(buildPath .. "/examples/coulomb_sum_3d/%{cfg.platform}_%{cfg.buildcfg}")
    objdir(buildPath .. "/examples/coulomb_sum_3d/%{cfg.platform}_%{cfg.buildcfg}/obj")
    
-- Unit tests configuration   
    
project "Tests"
    kind "ConsoleApp"
    
    files { "tests/**.hpp", "tests/**.cpp", "tests/**.cl" }
    includedirs { "include/**", "tests/**" }
    
    links { "KernelTuningToolkit" }
    defines { "CATCH_CPP11_OR_GREATER" }
    
    targetdir(buildPath .. "/tests/%{cfg.platform}_%{cfg.buildcfg}")
    objdir(buildPath .. "/tests/%{cfg.platform}_%{cfg.buildcfg}/obj")
    
    findLibraries()
