-- Helper function to find compute API headers and libraries

cuda_examples = false

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
        
        if not _OPTIONS["no-cuda"] then
            cuda_examples = true
            defines { "PLATFORM_CUDA" }
            links { "cuda", "nvrtc" }
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
    
    targetdir(buildPath .. "/%{cfg.platform}_%{cfg.buildcfg}")
    objdir(buildPath .. "/%{cfg.platform}_%{cfg.buildcfg}/obj")

-- Library configuration
   
project "KernelTuningToolkit"
    kind "SharedLib"
    
    files { "source/**.h", "source/**.hpp", "source/**.cpp" }
    includedirs { "source/**" }
    defines { "KTT_LIBRARY" }
    
    local libraries = findLibraries()
    if not libraries then
        printf("Warning: Compute API libraries were not found.")
    end

-- Examples configuration 

if not _OPTIONS["no-examples"] then

project "ExampleSimple"
    kind "ConsoleApp"
    
    files { "examples/simple/*.cpp", "examples/simple/*.cl" }
    includedirs { "include/**" }
    links { "KernelTuningToolkit" }

project "ExampleOpenCLInfo"
    kind "ConsoleApp"
    
    files { "examples/opencl_info/*.cpp" }
    includedirs { "include/**" }
    links { "KernelTuningToolkit" }

project "ExampleCoulombSum"
    kind "ConsoleApp"
    
    files { "examples/coulomb_sum/*.cpp", "examples/coulomb_sum/*.cl" }
    includedirs { "include/**" }
    links { "KernelTuningToolkit" }

project "ExampleCoulombSum3D"
    kind "ConsoleApp"

    files { "examples/coulomb_sum_3d/*.cpp", "examples/coulomb_sum_3d/*.cl" }
    includedirs { "include/**" }
    links { "KernelTuningToolkit" }

project "ExampleReduction"
    kind "ConsoleApp"

    files { "examples/reduction/*.h", "examples/reduction/*.cpp", "examples/reduction/*.cl" }
    includedirs { "include/**" }
    links { "KernelTuningToolkit" }

if cuda_examples then

project "ExampleSimpleCuda"
    kind "ConsoleApp"
    
    files { "examples/simple_cuda/*.cpp", "examples/simple_cuda/*.cu" }
    includedirs { "include/**" }
    links { "KernelTuningToolkit" }
    
end -- cuda_examples

end -- _OPTIONS["no-examples"]
    
-- Unit tests configuration   

if _OPTIONS["tests"] then

project "Tests"
    kind "ConsoleApp"
    
    files { "tests/**.hpp", "tests/**.cpp", "tests/**.cl", "source/**.h", "source/**.hpp", "source/**.cpp" }
    includedirs { "tests/**", "source/**" }
    defines { "KTT_TESTS", "DO_NOT_USE_WMAIN" }
    
    findLibraries()
    
end -- _OPTIONS["tests"]
