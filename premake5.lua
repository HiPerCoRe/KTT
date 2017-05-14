-- Helper functions to find compute API headers and libraries

function initOpencl()
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
    
    return false
end

function initCuda()
    local path = os.getenv("CUDA_PATH")
    if (path) then
        defines { "USE_CUDA" }
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
        links { "cuda" }
        return true
	end
    
    return false
end

-- Command line arguments definition

newoption
{
   trigger     = "cuda",
   description = "Enables usage of CUDA API in addition to OpenCL (Nvidia platform only)"
}

-- Project configuration

workspace "KernelTuningToolkit"
    configurations { "Debug", "Release" }
    platforms { "x86", "x86_64" }
    location "build"
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
    
    targetdir("build/ktt/%{cfg.platform}_%{cfg.buildcfg}")
    objdir("build/ktt/obj/%{cfg.platform}_%{cfg.buildcfg}")
    
    local opencl = initOpencl()
    if not opencl then
        printf("Warning: OpenCL libraries were not found.")
    end
    
    if _OPTIONS["cuda"] then
        local cuda = initCuda()
        if not cuda then
            printf("Warning: CUDA libraries were not found.")
        end
    end

-- Examples configuration 

project "ExampleSimple"
    kind "ConsoleApp"
    
    files { "examples/simple/*.cpp", "examples/simple/*.cl" }
    includedirs { "include/**" }
    
    links { "KernelTuningToolkit" }
    
    targetdir("build/examples/simple/%{cfg.platform}_%{cfg.buildcfg}")
    objdir("build/examples/simple/obj/%{cfg.platform}_%{cfg.buildcfg}")
    
    initOpencl()
    if _OPTIONS["cuda"] then
        initCuda()
    end

project "ExampleOpenCLInfo"
    kind "ConsoleApp"
    
    files { "examples/opencl_info/*.cpp" }
    includedirs { "include/**" }
    
    links { "KernelTuningToolkit" }
    
    targetdir("build/examples/opencl_info/%{cfg.platform}_%{cfg.buildcfg}")
    objdir("build/examples/opencl_info/obj/%{cfg.platform}_%{cfg.buildcfg}")
   
    initOpencl()
    if _OPTIONS["cuda"] then
        initCuda()
    end

project "ExampleCoulombSum"
    kind "ConsoleApp"
    
    files { "examples/coulomb_sum/*.cpp", "examples/coulomb_sum/*.cl" }
    includedirs { "include/**" }
    
    links { "KernelTuningToolkit" }
    
    targetdir("build/examples/coulomb_sum/%{cfg.platform}_%{cfg.buildcfg}")
    objdir("build/examples/coulomb_sum/obj/%{cfg.platform}_%{cfg.buildcfg}")
   
    initOpencl()
    if _OPTIONS["cuda"] then
        initCuda()
    end

project "ExampleCoulombSum3D"
    kind "ConsoleApp"

    files { "examples/coulomb_sum_3d/*.cpp", "examples/coulomb_sum_3d/*.cl" }
    includedirs { "include/**" }

    links { "KernelTuningToolkit" }

    targetdir("build/examples/coulomb_sum_3d/%{cfg.platform}_%{cfg.buildcfg}")
    objdir("build/examples/coulomb_sum_3d/obj/%{cfg.platform}_%{cfg.buildcfg}")

    initOpencl()
    if _OPTIONS["cuda"] then
        initCuda()
    end
    
-- Unit tests configuration   
    
project "Tests"
    kind "ConsoleApp"
    
    files { "tests/**.hpp", "tests/**.cpp", "tests/**.cl" }
    includedirs { "include/**", "tests/**" }
    
    links { "KernelTuningToolkit" }
    defines { "CATCH_CPP11_OR_GREATER" }
    
    targetdir("build/tests/%{cfg.platform}_%{cfg.buildcfg}")
    objdir("build/tests/obj/%{cfg.platform}_%{cfg.buildcfg}")
    
    initOpencl()
    if _OPTIONS["cuda"] then
        initCuda()
    end
