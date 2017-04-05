-- Helper functions to find OpenCL headers and libraries --

function initOpenCL()
	local path = os.getenv("INTELOCLSDKROOT")
	if (path) then
		defines { "CL_PLATFORM_INTEL" }
		includedirs { "$(INTELOCLSDKROOT)/include" }
		
        filter "platforms:x86"
			libdirs { "$(INTELOCLSDKROOT)/lib/x86" }
        
		filter "platforms:x86_64"
			libdirs { "$(INTELOCLSDKROOT)/lib/x64" }
        
		filter {}
		links {"OpenCL"}
		return true
	end
    
    path = os.getenv("CUDA_PATH")
    if (path) then
        defines { "CL_PLATFORM_NVIDIA" }
        includedirs { "$(CUDA_PATH)/include" }
		
        filter "platforms:x86"
			libdirs { "$(CUDA_PATH)/lib" }
            
		filter "platforms:x86_64"
			libdirs { "$(CUDA_PATH)/lib64" }
            
        filter {}
		links { "OpenCL" }
		return true
	end
    
    path = os.getenv("AMDAPPSDKROOT")
    if (path) then
        defines { "CL_PLATFORM_AMD" }
        includedirs { "$(AMDAPPSDKROOT)/include" }
        
		filter "platforms:x86"
			libdirs { "$(AMDAPPSDKROOT)/lib/x86" }
        
		filter "platforms:x86_64"
			libdirs { "$(AMDAPPSDKROOT)/lib/x86_64" }
        
		filter {}
		links { "OpenCL" }
        return true
    end
    
	return false
end

-- Command line arguments definition --

newoption
{
   trigger     = "cuda",
   description = "Enables usage of CUDA API instead of OpenCL"
}

-- Project configuration --

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
    
    filter {}
    
project "KernelTuningToolkit"
    kind "StaticLib"
    
    files { "source/**.h", "source/**.cpp" }
    includedirs { "source/**" }
    
    filter "configurations:Debug"
        defines { "DEBUG" }
        symbols "On"
    
    filter "configurations:Release"
        defines { "NDEBUG" }
        optimize "On"
    
    filter {}
    
    targetdir("build/ktt/%{cfg.platform}_%{cfg.buildcfg}")
    objdir("build/ktt/obj/%{cfg.platform}_%{cfg.buildcfg}")
    
    if not _OPTIONS["cuda"] then
        defines { "USE_OPENCL" }
        local result = initOpenCL()
        
        if not result then
            printf("Warning: OpenCL libraries were not found.")
        end
    else
        defines { "USE_CUDA" }
        printf("Warning: CUDA platform is not supported yet.")
    end

-- Examples configuration --    

project "ExampleSimple"
    kind "ConsoleApp"
    
    files { "examples/simple/*.cpp", "examples/simple/*.cl" }
    includedirs { "include/**" }
    
    links { "KernelTuningToolkit" }
    
    filter "configurations:Debug"
        defines { "DEBUG" }
        symbols "On"
        
    filter "configurations:Release"
        defines { "NDEBUG" }
        optimize "On"
        
    filter {}
    
    targetdir("build/examples/simple/%{cfg.platform}_%{cfg.buildcfg}")
    objdir("build/examples/simple/obj/%{cfg.platform}_%{cfg.buildcfg}")

    if not _OPTIONS["cuda"] then
        initOpenCL()
    else
        -- CUDA not supported yet
    end

project "ExampleOpenCLInfo"
    kind "ConsoleApp"
    
    files { "examples/opencl_info/*.cpp" }
    includedirs { "include/**" }
    
    links { "KernelTuningToolkit" }
    
    filter "configurations:Debug"
        defines { "DEBUG" }
        symbols "On"
        
    filter "configurations:Release"
        defines { "NDEBUG" }
        optimize "On"
        
    filter {}
    
    targetdir("build/examples/opencl_info/%{cfg.platform}_%{cfg.buildcfg}")
    objdir("build/examples/opencl_info/obj/%{cfg.platform}_%{cfg.buildcfg}")
   
    if not _OPTIONS["cuda"] then
        initOpenCL()
    else
        -- CUDA not supported yet
    end
 
-- Unit tests configuration --    
    
project "Tests"
    kind "ConsoleApp"
    
    files { "tests/**.hpp", "tests/**.cpp", "tests/**.cl" }
    includedirs { "include/**", "tests/**" }
    
    links { "KernelTuningToolkit" }
    defines { "CATCH_CPP11_OR_GREATER" }
    
    filter "configurations:Debug"
        defines { "DEBUG" }
        symbols "On"
        
    filter "configurations:Release"
        defines { "NDEBUG" }
        optimize "On"
        
    filter {}
    
    targetdir("build/tests/%{cfg.platform}_%{cfg.buildcfg}")
    objdir("build/tests/obj/%{cfg.platform}_%{cfg.buildcfg}")
    
    if not _OPTIONS["cuda"] then
        initOpenCL()
    else
        -- CUDA not supported yet
    end
