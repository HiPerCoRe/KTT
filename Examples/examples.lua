assert(openClProjects ~= nil)

-- Helper function to add example projects
function addExampleProject(name, kernelExt, apiDefine, useRefVersions, shouldEnableOpenMP)
    local projectName = name .. (useRefVersions and "Reference" or "") .. kernelExt

    local cppFiles
    if useRefVersions then
        cppFiles = {"LegacyExamples/" .. name .. "/*.cpp"}
    else
        cppFiles = {name .. "/*.cpp"}
    end

    local exLib
    if apiDefine == "KTT_CUDA_EXAMPLE" then
        exLib = "ExamplesLibCuda"
    elseif apiDefine == "KTT_OPENCL_EXAMPLE" then
        exLib = "ExamplesLibOpenCl"
    else
        exLib = "ExamplesLibCpp"
    end

    project(projectName)
        kind "ConsoleApp"
        files {table.unpack(cppFiles)}
        includedirs {"../Source", "Common"}
        defines {apiDefine}
        links {"ktt", exLib}
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


project "ExamplesLibCuda"
    kind "StaticLib"
    files
    {
        "Common/*.cpp"
    }
    includedirs {"../Source"}
    defines {"KTT_CUDA_EXAMPLE"}

project "ExamplesLibOpenCl"
    kind "StaticLib"
    files
    {
        "Common/*.cpp"
    }
    includedirs {"../Source"}
    defines {"KTT_OPENCL_EXAMPLE"}

project "ExamplesLibCpp"
    kind "StaticLib"
    files
    {
        "Common/*.cpp"
    }
    includedirs {"../Source"}
    defines {"KTT_CPP_EXAMPLE"}

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
    