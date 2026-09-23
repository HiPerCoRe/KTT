-- Tutorials are added automatically by traversing this folder.
-- 06ExampleCategory/03HelloWorld becomes 0603HelloWorld(Cuda|OpenCl|Vulkan|Cpp).
-- If the project folder contains multiple C++ files with compute API suffixes (TunerInitialization), the
-- variants are generated from those; if there's only one file (MultipleBackends), the variants are
-- generated from the kernels present. See the tutorialApis table for what corresponds
-- to each other.
-- The kernel file of each variant is passed to its compilation as the KTT_TUTORIAL_KERNEL_FILE
-- define, so that tutorial code never contains the path to its own folder.
--
-- There is a testing script in the Scripts folder that can be used to test if generation works correctly.
-- The recorded project list will, however, become invalid whenever the folder structure is changed.
-- In that case, run the script with the --update flag.

assert(openClProjects ~= nil)

tutorialApis =
{
    { suffix = "Cpp",    enabledFlag = "cppProjects",    kernelExtension = "cppkernel" },
    { suffix = "Cuda",   enabledFlag = "cudaProjects",   kernelExtension = "cu" },
    { suffix = "OpenCl", enabledFlag = "openClProjects", kernelExtension = "cl" },
    { suffix = "Vulkan", enabledFlag = "vulkanProjects", kernelExtension = "glsl" },
}

-- Tutorials whose host code uses compute API objects directly, includes
-- compute API headers too and needs to link the libraries.
tutorialsUsingComputeApiHeaders =
{
    ["03Advanced/05ComputeApiInitializer"] = true,
}

local function sorted(list)
    table.sort(list)
    return list
end

local function numberPrefix(folder)
    return string.match(path.getbasename(folder), "^%d+")
end

-- The kernel path passed to the host code as KTT_TUTORIAL_KERNEL_FILE, built with the same rule
-- as the kernel file name in addTutorial. Passing the path from the build keeps tutorial sources
-- free of the name of their own folder, so renaming a folder changes only the build, not the
-- tutorial code. The leading "../" is kept from the paths the sources hardcoded before, so the
-- string that the host code produces by prepending kernelPrefix is unchanged.
function tutorialKernelFileDefine(kernelFile)
    return 'KTT_TUTORIAL_KERNEL_FILE="../Tutorials/' .. kernelFile .. '"'
end

function tutorialApiOfHostFile(file)
    if path.getextension(file) ~= ".cpp" then
        return nil
    end

    local name = path.getbasename(file)

    for _, api in ipairs(tutorialApis) do
        if string.endswith(name, api.suffix) then
            return api
        end
    end

    return nil
end

function addTutorialProject(category, tutorial, api, tutorialFiles, kernelFile)
    project(category .. path.getbasename(tutorial) .. api.suffix)
        kind "ConsoleApp"
        files {table.unpack(tutorialFiles)}
        includedirs {"../Source"}
        defines {"KTT_" .. string.upper(api.suffix) .. "_TUTORIAL"}
        links {"ktt"}

        if kernelFile then
            defines {tutorialKernelFileDefine(kernelFile)}
        end

        if tutorialsUsingComputeApiHeaders[tutorial] then
            linkComputeLibraries()
        end
end

function addTutorial(tutorial)
    local category = numberPrefix(path.getdirectory(tutorial))
    local apiHostFiles = {}
    local sharedHostFiles = {}

    for _, file in ipairs(sorted(os.matchfiles(tutorial .. "/*"))) do
        local api = tutorialApiOfHostFile(file)

        if api then
            apiHostFiles[api.suffix] = file
        elseif path.getextension(file) == ".cpp" then
            table.insert(sharedHostFiles, file)
        end
    end

    if #sharedHostFiles > 1 then
        printf("Warning: tutorial %s has more than one host file shared between APIs, only %s is used", tutorial, sharedHostFiles[1])
    end

    for _, api in ipairs(tutorialApis) do
        if _G[api.enabledFlag] then
            local hostFile = apiHostFiles[api.suffix] or sharedHostFiles[1]
            local kernelFile = tutorial .. "/" .. api.suffix .. "Kernel." .. api.kernelExtension
            local hasKernelFile = os.isfile(kernelFile)

            if hostFile and (apiHostFiles[api.suffix] or hasKernelFile) then
                local files = {hostFile}

                if hasKernelFile then
                    table.insert(files, kernelFile)
                end

                addTutorialProject(category, tutorial, api, files, hasKernelFile and kernelFile)
            end
        end
    end
end

function addAllTutorials()
    for _, level in ipairs(sorted(os.matchdirs("*"))) do
        if numberPrefix(level) then
            for _, tutorial in ipairs(sorted(os.matchdirs(level .. "/*"))) do
                if numberPrefix(tutorial) then
                    addTutorial(tutorial)
                end
            end
        end
    end
end

addAllTutorials()
