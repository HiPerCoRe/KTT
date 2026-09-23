-- Tutorials are added automatically by traversing this folder.
-- 06ExampleCategory/03HelloWorld becomes 0603HelloWorld(Cuda|OpenCl|Vulkan|Cpp).
-- If the project folder contains multiple C++ files with compute API suffixes (TunerInitialization), the
-- variants are generated from those; if there's only one file (MultipleBackends), the variants are
-- generated from the kernels present. See the tutorialApis table for what corresponds
-- to each other.
-- The kernel file of each variant is passed to its compilation as the KTT_TUTORIAL_KERNEL_FILE
-- define, so that tutorial code never contains the path to its own folder.
--
-- Inside a level, tutorial folders are numbered from 01 with no gaps and no prefix used twice, in
-- the order the tutorials are meant to be read. Generation warns when that does not hold.
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

-- Warns when the numeric prefixes of the tutorials of one level do not describe a reading order:
-- a prefix that several folders share, or prefixes that do not run in order from the first
-- position.
local function validateTutorialNumbering(level, tutorials)
    local foldersByPrefix = {}
    local prefixes = {}
    local width = 0

    for _, tutorial in ipairs(tutorials) do
        local prefix = numberPrefix(tutorial)
        local position = tonumber(prefix)
        width = math.max(width, #prefix)

        if not foldersByPrefix[position] then
            foldersByPrefix[position] = {}
            table.insert(prefixes, position)
        end

        table.insert(foldersByPrefix[position], path.getbasename(tutorial))
    end

    table.sort(prefixes)

    local function formattedPrefix(position)
        return string.format("%0" .. width .. "d", position)
    end

    local function correctNameExample()
        local example = {}

        for position = 1, math.min(#prefixes, 3) do
            table.insert(example, formattedPrefix(position) .. "TutorialName")
        end

        return table.concat(example, ", ")
    end

    for _, prefix in ipairs(prefixes) do
        local folders = foldersByPrefix[prefix]

        if #folders > 1 then
            printf("Warning: tutorials %s in level %s all have the prefix %s; every tutorial of a level needs a different prefix, in the form %s", table.concat(folders, ", "), level, formattedPrefix(prefix), correctNameExample())
        end
    end

    local found = {}
    local expected = {}

    for position, prefix in ipairs(prefixes) do
        table.insert(found, formattedPrefix(prefix))
        table.insert(expected, formattedPrefix(position))
    end

    if table.concat(found, ", ") ~= table.concat(expected, ", ") then
        printf("Warning: the tutorials of level %s do not run in order from the first position: found prefixes %s, expected %s, in the form %s", level, table.concat(found, ", "), table.concat(expected, ", "), correctNameExample())
    end
end

function addAllTutorials()
    for _, level in ipairs(sorted(os.matchdirs("*"))) do
        if numberPrefix(level) then
            local tutorials = {}

            for _, tutorial in ipairs(sorted(os.matchdirs(level .. "/*"))) do
                if numberPrefix(tutorial) then
                    table.insert(tutorials, tutorial)
                end
            end

            validateTutorialNumbering(level, tutorials)

            for _, tutorial in ipairs(tutorials) do
                addTutorial(tutorial)
            end
        end
    end
end

addAllTutorials()
