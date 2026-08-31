#pragma once

#include "CliComponent.h"
#include "CompilerTuningComponent.h"
#include "RunStats.hpp"
#include <Ktt.h>
#include <memory>
#include <random>
#include <type_traits>
#include <optional>

#ifndef RAND_MAX
#define RAND_MAX UINT_MAX
#endif

/** @class Base class providing common functionality for Examples.
  *
  * Inherited by Examples that do not use the reference functionality.
  * This class handles initialization of the KTT tuner, CLI argument parsing,
  * kernel loading, and execution of tuning/execution phases. Derived classes
  * must implement InitData(), InitKernel(), and InitTuningSpace().
  * Optionally, they may implement InitCLI() as well.
  *
  * Many useful helper methods are provided as well.
  */
class ExampleBase {
protected:
    /** @fn Construct a new Example Base object.
      * @param argc Argument count from main().
      * @param argv Argument vector from main().
      * @param exampleFolderPath Path to the example folder containing kernels. Relative to KTT root folder.
      * @param defaultKernelFileBaseName Base name for the kernel source file, without the suffix.
      */
    ExampleBase(int argc, char** argv,
        std::string exampleFolderPath, std::string defaultKernelFileBaseName);

public:

    /** @fn Run the tuning process after post-initialization */
    void Run();

    /** @fn Factory method for ExampleBase and ExampleReferenceComputation. 
      * Constructs a new object and calls its PostInitialize() method.
      */
    template <class T>
    static std::unique_ptr<T> Create(int argc, char** argv, 
            std::string exampleFolderPath, std::string defaultKernelFileBaseName)
    {
        std::unique_ptr<T> ex(new T(argc, argv, exampleFolderPath, defaultKernelFileBaseName));
        ex->PostInitialize();
        return ex;
    }

    // Public virtual constructor necessary for a base class intended to be used with pointers.
    virtual ~ExampleBase() = default;

protected:
    const ktt::ComputeApi m_computeApi;
    int m_argc;
    char **m_argv;

    std::string m_kernelFile;
    std::shared_ptr<ktt::Tuner> m_tuner;

    ktt::KernelDefinitionId m_definition;
    ktt::KernelId m_kernel;

    CliComponent m_cli;
    std::unique_ptr<CompilerTuningComponent> m_compilerTuning;

    std::optional<ktt::PreciseMeasurementParameters> m_preciseParams;
    float m_preheatingSeconds;
    std::unique_ptr<ktt::StopCondition> m_stopCondition;
    
    ktt::PlatformIndex m_platform = 0;
    ktt::PlatformIndex m_device = 0;
    bool m_useProfiling = false;
    bool m_rapidTest = false;
    bool m_useDynamicTuning = false;
    double m_dynamicTuningTime = 0;
    std::unique_ptr<ktt::Searcher> m_searcher;
    std::string m_profileSearchModelPath;

    /** @fn Static method that constructs a kernel file path with the provided parameters. 
      * Takes into account the expected bin folder location and the current platform.
      *
      * @param exampleFolderPath Location of the Example's folder. Relative to the KTT root folder. ("Example/ExampleName")
      * @param baseName Name of the kernel file, without suffix.
      * @param suffix Optional manual suffix parameter if different than the usual for the current compute API.
      */
    static std::string GetKernelFilePath(std::string exampleFolderPath, std::string baseName, std::optional<std::string> suffix = std::nullopt);

    /** @fn Calls the init functions in the order in which they are declared in ExampleBase.h. Should not need to be modified.
      * The order is: InitCLI -> ProcessCLI -> InitTuner -> InitData -> InitKernel -> InitTuningSpace -> Preheat -> InitSearcher.
      */
    virtual void PostInitialize();

    /** @fn Adds the default CLI options, may be overriden for customization. When overriding, make sure to do ExampleBase::InitCLI()
      * to include default options as well. See CliComponent for details. 
      * Good place to use UseInputSizeOption(...).
      */
    virtual void InitCLI();

    /** @fn Calls m_cli.ProcessCLI(...). See CliComponent for details. */
    void ProcessCLI();
    
    /** @fn Initializes the tuner. Acts on the UseFastMath and UseOpenMP flag setters.
      *
      * KTT requires that platform and device IDs and the compute API are passed into the tuner's constructor, and they
      * cannot be changed after this. The user may change these through the CLI, so the tuner must be initialized after
      * ProcessCLI.
      */
    void InitTuner();

    /** @fn Abstract method. Intended to implement buffer initialization. Good place to use FillBuffers<T>(...).
      *
      * CLI input is already processed when this method is called, so one can take the user's choices into account.
      */
    virtual void InitData() = 0;

    /** @fn Abstract method. Intended to implement kernel and kernel argument initialization.
      * Good place to use InitKernelDefault(...).
      */
    virtual void InitKernel() = 0;

    /** @fn Abstract method. Intended to implement tuning space initialization. */
    virtual void InitTuningSpace() = 0;

    /** @fn Preheats the GPU if the user passed the appropriate CLI flag. Is part of initialiation because it needs to run 
      * before the searcher is initialized.
      */
    void Preheat();

    /** @fn Initializes the searcher based on CLI options passed by the user. Defaults to deterministic. */
    void InitSearcher();

    /** @fn Sets a flag to make the tuner use fast math. Can be used with UseOpenMP (compiler options get added together)
      * WARNING: Overrides any prior user-defined compiler options set with Tuner::SetCompilerOptions.
      */
    void UseFastMath();

    /** @fn Sets a flag to make the tuner use OpenMP. Can be used with UseFastMath (compiler options get added together) 
      * WARNING: Overrides any prior user-defined compiler options set with Tuner::SetCompilerOptions.
      */
    void UseOpenMP();

    /** @fn Initializes m_compilerTuning and enables the use of its AddCompilerParameter method. */
    void UseCompilerTuning();

    /** @fn Adds an --inputSize option to m_cli, which expects numDimensions arguments and sets inputSize.
      * @param numDimensions Number of dimensions that the input has.
      * @param inputSize A reference to ktt::DimensionVector which will be modified depending on user input.
      */
    void UseInputSizeOption(int numDimensions, ktt::DimensionVector &inputSize);

    /** @fn Helper method that fills buffers of a numerical type with random values from a uniform distribution.
      * @param buffers The buffers to be filled.
      * @param minimum Default 0. Lower end of the random distribution.
      * @param maximum Default 10. Higher end of the random distribution.
      */
    template <typename T>
    void FillBuffers(const std::vector<std::vector<T>*> &buffers, T minimum = 0, T maximum = 10) 
    {
        std::random_device device;
        std::default_random_engine engine(device());

        static_assert(std::is_arithmetic_v<T>,
                  "FillBuffers accepts only numeric types");
        using Dist = std::conditional_t<
            std::is_floating_point_v<T>,
            std::uniform_real_distribution<T>,
            std::uniform_int_distribution<T>
        >;
        Dist distribution(minimum, maximum);

        for (std::vector<T> *buffer : buffers) 
        {
            for (size_t i = 0; i < buffer->size(); ++i) 
            {
            buffer->at(i) = distribution(engine); 
            }
        } 
    }

    /** @fn Helper method that initializes a simple kernel.
      * @param kernelFunctionName Kernel function name that should be called by KTT when running the kernel.
      * @param kernelName Human-readable kernel name.
      * @param ndRangeDimensions Default global grid size, assuming local group size of 1 in all dimensions.
      * @param arguments Argument IDs. Must be in the same order as what the kernel function expects.
      */
    void InitKernelDefault(const std::string &kernelFunctionName, const std::string &kernelName,
                           const ktt::DimensionVector &ndRangeDimensions, const std::vector<ktt::ArgumentId> &arguments);

private:
    bool m_useFastMath = false;
    bool m_useOpenMP = false;
    bool m_warnedFastMath = false;
    bool m_warnedOpenMP = false;

    void CheckTunerFlags();

    void RunDynamic();
    void RunOffline();

    void PrintProgress(const std::string& phaseName, int currentRun, double elapsedSeconds,
                       double timeBudget, double bestDuration, double throughput);

    RunStats RunTuningPhase(const std::chrono::steady_clock::time_point& startTime,
                            double timeBudgetSeconds, int printInterval);
    RunStats RunExecutionPhase(const std::chrono::steady_clock::time_point& startTime,
                               double timeBudgetSeconds, const ktt::KernelConfiguration& bestConfig,
                               int printInterval);

};
