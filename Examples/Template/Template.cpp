#include "ExampleBase.h"
// Include these and derive from them if Example uses 
// a reference kernel/computation.
// #include "ExampleReferenceKernel.h"
// #include "ExampleReferenceComputation.h"

using namespace std;

class Template : public ExampleBase
{
protected:
    Template(int argc, char **argv,
             string exampleFolderPath, string defaultKernelFileBaseName) :
        ExampleBase(argc, argv, exampleFolderPath, defaultKernelFileBaseName)
    {
        // UseFastMath();
        // UseOpenMP();  // Mainly in case of C++ tuning. See CoulombSum3d.
        // UseCompilerTuning();

        // Do not manipulate m_tuner here, it is initialized after ProcessCLI; the earliest overridable method in which m_tuner is valid is InitData.
        // See annotation for ExampleBase::InitTuner().
    }

    // This friend declaration allows the Create() factory method to access protected constructor.
    // When inheriting ExampleReferenceKernel friend that class, it has a different Create().
    friend ExampleBase;

    // Argument IDs and buffers will likely need to be class members.

    void InitCLI() override 
    {
        // Customize CLI options for this Example, call base to include the default ones.
        ExampleBase::InitCLI();

        // Description is automatically added to the --help text.
        // m_cli.AddOption({[this](const vector<string> &args){
        //         m_kSizeM = stoi(args[0]);
        //         m_kSizeN = stoi(args[1]);
        //         m_kSizeK = stoi(args[2]);
        //     }, "--matSizes", "Sets matrix sizes (expects 3 ints). Matrix sizes will be:"
        //     "\n\tA: M x K\n\tB: K x N\n\tC: M x N", "<M> <N> <K>", 3
        // });
        // UseInputSizeOption(3, m_inputSize);
    }

    void InitData() override
    {
        // Resize data buffers based on CLI parameters
        // Example:
        // m_inputData.resize(m_inputSize);
        // m_outputData.resize(m_inputSize);

        // Initialize data
        // Optionally use the FillBuffers method.
        // Example:
        // FillBuffers<float>({&m_inputData}, 0.0f, 100.0f);
        // FillBuffers<int>({&m_indices}, 0, m_problemSize - 1);
    }

    void InitKernel() override
    {
        // Register kernel arguments.
        // Load the kernel from the kernel file (m_kernelFile).
        // The kernel file should be in the same folder as this .cpp file.

        // Optional use of the InitKernelDefault method.
        // Example:
        // const ktt::DimensionVector ndRangeDimensions(m_problemSize);
        // InitKernelDefault("myKernelFunction", "My Kernel Name", ndRangeDimensions,
        //                   {m_inputId, m_outputId, m_sizeId});
    }

    void InitTuningSpace() override
    {
        // Add tunable parameters for the kernel.

        // Add constraints to prune invalid parameter combinations.

        // Add thread modifiers to adjust global/local work sizes.

        // If UseCompilerTuning() was called, you may:
        // m_compilerTuning->AddCompilerParameter("-O", {"1", "2", "3"}); // if tuning Cpp kernel
        // m_compilerTuning->AddCompilerParameter("-cl-finite-math-only"); // if OpenCL kernel
        // m_compilerTuning->AddCompilerParameter("--maxrregcount ", {"0", "32", "40", "48", "64"}); // if CUDA
        // etc.
        // See documentation for details. Compiler parameters can be tuned with the others or separately.
    }

    /*
    // If using ExampleReference*, also implement InitReference():
    void InitReference() override
    {
        // Optional use of InitReferenceKernelDefault when applicable:
        // Example:
        // const ktt::DimensionVector ndRangeDimensions(m_problemSize);
        // const ktt::DimensionVector workGroupDimensions(256);
        // InitReferenceKernelDefault("referenceKernel", ndRangeDimensions, workGroupDimensions,
        //                            {m_inputId, m_outputId}, {m_outputId});
    }
    */
};

int main(int argc, char **argv)
{
    unique_ptr<Template> example = Template::Create<Template>(
        argc, argv, "Examples/Template", "Template"/*, "TemplateReference" // when inheriting ExampleReferenceKernel*/
    );

    example->Run();

    return 0;
}
