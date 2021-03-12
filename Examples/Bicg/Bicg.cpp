#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <Ktt.h>

#if defined(_MSC_VER)
const std::string kernelPrefix = "";
#else
const std::string kernelPrefix = "../";
#endif

#if KTT_CUDA_EXAMPLE
    const std::string defaultKernelFile = kernelPrefix + "../Examples/Bicg/Bicg.cu";
    const std::string defaultReferenceKernelFile = kernelPrefix + "../Examples/Bicg/BicgReference.cu";
    const auto computeApi = ktt::ComputeApi::CUDA;
#elif KTT_OPENCL_EXAMPLE
    const std::string defaultKernelFile = kernelPrefix + "../Examples/Bicg/Bicg.cl";
    const std::string defaultReferenceKernelFile = kernelPrefix + "../Examples/Bicg/BicgReference.cl";
    const auto computeApi = ktt::ComputeApi::OpenCL;
#endif

	// Toggle rapid test (e.g., disable output validation).
const bool rapidTest = false;

// Toggle kernel profiling.
const bool useProfiling = false;

// Add denser values to tuning parameters (useDenseParameters = true).
const bool useDenseParameters = false;

// Add wider ranges of tuning parameters (useWideParameters  = true).
const bool useWideParameters = false;

int main(int argc, char** argv)
{
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = defaultKernelFile;
    std::string referenceKernelFile = defaultReferenceKernelFile;

    if (argc >= 2)
    {
        platformIndex = std::stoul(std::string(argv[1]));

        if (argc >= 3)
        {
            deviceIndex = std::stoul(std::string(argv[2]));

            if (argc >= 4)
            {
                kernelFile = std::string(argv[3]);

                if (argc >= 5)
                {
                    referenceKernelFile = std::string(argv[4]);
                }
            }
        }
    }

	int m;
	int n;

    // Problem size
    if constexpr (!useProfiling)
    {
		m = 16384;
		n = 16384;
    }
    else
    {
        m = 4096;
        n = 4096;
    }

    // Thread dimensions
    const int WORK_GROUP_X = 256;
    const int WORK_GROUP_Y = 1;

    // New NVidia GPUs have max. work-group size of 1024
    const int MAX_WORK_GROUP_SIZE = 1024;

	// Declare kernel parameters
	const ktt::DimensionVector ndRangeDimensions(m, n / 64); // replaced in manipulator
	const ktt::DimensionVector workGroupDimensions(32, 4); // replaced in manipulator
	const ktt::DimensionVector referenceNdRangeDimensions1(static_cast<size_t>(ceil(n / WORK_GROUP_X)) * WORK_GROUP_X, WORK_GROUP_Y);
	const ktt::DimensionVector referenceNdRangeDimensions2(static_cast<size_t>(ceil(m / WORK_GROUP_X)) * WORK_GROUP_X, WORK_GROUP_Y);
	const ktt::DimensionVector referenceWorkGroupDimensions(WORK_GROUP_X, WORK_GROUP_Y);

	// Declare data variables
	std::vector<float> A(n * m);
	std::vector<float> x1(m);
	std::vector<float> x2(n);
	// larger versions of vectors needed for ATOMICS == 0, they are reduced later in separate kernels
	std::vector<float> y1(n * m / 16, 0.0f); // 16 is the lowest TILE size, so M/16 is maximum number of x-blocks
	std::vector<float> y2(m * n / 128, 0.0f); // 128 is the lowest ROWS_PROCESSED value, so N/128 is maximum number of y-blocks

	// Initialize data
	std::random_device device;
	std::default_random_engine engine(device());
	std::uniform_real_distribution<float> distribution(0.0f, 100.0f);

	for (int j = 0; j < m; ++j)
	{
		x1[j] = distribution(engine);
	}
		
	for (int i = 0; i < n; ++i)
	{
		x2[i] = distribution(engine);

		for (int j = 0; j < m; ++j)
		{
			A[i * m + j] = distribution(engine);
		}
	}

	// Create tuner object for specified platform and device
	ktt::Tuner tuner(platformIndex, deviceIndex, computeApi);
    tuner.SetGlobalSizeType(ktt::GlobalSizeType::OpenCL);
	tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

	if constexpr (useProfiling)
	{
        printf("Executing with profiling switched ON.\n");
        tuner.SetProfiling(true);
	}

	// Add all arguments utilized by kernels
	const ktt::ArgumentId AId = tuner.AddArgumentVector(A, ktt::ArgumentAccessType::ReadWrite);
	const ktt::ArgumentId x1Id = tuner.AddArgumentVector(x1, ktt::ArgumentAccessType::ReadOnly);
	const ktt::ArgumentId x2Id = tuner.AddArgumentVector(x2, ktt::ArgumentAccessType::ReadOnly);
	const ktt::ArgumentId y1Id = tuner.AddArgumentVector(y1, ktt::ArgumentAccessType::ReadWrite);
	const ktt::ArgumentId y2Id = tuner.AddArgumentVector(y2, ktt::ArgumentAccessType::ReadWrite);
	const ktt::ArgumentId mFusedRefId = tuner.AddArgumentScalar(m);
	const ktt::ArgumentId nFusedRefId = tuner.AddArgumentScalar(256);
	const ktt::ArgumentId mRefId = tuner.AddArgumentScalar(m);
	const ktt::ArgumentId nRefId = tuner.AddArgumentScalar(n);

	// Add kernel definitions to tuner, create a composite kernel
	const ktt::KernelDefinitionId definitionFused = tuner.AddKernelDefinitionFromFile("bicgFused", kernelFile, ndRangeDimensions,
		workGroupDimensions);
	const ktt::KernelDefinitionId definitionReduction1 = tuner.AddKernelDefinitionFromFile("bicgReduction1", kernelFile,
		referenceNdRangeDimensions1, referenceWorkGroupDimensions);
	const ktt::KernelDefinitionId definitionReduction2 = tuner.AddKernelDefinitionFromFile("bicgReduction2", kernelFile,
		referenceNdRangeDimensions1, referenceWorkGroupDimensions);
	const ktt::KernelDefinitionId definitionFusedReference = tuner.AddKernelDefinitionFromFile("bicgFusedRef", referenceKernelFile,
		ndRangeDimensions, workGroupDimensions);
	const ktt::KernelDefinitionId definition1 = tuner.AddKernelDefinitionFromFile("bicgKernel1", referenceKernelFile, referenceNdRangeDimensions1,
		referenceWorkGroupDimensions);
	const ktt::KernelDefinitionId definition2 = tuner.AddKernelDefinitionFromFile("bicgKernel2", referenceKernelFile, referenceNdRangeDimensions2,
		referenceWorkGroupDimensions);
	
	const ktt::KernelId kernel = tuner.CreateCompositeKernel("BicgPolyBenchAndFused", {definition1, definition2, definitionFused,
		definitionFusedReference, definitionReduction1, definitionReduction2}, [definition1, definition2, definitionFused,
		definitionFusedReference, definitionReduction1, definitionReduction2](ktt::ComputeInterface& interface)
	{
		const std::vector<ktt::ParameterPair>& parameterValues = interface.GetCurrentConfiguration().GetPairs();

        if (ktt::ParameterPair::GetParameterValue<uint64_t>(parameterValues, "FUSED") == 2)
		{
			if constexpr (!useProfiling)
			{
				interface.RunKernel(definitionFused);
			}
			else
			{
				interface.RunKernelWithProfiling(definitionFused);
			}

            if (ktt::ParameterPair::GetParameterValue<uint64_t>(parameterValues, "ATOMICS") == 0)
			{
				interface.RunKernel(definitionReduction1);
				interface.RunKernel(definitionReduction2);
            }
		}
        else if (ktt::ParameterPair::GetParameterValue<uint64_t>(parameterValues, "FUSED") == 1)
		{
			interface.RunKernel(definitionFusedReference);
        }
        else
		{
            interface.RunKernel(definition1);
			interface.RunKernel(definition2);
        }
	});
	
	// Add parameters to tuned kernel
	if constexpr (useProfiling)
	{
		tuner.AddParameter(kernel, "FUSED", std::vector<uint64_t>{/*0, 1,*/ 2}); // non-optimized kernels are not profiled
	}
	else
	{
		tuner.AddParameter(kernel, "FUSED", std::vector<uint64_t>{/*0, 1,*/ 2});
	}

	tuner.AddParameter(kernel, "BICG_BATCH", std::vector<uint64_t>{1, 2, 4, 8, 16, 32, 64});
	tuner.AddParameter(kernel, "USE_SHARED_MATRIX", std::vector<uint64_t>{0, 1});
	tuner.AddParameter(kernel, "USE_SHARED_VECTOR_1", std::vector<uint64_t>{0, 1});
	tuner.AddParameter(kernel, "USE_SHARED_VECTOR_2", std::vector<uint64_t>{0, 1});
	tuner.AddParameter(kernel, "USE_SHARED_REDUCTION_1", std::vector<uint64_t>{0, 1});
	tuner.AddParameter(kernel, "USE_SHARED_REDUCTION_2", std::vector<uint64_t>{0, 1});
	tuner.AddParameter(kernel, "ATOMICS", std::vector<uint64_t>{0, 1});
	tuner.AddParameter(kernel, "UNROLL_BICG_STEP", std::vector<uint64_t>{0, 1});
	tuner.AddParameter(kernel, "ROWS_PROCESSED", std::vector<uint64_t>{128, 256, 512, 1024});
	tuner.AddParameter(kernel, "TILE", std::vector<uint64_t>{16, 32, 64});

    // Specify thread modifiers
    auto globalModifierX = [m](const uint64_t, const std::vector<uint64_t>&) {return m;};
    auto globalModifierY = [n](const uint64_t, const std::vector<uint64_t>& vector) {return n / vector.at(0) * vector.at(1) / vector.at(2);};
    auto localModifierX = [](const uint64_t, const std::vector<uint64_t>& vector) {return vector.at(0);};
    auto localModifierY = [](const uint64_t, const std::vector<uint64_t>& vector) {return vector.at(0) / vector.at(1);};

	tuner.AddThreadModifier(kernel, {definitionFused}, ktt::ModifierType::Global, ktt::ModifierDimension::X, {}, globalModifierX);
    tuner.AddThreadModifier(kernel, {definitionFused}, ktt::ModifierType::Global, ktt::ModifierDimension::Y,
		{"ROWS_PROCESSED", "TILE", "BICG_BATCH"}, globalModifierY);
    tuner.AddThreadModifier(kernel, {definitionFused}, ktt::ModifierType::Local, ktt::ModifierDimension::X, {"TILE"}, localModifierX);
    tuner.AddThreadModifier(kernel, {definitionFused}, ktt::ModifierType::Local, ktt::ModifierDimension::Y, {"TILE", "BICG_BATCH"},
		localModifierY);

	// Specify constraints
	// All of the parameters are used only in the fused kernel
	auto fused = [](const std::vector<uint64_t>& vector) {return vector.at(0) == 2 || ((vector.at(0) == 0 || vector.at(0) == 1) && vector.at(1) == 4 && vector.at(2) == 1 && vector.at(3) == 1 && vector.at(4) == 1 && vector.at(5) == 1 && vector.at(6) == 1 && vector.at(7) == 1 && vector.at(8) == 1 && vector.at(9) == 512 && vector.at(10) == 32); };
	tuner.AddConstraint(kernel, {"FUSED", "BICG_BATCH", "USE_SHARED_MATRIX", "USE_SHARED_VECTOR_1", "USE_SHARED_VECTOR_2", "USE_SHARED_REDUCTION_1", "USE_SHARED_REDUCTION_2", "ATOMICS", "UNROLL_BICG_STEP", "ROWS_PROCESSED", "TILE"}, fused);
	// New NVidia GPUs have max. workgroup size of 1024, so   tile_x * tile_y <= 1024   ==>   tile_x * (tile_x / batch) <= 1024 and batch <= tile
	auto maxWgSize = [MAX_WORK_GROUP_SIZE](const std::vector<uint64_t>& vector) {return (vector.at(0) * vector.at(0) / vector.at(1) <= MAX_WORK_GROUP_SIZE) && (vector.at(1) <= vector.at(0)); };
	tuner.AddConstraint(kernel, {"TILE", "BICG_BATCH"}, maxWgSize);

	tuner.SetArguments(definitionFused, {AId, x1Id, y1Id, x2Id, y2Id, mRefId, nRefId});
	tuner.SetArguments(definitionReduction1, {mRefId, nRefId, y1Id});
	tuner.SetArguments(definitionReduction2, {mRefId, nRefId, y2Id});
	tuner.SetArguments(definitionFusedReference, {AId, x2Id, y2Id, x1Id, y1Id, nFusedRefId, mFusedRefId}); // reference fused kernel uses swapped M and N. same for x1/x2 and y1/y2
	tuner.SetArguments(definition1, {AId, x1Id, y1Id, mRefId, nRefId});
	tuner.SetArguments(definition2, {AId, x2Id, y2Id, mRefId, nRefId});

	if constexpr (useProfiling)
	{
		tuner.SetProfiledDefinitions(kernel, {definitionFused});
	}

	if constexpr (!rapidTest)
	{
        tuner.SetValidationMethod(ktt::ValidationMethod::SideBySideRelativeComparison, 0.001);
        tuner.SetValidationRange(y1Id, n);
        tuner.SetValidationRange(y2Id, m);

		tuner.SetReferenceComputation(y1Id, [m, n, &A, &x1](void* buffer)
		{
			float* y1 = static_cast<float*>(buffer);

			for (int i = 0; i < n; ++i)
			{
				y1[i] = 0.0f;

				for (int j = 0; j < m; ++j)
				{
					y1[i] = y1[i] + A[i * m + j] * x1[j];
				}
			}
		});

        tuner.SetReferenceComputation(y2Id, [m, n, &A, &x2](void* buffer)
        {
			float* y2 = static_cast<float*>(buffer);

            for (int i = 0; i < m; ++i)
            {
                y2[i] = 0.0f;
            }

            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < m; ++j)
                {
                    y2[j] = y2[j] + x2[i] * A[i * m + j];
                }
            }
        });
	}

	// Launch kernel tuning
	const auto results = tuner.TuneKernel(kernel);
	tuner.SaveResults(results, "BicgOutput", ktt::OutputFormat::JSON);

	return 0;
}
