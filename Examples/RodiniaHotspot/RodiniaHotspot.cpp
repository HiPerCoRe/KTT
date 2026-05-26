#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../ExampleReferenceKernel.h"

using namespace std;

struct HotspotConfiguration : ExampleRefKernelConfiguration 
{
  uint64_t simIters = 2;
  uint64_t m_grid_rows = 1024;
  uint64_t m_grid_cols = m_grid_rows;
  string inTempFile = "";
  string inPowerFile = "";
  string outTempFile = "";
};

void SetUpHotspotOptions(vector<CliOption> &options, HotspotConfiguration &config)
{
  options.emplace_back([&config](const vector<string> &args) {
    config.simIters = stoul(args[0]);
  }, "--simIters", "Set the number of simulation iterations (expects int)", "<iterNum>", 1);
  options.emplace_back([&config](const vector<string> &args) {
    config.m_grid_rows = stoul(args[0]);
    config.m_grid_cols = stoul(args[1]);
  }, "--gridSize", "Set the grid m_size (expects int, int)", "<rows> <cols>", 2);
  options.emplace_back([&config](const vector<string> &args) {
    config.inTempFile = args[0];
    config.inPowerFile = args[1];
    config.outTempFile = args[2];
  }, "--files", "Sets the files input temperature and m_power, and output temperature (expects string, string, string)", "<inTempFile> <inPowerFile> <outTempFile>", 3);
}

HotspotConfiguration HotspotProcessInput(int argc, char **argv) 
{
  HotspotConfiguration config;
  vector<CliOption> options;
  SetUpCommonOptions(options, &config);
  SetUpRefKernelOption(options, config);
  SetUpHotspotOptions(options, config);

  IterateArguments(argc, argv, options);

  return config;
}

class Hotspot : public ExampleReferenceKernel {
  Hotspot(shared_ptr<HotspotConfiguration> config, int defaultProblemSize, string exampleFolderPath, string defaultKernelFileBaseName, string defaultReferenceKernelFileBaseName):
    ExampleReferenceKernel(config, defaultProblemSize, exampleFolderPath, defaultKernelFileBaseName, defaultReferenceKernelFileBaseName)
  {
    m_totalIterations = config->simIters;
    m_grid_rows = config->m_grid_rows;
    m_grid_cols = config->m_grid_cols;
    m_inTempFile = config->inTempFile.empty() ? GetKernelFilePath(exampleFolderPath, "Data/temp_1024", "") : config->inTempFile;
    m_inPowerFile = config->inPowerFile.empty() ? GetKernelFilePath(exampleFolderPath, "Data/power_1024", "") : config->inTempFile;
    m_outTempFile = config->outTempFile.empty() ? "out.txt" : config->inTempFile;
  }
public:
  static std::unique_ptr<Hotspot> Create(
    int argc, char** argv, 
    int defaultProblemSize,
    std::string exampleFolderPath,
    std::string defaultKernelFileBaseName, 
    std::string defaultRefKernelFileBaseName
  ) {
    auto config = std::make_shared<HotspotConfiguration>(HotspotProcessInput(argc, argv));
    std::unique_ptr<Hotspot> ex(new Hotspot(config, defaultProblemSize, exampleFolderPath, defaultKernelFileBaseName, defaultRefKernelFileBaseName));
    ex->PostInitialize();
    return ex;
  }

protected:

  int m_grid_rows,m_grid_cols;
  string m_inTempFile, m_inPowerFile, m_outTempFile;

  const uint64_t m_pyramidHeight = 2;
  uint64_t m_totalIterations;
  int m_iteration = 0;

  /* maximum m_power density possible (say 300W for a 10mm x 10mm chip)	*/
  const float MAX_PD = 3.0e6;
  /* required precision in degrees	*/
  const float PRECISION = 0.001f;
  const float SPEC_HEAT_SI = 1.75e6;
  const int K_SI = 100;
  /* capacitance fitting factor	*/
  const float FACTOR_CHIP = 0.5f;

  /* chip parameters	*/
  const float t_chip = 0.0005f;
  const float chip_height = 0.016f;
  const float chip_width = 0.016f;
  /* ambient temperature, assuming no package at all	*/
  const float amb_temp = 80.0f;

  const uint64_t BLOCK_SIZE_REF = 16; //has to be the same value as in reference kernel
  const uint64_t PYRAMID_HEIGHT_REF = 2;
  const uint64_t BLOCK_SIZE_C = BLOCK_SIZE_REF;
  const uint64_t BLOCK_SIZE_R = BLOCK_SIZE_REF;

  ktt::ArgumentId m_iterationId;
  ktt::ArgumentId m_powerId;
  ktt::ArgumentId m_tempSrcId;
  ktt::ArgumentId m_tempDstId;
  ktt::ArgumentId m_grid_colsId;
  ktt::ArgumentId m_grid_rowsId;

  ktt::ArgumentId m_borderColsId;
  ktt::ArgumentId m_borderRowsId;
  ktt::ArgumentId m_CapId;
  ktt::ArgumentId m_RxId;
  ktt::ArgumentId m_RyId;
  ktt::ArgumentId m_RzId;
  ktt::ArgumentId m_stepId;
    
  int m_borderRows, m_borderCols;
  float m_grid_height, m_grid_width;
  long m_size;

  std::vector<float> m_tempSrc; 
  std::vector<float> m_tempDst; 
  std::vector<float> m_power; 

  float m_Cap;
  float m_Rx, m_Ry, m_Rz;
  float m_max_slope;
  float m_step;

  const int EXPAND_RATE = 2; // add one iteration will extend the pyramid base by 2 per each borderline

  void readinput(std::vector<float> &vect, const std::string& file) {
    readinput(vect.data(), file);
  }

  void readinput(float* vect, const std::string& file) {
    std::ifstream input(file);
    if (!input.is_open()) {
      throw std::runtime_error("The file " + file + " was not opened");
    }

    float val;
    for (int i = 0; i < m_grid_rows; ++i) {
      for (int j = 0; j < m_grid_cols; ++j) {
        if (!(input >> val)) {
          throw std::runtime_error("Invalid file format or unexpected end of file in " + file);
        }
        vect[i * m_grid_cols + j] = val;
      }
    }
  }

  void writeoutput(std::vector<float>& vect, const std::string& file) {
    writeoutput(vect.data(), file);
  }

  void writeoutput(float* vect, const std::string& file) {
    std::ofstream output(file);
    if (!output.is_open()) {
      throw std::runtime_error("The file " + file + " was not opened");
    }

    output << std::fixed << std::setprecision(4);
    for (int i = 0; i < m_grid_rows; ++i) {
      for (int j = 0; j < m_grid_cols; ++j) {
        output << i * m_grid_cols + j << "\t" << vect[i * m_grid_cols + j] << "\n";
      }
    }
  }


  void InitData() override
  {
    #if USE_PROFILING == 1
    printf("Executing with profiling switched ON.\n");
    m_tuner.setKernelProfiling(true);
    #endif

    m_tuner.SetCompilerOptions("-I./");
    m_tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    m_size = m_grid_rows*m_grid_cols;
    // --------------- pyramid parameters --------------- 
    m_grid_height = chip_height / m_grid_rows;
    m_grid_width = chip_width / m_grid_cols;

    m_Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * m_grid_width * m_grid_height;
    m_Rx = m_grid_width / (2.0f * K_SI * t_chip * m_grid_height);
    m_Ry = m_grid_height / (2.0f * K_SI * t_chip * m_grid_width);
    m_Rz = t_chip / (K_SI * m_grid_height * m_grid_width);

    m_max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    m_step = PRECISION / m_max_slope;

    m_tempSrc = std::vector<float>(m_size, 0.0);
    m_tempDst = std::vector<float>(m_size, 0.0);
    m_power = std::vector<float>(m_size, 0.0);

    // Read input data from disk
    readinput(m_tempSrc, m_inTempFile);
    readinput(m_power, m_inPowerFile);
  }

  void InitKernel() override
  {
    // Add all arguments utilized by kernels
    m_iterationId = m_tuner.AddArgumentScalar(m_iteration);
    m_powerId = m_tuner.AddArgumentVector(m_power, ktt::ArgumentAccessType::ReadOnly);
    m_tempSrcId = m_tuner.AddArgumentVector(std::vector<float>(m_tempSrc), ktt::ArgumentAccessType::ReadWrite);
    m_tempDstId = m_tuner.AddArgumentVector(std::vector<float>(m_tempDst), ktt::ArgumentAccessType::ReadWrite);
    m_grid_colsId = m_tuner.AddArgumentScalar(m_grid_cols);
    m_grid_rowsId = m_tuner.AddArgumentScalar(m_grid_rows);

    m_borderColsId = m_tuner.AddArgumentScalar(m_borderCols);
    m_borderRowsId = m_tuner.AddArgumentScalar(m_borderRows);
    m_CapId = m_tuner.AddArgumentScalar(m_Cap);
    m_RxId = m_tuner.AddArgumentScalar(m_Rx);
    m_RyId = m_tuner.AddArgumentScalar(m_Ry);
    m_RzId = m_tuner.AddArgumentScalar(m_Rz);
    m_stepId = m_tuner.AddArgumentScalar(m_step);

    // Total NDRange m_size matches number of grid points
    const ktt::DimensionVector ndRangeDimensions;
    InitKernelDefault("hotspot", "Hotspot", ndRangeDimensions, { m_iterationId, 
        m_powerId, m_tempSrcId, m_tempDstId,
        m_grid_colsId, m_grid_rowsId, m_borderColsId, m_borderRowsId,
        m_CapId, m_RxId, m_RyId, m_RzId, m_stepId });
    
    m_tuner.SetLauncher(m_kernel, [this](ktt::ComputeInterface &interface) {
      const vector<ktt::ParameterPair>& pairs = interface.GetCurrentConfiguration().GetPairs();
      uint64_t blocksizeRows = ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "BLOCK_SIZE_ROWS");
      uint64_t blocksizeCols = ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "BLOCK_SIZE_COLS");
      uint64_t pyramid_height = ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "PYRAMID_HEIGHT");
      uint64_t workGroupY = ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "WORK_GROUP_Y");
      uint64_t smallBlockCol = blocksizeCols - pyramid_height * EXPAND_RATE;
      uint64_t smallBlockRow = blocksizeRows - pyramid_height * EXPAND_RATE;
      uint64_t blockCols = m_grid_cols / smallBlockCol + ((m_grid_cols % smallBlockCol == 0) ? 0 : 1);
      uint64_t blockRows = (m_grid_rows / smallBlockRow) / (blocksizeRows / workGroupY) + ((m_grid_rows % smallBlockRow == 0) ? 0 : 1);
      uint64_t m_borderCols = pyramid_height * EXPAND_RATE / 2;
      uint64_t m_borderRows = pyramid_height * EXPAND_RATE / 2;
      interface.UpdateScalarArgument(m_borderColsId, &m_borderCols);
      interface.UpdateScalarArgument(m_borderRowsId, &m_borderRows);

      const ktt::DimensionVector ndRangeDimensions(blocksizeCols * blockCols, blocksizeRows * blockRows, 1);
      const ktt::DimensionVector workGroupDimensions(blocksizeCols, workGroupY, 1);
      bool srcPosition = true; //if the original pointer to source temperature is in m_tempSrc, true. otherwise false.

      for (uint64_t t = 0; t < m_totalIterations; t += pyramid_height) {

        // Specify kernel arguments
        int iter = static_cast<int>(min(pyramid_height, m_totalIterations - t));
        interface.UpdateScalarArgument(m_iterationId, &iter);
        // run the kernel
        interface.RunKernel(m_definition, ndRangeDimensions, workGroupDimensions);

        if (t + pyramid_height < m_totalIterations) //if there will be another iteration
        {
          // Swap the source and destination temperatures
          interface.SwapArguments(m_definition, m_tempSrcId, m_tempDstId);
          srcPosition = !srcPosition;
        }
      }

      if (!srcPosition)
      {
        interface.CopyBuffer(m_tempDstId, m_tempSrcId, m_size);
      }
    });
  }

  void InitTuningSpace() override
  {
    // Multiply workgroup m_size in dimensions x and y by two parameters that follow (effectively setting workgroup size to parameters' values)
    m_tuner.AddParameter(m_kernel, "BLOCK_SIZE_ROWS", vector<uint64_t>{8, 16, 32, 64});
    m_tuner.AddParameter(m_kernel, "BLOCK_SIZE_COLS", vector<uint64_t>{8, 16, 32, 64});
    m_tuner.AddParameter(m_kernel, "PYRAMID_HEIGHT", vector<uint64_t>{1, 2, 4, 8});
    m_tuner.AddParameter(m_kernel, "WORK_GROUP_Y", vector<uint64_t>{4, 8, 16, 32, 64});
    m_tuner.AddParameter(m_kernel, "LOCAL_MEMORY", vector<uint64_t>{0, 1});
    m_tuner.AddParameter(m_kernel, "LOOP_UNROLL", vector<uint64_t>{0,1});
    // Add conditions
    auto enoughToCompute = [](const std::vector<size_t>& vector) {
      return vector.at(0)/(vector.at(2)*2) > 1 && vector.at(1)/(vector.at(2)*2) > 1;
    };
    m_tuner.AddConstraint(m_kernel, {"BLOCK_SIZE_COLS", "WORK_GROUP_Y", "PYRAMID_HEIGHT"}, enoughToCompute);
    auto workGroupSmaller = [](const std::vector<size_t>& vector) {return vector.at(0)<=vector.at(1);};
    auto workGroupDividable = [](const std::vector<size_t>& vector) {return vector.at(1)%vector.at(0) == 0;};
    m_tuner.AddConstraint(m_kernel, {"WORK_GROUP_Y", "BLOCK_SIZE_ROWS"}, workGroupSmaller);
    m_tuner.AddConstraint(m_kernel, {"WORK_GROUP_Y", "BLOCK_SIZE_ROWS"}, workGroupDividable);
  }

  void InitReference() override
  {
    const ktt::DimensionVector ndRangeDimensions;
    const ktt::DimensionVector workGroupDimensions;
    InitReferenceKernelDefault("hotspot", ndRangeDimensions, workGroupDimensions, { m_iterationId, 
        m_powerId, m_tempSrcId, m_tempDstId,
        m_grid_colsId, m_grid_rowsId, m_borderColsId, m_borderRowsId,
        m_CapId, m_RxId, m_RyId, m_RzId, m_stepId }, {m_tempDstId});

    m_tuner.SetLauncher(m_refKernel, [this](ktt::ComputeInterface &interface) {
      int smallBlockCol = BLOCK_SIZE_REF-PYRAMID_HEIGHT_REF*EXPAND_RATE;
      int smallBlockRow = BLOCK_SIZE_REF-PYRAMID_HEIGHT_REF*EXPAND_RATE;
      int blockCols = m_grid_cols/smallBlockCol+((m_grid_cols%smallBlockCol==0)?0:1);
      int blockRows = m_grid_rows/smallBlockRow+((m_grid_rows%smallBlockRow==0)?0:1);
      int borderCols = PYRAMID_HEIGHT_REF*EXPAND_RATE/2;
      int borderRows = PYRAMID_HEIGHT_REF*EXPAND_RATE/2;

      const ktt::DimensionVector ndRangeDimensions(BLOCK_SIZE_REF*blockCols,BLOCK_SIZE_REF*blockRows, 1);
      const ktt::DimensionVector workGroupDimensions(BLOCK_SIZE_REF, BLOCK_SIZE_REF, 1);
      interface.UpdateScalarArgument(m_borderRowsId, &borderRows);
      interface.UpdateScalarArgument(m_borderColsId, &borderCols);
      bool srcPosition = true; //if the original pointer to source temperature is in m_tempSrc, true. otherwise false.

      for (uint64_t t = 0; t < m_totalIterations; t += PYRAMID_HEIGHT_REF) {

        // Specify kernel arguments
        int iter = static_cast<int>(min(PYRAMID_HEIGHT_REF, m_totalIterations - t));
        interface.UpdateScalarArgument(m_iterationId, &iter);
        // Run the kernel
        interface.RunKernel(m_definition, ndRangeDimensions, workGroupDimensions);

        if (t + PYRAMID_HEIGHT_REF < m_totalIterations) //if there will be another iteration
        {
          // Swap the source and destination temperatures
          interface.SwapArguments(m_definition, m_tempSrcId, m_tempDstId);
          srcPosition = !srcPosition;
        }
      }

      if (!srcPosition)
      {
        interface.CopyBuffer(m_tempDstId, m_tempSrcId, m_size);
      }
    });
  }
};


int main(int argc, char** argv)
{
  unique_ptr<Hotspot> hotspot = Hotspot::Create(
    argc, argv, 0, "Examples/RodiniaHotspot", "Hotspot", "HotspotReference"
  );
  hotspot->Run();

  return 0;
}



