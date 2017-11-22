/** @file tuning_manipulator.h
  * @brief Functionality related to customizing kernel runs with tuning manipulator.
  */
#pragma once

#include <cstddef>
#include <utility>
#include <vector>
#include "ktt_platform.h"
#include "ktt_types.h"
#include "api/dimension_vector.h"
#include "api/parameter_pair.h"

namespace ktt
{

class TuningRunner;
class ManipulatorInterface;

/** @class TuningManipulator
  * @brief Class which can be used to customize kernel launch in order to run some part of computation on CPU, utilize iterative kernel launches,
  * kernel compositions and more. In order to use this functionality, new class which publicly inherits from tuning manipulator class has to be
  * defined.
  */
class KTT_API TuningManipulator
{
public:
    /** @fn virtual ~TuningManipulator()
      * @brief Tuning manipulator destructor. Inheriting class can override destructor with custom implementation. Default implementation is
      * provided by KTT library.
      */
    virtual ~TuningManipulator();

    /** @fn virtual void launchComputation(const KernelId id) = 0
      * @brief This method is responsible for directly running the computation and ensuring that correct results are computed. It may utilize any
      * other method inside the tuning manipulator as well as any user-defined methods. Any other tuning manipulator methods run from this method
      * only affect current invocation of launchComputation() method. Inheriting class must provide implementation for this method.
      *
      * When tuning manipulator is used, total execution duration is calculated from two components. First component is the sum of execution times
      * of all kernel launches inside this method. Second component is the execution time of the method itself, minus the execution times of kernel
      * launches. Initial buffer transfer times are not included in the total duration, same as in the case of kernel tuning without manipulator.
      * Other buffer update and retrieval times are included in the second component.
      * @param id Id of a kernel or kernel composition which was used to launch kernel from tuner API.
      */
    virtual void launchComputation(const KernelId id) = 0;

    /** @fn virtual bool enableArgumentPreload() const
      * @brief Controls whether arguments for all kernels that are part of manipulator will be automatically uploaded to corresponding compute API
      * buffers before any kernel is run in the current invocation of launchComputation() method. Argument preload is turned on by default.
      *
      * Turning this behavior off is useful when utilizing kernel compositions where different kernels use different arguments which would not all
      * fit into available memory. Buffer creation and deletion can be then controlled by using createArgumentBuffer() and destroyArgumentBuffer()
      * methods for corresponding arguments. Any leftover arguments after launchComputation() method finishes will still be automatically cleaned up.
      * Inheriting class can override this method. 
      * @return Flag which controls whether the argument preload is enabled or not.
      */
    virtual bool enableArgumentPreload() const;

    /** @fn void runKernel(const KernelId id)
      * @brief Runs kernel with specified id using thread sizes based on the current configuration.
      * @param id Id of kernel which is run. It must either match the id used to launch kernel from tuner API or be included inside composition
      * which was launched from tuner API.
      */
    void runKernel(const KernelId id);

    /** @fn void runKernel(const KernelId id, const DimensionVector& globalSize, const DimensionVector& localSize)
      * @brief Runs kernel with specified id using specified thread sizes.
      * @param id Id of kernel which is run. It must either match the id used to launch kernel from tuner API or be included inside composition which
      * was launched from tuner API.
      * @param globalSize Dimensions for global size with which the kernel is run.
      * @param localSize Dimensions for local size with which the kernel is run.
      */
    void runKernel(const KernelId id, const DimensionVector& globalSize, const DimensionVector& localSize);

    /** @fn DimensionVector getCurrentGlobalSize(const KernelId id) const
      * @brief Returns global thread size of specified kernel based on the current configuration.
      * @param id Id of kernel for which the global size is retrieved. It must either match the id used to launch kernel from tuner API or
      * be included inside composition which was launched from tuner API.
      * @return Global thread size of specified kernel.
      */
    DimensionVector getCurrentGlobalSize(const KernelId id) const;

    /** @fn DimensionVector getCurrentLocalSize(const KernelId id) const
      * @brief Returns local thread size of specified kernel based on the current configuration.
      * @param id Id of kernel for which the local size is retrieved. It must either match the id used to launch kernel from tuner API or
      * be included inside composition which was launched from tuner API.
      * @return Local thread size of specified kernel.
      */
    DimensionVector getCurrentLocalSize(const KernelId id) const;

    /** @fn std::vector<ParameterPair> getCurrentConfiguration() const
      * @brief Returns configuration used inside current invocation of launchComputation() method.
      * @return Current configuration. See ::ParameterPair for more information.
      */
    std::vector<ParameterPair> getCurrentConfiguration() const;

    /** @fn void updateArgumentScalar(const ArgumentId id, const void* argumentData)
      * @brief Updates specified scalar argument.
      * @param id Id of scalar argument which is updated.
      * @param argumentData Pointer to new data for scalar argument. Data types for old and new data have to match.
      */
    void updateArgumentScalar(const ArgumentId id, const void* argumentData);

    /** @fn void updateArgumentLocal(const ArgumentId id, const size_t numberOfElements)
      * @brief Updates specified local memory argument.
      * @param id Id of local memory argument which is updated.
      * @param numberOfElements Number of local memory elements inside updated argument. Data types for old and new data match.
      */
    void updateArgumentLocal(const ArgumentId id, const size_t numberOfElements);

    /** @fn void updateArgumentVector(const ArgumentId id, const void* argumentData)
      * @brief Updates specified vector argument. Does not modify argument size.
      * @param id Id of vector argument which is updated.
      * @param argumentData Pointer to new data for vector argument. Number of elements and data types for old and new data have to match.
      */
    void updateArgumentVector(const ArgumentId id, const void* argumentData);

    /** @fn void updateArgumentVector(const ArgumentId id, const void* argumentData, const size_t numberOfElements)
      * @brief Updates specified vector argument. Possibly also modifies argument size.
      * @param id Id of vector argument which is updated.
      * @param argumentData Pointer to new data for vector argument. Data types for old and new data have to match.
      * @param numberOfElements Number of elements inside updated vector argument.
      */
    void updateArgumentVector(const ArgumentId id, const void* argumentData, const size_t numberOfElements);

    /** @fn void getArgumentVector(const ArgumentId id, void* destination) const
      * @brief Retrieves specified vector argument.
      * @param id Id of vector argument which is retrieved.
      * @param destination Pointer to destination where vector argument data will be copied. Destination buffer size needs to be equal or greater
      * than argument size.
      */
    void getArgumentVector(const ArgumentId id, void* destination) const;

    /** @fn void getArgumentVector(const ArgumentId id, void* destination, const size_t numberOfElements) const
      * @brief Retrieves part of specified vector argument.
      * @param id Id of vector argument which is retrieved.
      * @param destination Pointer to destination where vector argument data will be copied. Destination buffer size needs to be equal or greater
      * than size of specified number of elements.
      * @param numberOfElements Number of elements which will be copied to specified destination, starting with first element.
      */
    void getArgumentVector(const ArgumentId id, void* destination, const size_t numberOfElements) const;

    /** @fn void changeKernelArguments(const KernelId id, const std::vector<ArgumentId>& argumentIds)
      * @brief Changes kernel arguments for specified kernel by providing corresponding argument ids.
      * @param id Id of kernel for which the arguments are changed.
      * @param argumentIds Ids of arguments to be used by specified kernel. Order of ids must match the order of kernel arguments specified in kernel
      * function. Argument ids for single kernel must be unique.
      */
    void changeKernelArguments(const KernelId id, const std::vector<ArgumentId>& argumentIds);

    /** @fn void swapKernelArguments(const KernelId id, const ArgumentId argumentIdFirst, const ArgumentId argumentIdSecond)
      * @brief Swaps positions of specified kernel arguments for specified kernel.
      * @param id Id of kernel for which the arguments are swapped.
      * @param argumentIdFirst Id of the first argument which is swapped.
      * @param argumentIdSecond Id of the second argument which is swapped.
      */
    void swapKernelArguments(const KernelId id, const ArgumentId argumentIdFirst, const ArgumentId argumentIdSecond);

    /** @fn void createArgumentBuffer(const ArgumentId id)
      * @brief Transfers specified kernel argument to a buffer from which it can be accessed by compute API. This method should be utilized only if
      * argument preload is disabled. See enableArgumentPreload() for more information.
      * @param id Id of argument for which the buffer is created.
      */
    void createArgumentBuffer(const ArgumentId id);

    /** @fn void destroyArgumentBuffer(const ArgumentId id)
      * @brief Destroys compute API buffer for specified kernel argument. This method should be utilized only if argument preload is disabled.
      * See enableArgumentPreload() for more information.
      * @param id Id of argument for which the buffer is destroyed.
      */
    void destroyArgumentBuffer(const ArgumentId id);

    /** @fn static size_t getParameterValue(const std::string& parameterName, const std::vector<ParameterPair>& parameterPairs)
      * @brief Returns integer value of specified parameter from provided vector of parameters.
      * @return Integer value of specified parameter.
      */
    static size_t getParameterValue(const std::string& parameterName, const std::vector<ParameterPair>& parameterPairs);

    /** @fn static double getParameterValueDouble(const std::string& parameterName, const std::vector<ParameterPair>& parameterPairs)
      * @brief Returns floating-point value of specified parameter from provided vector of parameters.
      * @return Floating-point value of specified parameter.
      */
    static double getParameterValueDouble(const std::string& parameterName, const std::vector<ParameterPair>& parameterPairs);

    friend class TuningRunner;

private:
    ManipulatorInterface* manipulatorInterface;
};

} // namespace ktt
