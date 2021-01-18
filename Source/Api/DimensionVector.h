/** @file DimensionVector.h
  * Functionality related to specifying thread sizes of a kernel.
  */
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <Kernel/ModifierAction.h>
#include <Kernel/ModifierDimension.h>
#include <KttPlatform.h>

namespace ktt
{

/** @class DimensionVector
  * Class which holds information about either global or local thread size of a single kernel.
  */
class KTT_API DimensionVector
{
public:
    /** @fn DimensionVector()
      * Default constructor, creates dimension vector with thread sizes in all dimensions set to 1.
      */
    DimensionVector();

    /** @fn explicit DimensionVector(const size_t sizeX)
      * Constructor which creates dimension vector with specified thread size in dimension x and thread sizes in other dimensions set to 1.
      * @param sizeX Thread size in dimension x.
      */
    explicit DimensionVector(const size_t sizeX);

    /** @fn explicit DimensionVector(const size_t sizeX, const size_t sizeY)
      * Constructor which creates dimension vector with specified thread sizes in dimensions x and y and thread size in dimension z set to 1.
      * @param sizeX Thread size in dimension x.
      * @param sizeY Thread size in dimension y.
      */
    explicit DimensionVector(const size_t sizeX, const size_t sizeY);

    /** @fn explicit DimensionVector(const size_t sizeX, const size_t sizeY, const size_t sizeZ)
      * Constructor which creates dimension vector with specified thread sizes in all dimensions.
      * @param sizeX Thread size in dimension x.
      * @param sizeY Thread size in dimension y.
      * @param sizeZ Thread size in dimension z.
      */
    explicit DimensionVector(const size_t sizeX, const size_t sizeY, const size_t sizeZ);

    /** @fn explicit DimensionVector(const std::vector<size_t>& vector)
      * Constructor which creates dimension vector with thread sizes based on up to first three elements of provided vector. If size of vector
      * is less than 3, remaining thread sizes are set to 1.
      * @param vector Source vector for dimension vector thread sizes.
      */
    explicit DimensionVector(const std::vector<size_t>& vector);

    /** @fn void SetSizeX(const size_t sizeX)
      * Setter for thread size in dimension x.
      * @param sizeX Thread size in dimension x.
      */
    void SetSizeX(const size_t sizeX);

    /** @fn void SetSizeY(const size_t sizeY)
      * Setter for thread size in dimension y.
      * @param sizeY Thread size in dimension y.
      */
    void SetSizeY(const size_t sizeY);

    /** @fn void SetSizeZ(const size_t sizeZ)
      * Setter for thread size in dimension z.
      * @param sizeZ Thread size in dimension z.
      */
    void SetSizeZ(const size_t sizeZ);

    /** @fn void SetSize(const ModifierDimension modifierDimension, const size_t size)
    * Setter for thread size in specified dimension.
    * @param modifierDimension Specifies which dimension size will be set.
    * @param size Thread size in specified dimension.
    */
    void SetSize(const ModifierDimension modifierDimension, const size_t size);

    /** @fn void Multiply(const DimensionVector& factor)
      * Multiplies thread sizes by values provided by specified dimension vector.
      * @param factor Source of values for thread size multiplication.
      */
    void Multiply(const DimensionVector& factor);

    /** @fn void Divide(const DimensionVector& divisor)
      * Divides thread sizes by values provided by specified dimension vector.
      * @param divisor Source of values for thread size division.
      */
    void Divide(const DimensionVector& divisor);

    /** @fn void RoundUp(const DimensionVector& multiple)
      * Rounds up thread sizes to be multiple of values provided by specified dimension vector.
      * @param multiple Source of values for thread size round up.
      */
    void RoundUp(const DimensionVector& multiple);

    /** @fn void ModifyByValue(const size_t value, const ModifierAction modifierAction, const ModifierDimension modifierDimension)
      * Modifies thread size in single dimension based on provided value and action.
      * @param value Value which will be used to modify thread size in single dimension based on specified action.
      * @param modifierAction Specifies which operation should be performed with thread size and specified value.
      * @param modifierDimension Specifies which dimension will be affected by the action.
      */
    void ModifyByValue(const size_t value, const ModifierAction modifierAction, const ModifierDimension modifierDimension);

    /** @fn size_t GetSizeX() const
      * Getter for thread size in dimension x.
      * @return Thread size in dimension x.
      */
    size_t GetSizeX() const;

    /** @fn size_t GetSizeY() const
      * Getter for thread size in dimension y.
      * @return Thread size in dimension y.
      */
    size_t GetSizeY() const;

    /** @fn size_t GetSizeZ() const
      * Getter for thread size in dimension z.
      * @return Thread size in dimension z.
      */
    size_t GetSizeZ() const;

    /** @fn size_t GetSize(const ModifierDimension modifierDimension) const
    * Getter for thread size in specified dimension.
    * @param modifierDimension Specifies which dimension size will be returned.
    * @return Thread size in specified dimension.
    */
    size_t GetSize(const ModifierDimension modifierDimension) const;

    /** @fn size_t GetTotalSize() const
      * Getter for total thread size. Total thread size is calculated by multiplying thread sizes in each dimension.
      * @return Total thread size.
      */
    size_t GetTotalSize() const;

    /** @fn std::vector<size_t> GetVector() const
      * Converts dimension vector to STL vector. Resulting vector will always contain 3 elements.
      * @return Converted STL vector.
      */
    std::vector<size_t> GetVector() const;

    /** @fn std::string GetString() const
      * Converts dimension vector to string.
      * @return String in format "(x, y, z)".
      */
    std::string GetString() const;

    /** @fn bool operator==(const DimensionVector& other) const
      * Comparison operator for dimension vector. Compares thread sizes in all 3 dimensions.
      * @return True if dimension vectors are equal. False otherwise.
      */
    bool operator==(const DimensionVector& other) const;

    /** @fn bool operator!=(const DimensionVector& other) const
      * Comparison operator for dimension vector. Compares thread sizes in all 3 dimensions.
      * @return True if dimension vectors are not equal. False otherwise.
      */
    bool operator!=(const DimensionVector& other) const;

private:
    size_t m_SizeX;
    size_t m_SizeY;
    size_t m_SizeZ;

    void AddValue(const size_t value, const ModifierDimension modifierDimension);
    void SubtractValue(const size_t value, const ModifierDimension modifierDimension);
    void MultiplyByValue(const size_t value, const ModifierDimension modifierDimension);
    void DivideByValue(const size_t value, const ModifierDimension modifierDimension);
    void DivideCeilByValue(const size_t value, const ModifierDimension modifierDimension);
};

} // namespace ktt
