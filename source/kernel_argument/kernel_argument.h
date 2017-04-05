#pragma once

#include <vector>

#include "../enum/argument_data_type.h"
#include "../enum/argument_memory_type.h"
#include "../enum/argument_quantity.h"

namespace ktt
{

class KernelArgument
{
public:
    template <typename T> explicit KernelArgument(const size_t id, const std::vector<T>& data, const ArgumentMemoryType& argumentMemoryType,
        const ArgumentQuantity& argumentQuantity):
        id(id),
        argumentMemoryType(argumentMemoryType),
        argumentQuantity(argumentQuantity)
    {
        initializeData(data);
    }

    template <typename T> void updateData(const std::vector<T>& data, const ArgumentQuantity& argumentQuantity)
    {
        if (typeid(T) == typeid(double) && argumentDataType != ArgumentDataType::Double
            || typeid(T) == typeid(float) && argumentDataType != ArgumentDataType::Float
            || typeid(T) == typeid(int) && argumentDataType != ArgumentDataType::Int)
        {
            throw std::runtime_error("Updated data provided for kernel argument have different data type");
        }

        this->argumentQuantity = argumentQuantity;
        initializeData(data);
    }

    size_t getId() const
    {
        return id;
    }

    const void* getData() const
    {
        switch (argumentDataType)
        {
        case ArgumentDataType::Double:
            return (void*)dataDouble.data();
        case ArgumentDataType::Float:
            return (void*)dataFloat.data();
        default:
            return (void*)dataInt.data();
        }
    }

    std::vector<double> getDataDouble() const
    {
        return dataDouble;
    }

    std::vector<float> getDataFloat() const
    {
        return dataFloat;
    }

    std::vector<int> getDataInt() const
    {
        return dataInt;
    }

    size_t getDataSize() const
    {
        switch (argumentDataType)
        {
        case ArgumentDataType::Double:
            return dataDouble.size() * sizeof(double);
        case ArgumentDataType::Float:
            return dataFloat.size() * sizeof(float);
        default:
            return dataInt.size() * sizeof(int);
        }
    }

    ArgumentDataType getArgumentDataType() const
    {
        return argumentDataType;
    }

    ArgumentMemoryType getArgumentMemoryType() const
    {
        return argumentMemoryType;
    }

    ArgumentQuantity getArgumentQuantity() const
    {
        return argumentQuantity;
    }

private:
    size_t id;
    std::vector<double> dataDouble;
    std::vector<float> dataFloat;
    std::vector<int> dataInt;
    ArgumentDataType argumentDataType;
    ArgumentMemoryType argumentMemoryType;
    ArgumentQuantity argumentQuantity;

    template <typename T> void initializeData(const std::vector<T>& data)
    {
        if (data.size() == 0)
        {
            throw std::runtime_error("Data provided for kernel argument is empty");
        }

        if (typeid(T) == typeid(double))
        {
            dataDouble.resize(data.size());
            std::memcpy(dataDouble.data(), data.data(), data.size() * sizeof(double));
            argumentDataType = ArgumentDataType::Double;
        }
        else if (typeid(T) == typeid(float))
        {
            dataFloat.resize(data.size());
            std::memcpy(dataFloat.data(), data.data(), data.size() * sizeof(float));
            argumentDataType = ArgumentDataType::Float;
        }
        else if (typeid(T) == typeid(int))
        {
            dataInt.resize(data.size());
            std::memcpy(dataInt.data(), data.data(), data.size() * sizeof(int));
            argumentDataType = ArgumentDataType::Int;
        }
        else
        {
            throw std::runtime_error(std::string("Unsupported argument data type was provided for kernel argument"));
        }
    }
};

} // namespace ktt
