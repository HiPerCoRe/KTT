#version 450
#extension GL_ARB_separate_shader_objects : enable

#define WORK_GROUP_SIZE 256

layout(local_size_x = WORK_GROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly buffer inputA
{
    float a[];
};

layout(std430, binding = 1) readonly buffer inputB
{
    float b[];
};

layout(std430, binding = 2) writeonly buffer outputResult
{
    float result[];
};

void main()
{
    uint index = gl_GlobalInvocationID.x;
    result[index] = a[index] + b[index];
}
