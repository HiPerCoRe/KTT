#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(local_size_x = LOCAL_SIZE_X, local_size_y = LOCAL_SIZE_Y, local_size_z = LOCAL_SIZE_Z) in;

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
