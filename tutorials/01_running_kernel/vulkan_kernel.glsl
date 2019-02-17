#version 450
#extension GL_ARB_separate_shader_objects : enable

#define WIDTH 3200
#define HEIGHT 2400
#define WORKGROUP_SIZE 32

layout(local_size_x = WORKGROUP_SIZE, local_size_y = WORKGROUP_SIZE, local_size_z = 1) in;
layout(std140, binding = 0) buffer buf
{
	vec4 imageData[];
};

void main()
{
	if (gl_GlobalInvocationID.x >= WIDTH || gl_GlobalInvocationID.y >= HEIGHT)
	{
		return;
	}
	
	float x = float(gl_GlobalInvocationID.x) / float(WIDTH);
	float y = float(gl_GlobalInvocationID.y) / float(HEIGHT);
	vec2 uv = vec2(x, y);
	float n = 0.0f;
	vec2 c = vec2(-0.445f, 0.0f) + (uv - 0.5f) * (2.0f + 1.7f * 0.2f);
	vec2 z = vec2(0.0f);
	const int M = 128;

	for (int i = 0; i < M; ++i)
	{
		z = vec2(z.x * z.x - z.y * z.y, 2.0f * z.x * z.y) + c;
		if (dot(z, z) > 2.0f) break;
		++n;
	}

	float t = float(n) / float(M);
	vec3 d = vec3(0.3f, 0.3f, 0.5f);
	vec3 e = vec3(-0.2f, -0.3f, -0.5f);
	vec3 f = vec3(2.1f, 2.0f, 3.0f);
	vec3 g = vec3(0.0f, 0.1f, 0.0f);
	vec4 color = vec4(d + e * cos(6.28318f * (f * t + g)), 1.0f);
	imageData[WIDTH * gl_GlobalInvocationID.y + gl_GlobalInvocationID.x] = color;
}
