__kernel void matrixTranspose(__global float* in_mat, __global float* out_mat, const int mat_size)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    out_mat[y + x*mat_size] = in_mat[x + y*mat_size];
}
