#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <stddef.h>
#include <sys/time.h>
#include <sys/resource.h>

static double second(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

#define DP 0

#if DP==0
#define REAL float
#else
#define REAL double
#endif

int main(int argc, char** argv)
{
    // Declare and initialize data (m, n > 1)
    unsigned int a = 4;
    unsigned int b = 4;
    unsigned int c = 4;
    int batch = 1024*64;

    if (argc >= 5) {
        a = atoi(argv[1]);
        b = atoi(argv[2]);
        c = atoi(argv[3]);
        batch = atoi(argv[4]);
    }

    std::cout << "Computing C = AB using " << batch << " matrices of sizes"
        << std::endl
        << "A: " << a << " x " << b << std::endl
        << "B: " << c << " x " << a << std::endl
        << "C: " << c << " x " << b << std::endl;

    std::vector<REAL> srcA(a*b*batch, 0.0f);
    std::vector<REAL> srcB(c*a*batch, 0.0f);
    std::vector<REAL> dst(c*b*batch, 0.0f);
    std::vector<REAL> check(c*b*batch, 0.0f);

    // fill with random data
    for (size_t i = 0; i < a*b*batch; i++)
    {
        srcA[i] = 10.0f*((REAL)rand()) / ((REAL) RAND_MAX);
    }
    for (size_t i = 0; i < c*a*batch; i++)
    {
        srcB[i] = 10.0f*((REAL)rand()) / ((REAL) RAND_MAX);
    }

    // allocate GPU data
    REAL *devA, *devB, *devC;
    cudaMalloc((void **)&devA, a*b*batch * sizeof(REAL));
    cudaMalloc((void **)&devB, c*a*batch * sizeof(REAL));
    cudaMalloc((void **)&devC, c*b*batch * sizeof(REAL));

    // copy input
    cudaMemcpy(devA, &(srcA[0]), a*b*batch * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, &(srcB[0]), c*a*batch * sizeof(REAL), cudaMemcpyHostToDevice);
    //cudaMemcpy(devC, &(srcC[0]), c*b*batch * sizeof(REAL), cudaMemcpyHostToDevice);

    // compute GEMM
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "CUBLAS initialization failed!" << std::endl;
        return 1;
    }
    REAL alpha = 1.0;
    REAL beta = 0.0;
    double start, stop;

    start = second();
    #if DP == 0
        cublasSgemmStridedBatched(handle, 
            CUBLAS_OP_N, CUBLAS_OP_N, 
            b, c, a, 
            &alpha,
            devA, b, a*b,
            devB, a, c*a,
            &beta,
            devC, b, c*b,
            batch);
    #else
        cublasDgemmStridedBatched(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            b, c, a,
            &alpha,
            devA, b, a*b,
            devB, a, c*a,
            &beta,
            devC, b, c*b,
            batch);
    #endif
    cudaDeviceSynchronize();
    stop = second();

    std::cout << "Performance: " << 1e-9*(double)(a*b*c*2)*(double)batch / (stop-start) << " GFlops" << std::endl;
    std::cout << "Memory BW: " << 1e-9*(double)(a*b+c*a+c*b)*(double)(batch)*(double)sizeof(REAL) / (stop-start) << " GB/s" << std::endl;

    // download results
    cudaMemcpy(&(dst[0]), devC, c*b*batch * sizeof(REAL), cudaMemcpyDeviceToHost);

    // release CUDA memory
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    // check results    
    for (int i = 0; i < batch; i++) 
        for (int j = 0; j < c; j++)
            for (int k = 0; k < b; k++) {
                REAL tmp = 0.0;
                /*for (int l = 0; l < a; l++)
                    tmp += srcA[i*a*b + k*a + l] * srcB[i*c*a + l*c + j];
                check[i*c*b + k*c + j] = tmp;*/
                for (int l = 0; l < a; l++)
                    tmp += srcA[i*a*b + k + l*b] * srcB[i*c*a + l + j*a];
                check[i*c*b + k + j*b] = tmp;
            }
    for (int i = 0; i < batch; i++) 
        for (int j = 0; j < c; j++)
            for (int k = 0; k < b; k++)
                if (fabs(check[i*c*b + k*c + j] - dst[i*c*b + k*c + j]) > 0.001) {
                    std::cerr << "Data mismatch at " << i << " " << j << " " << k << ": " << check[i*c*b + k*c + j] << " != " << dst[i*c*b + k*c + j] << std::endl;
                    return -1;
                }

    return 0;
}
