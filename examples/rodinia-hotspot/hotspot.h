#ifndef HOTSPOT_H
#define HOTSPOT_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#ifdef RD_WG_SIZE_0_0
  #define BLOCK_SIZE_REF RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
  #define BLOCK_SIZE_REF RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
  #define BLOCK_SIZE_REF RD_WG_SIZE
#else
#define BLOCK_SIZE_REF 8                                                             
#endif

#define PYRAMID_HEIGHT_REF 2
#define STR_SIZE 256
# define EXPAND_RATE 2// add one iteration will extend the pyramid base by 2 per each borderline

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5


#define MIN(a, b) ((a)<=(b) ? (a) : (b))




/* chip parameters	*/
const static float t_chip = 0.0005f;
const static float chip_height = 0.016f;
const static float chip_width = 0.016f;
/* ambient temperature, assuming no package at all	*/
const static float amb_temp = 80.0f;

/*// OpenCL globals
cl_context context;
cl_command_queue command_queue;
cl_device_id device;
cl_kernel kernel;

void writeoutput(float *, int, int, char *);
void readinput(float *, int, int, char *);
int compute_tran_temp(cl_mem, cl_mem[2], int, int, int, int, int, int, int, int, float *, float *);
void usage(int, char **);
void run(int, char **);
*/

void writeoutput(std::vector<float>&, int, int, char *);
void readinput(std::vector<float>&, int, int, char *);
void writeoutput(float*, int, int, char *);
void readinput(float*, int, int, char *);
void usage(int, char **);
#endif
