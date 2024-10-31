#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int main() {
    // problem sizes has to agree with sizes defined in JSON
    const int cells = 256;
    const int a = 64;
    const float gs = 0.5;

    float *grid = (float*)calloc(cells*cells*cells, sizeof(float));
    float *atoms = (float*)malloc(a*4*sizeof(float));

    // generator data has to agree with Generators in JSON
    for (int i = 0; i < a; i++) {
        atoms[i*4 + 0] = (float)i*0.1f;
        atoms[i*4 + 1] = (float)i*0.15f;
        atoms[i*4 + 2] = (float)i*0.2f;
        atoms[i*4 + 3] = (float)(i%10)*0.1f;
    }

    for (int z = 0; z < cells; z++)
        for (int y = 0; y < cells; y++)
            for (int x = 0; x < cells; x++) 
                for (int at = 0; at < a; at++){
                    float dx = (float)x*gs - atoms[at*4+0];
                    float dy = (float)y*gs - atoms[at*4+1];
                    float dz = (float)z*gs - atoms[at*4+2];
                    float e = atoms[at*4+3] / sqrtf(dx*dx + dy*dy + dz*dz);
                    grid[z*cells*cells + y*cells + x] += e;
                }

    FILE *fout = fopen("Coulomb3d.bin", "wb");
    fwrite(grid, cells*cells*cells, sizeof(float), fout);
    fclose(fout);

    return 0;
}
