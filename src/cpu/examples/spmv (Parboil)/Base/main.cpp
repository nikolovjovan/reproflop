/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include "parboil.h"

#include <cstdio>
#include <random>

#include "file.h"
#include "convert_dataset.h"

using namespace std;

constexpr uint32_t DEFAULT_SEED = 1549813198;

bool generate_vector (float *x_vector, int dim, uint32_t seed)
{
    if (nullptr == x_vector) {
        return false;
    }

    mt19937 gen(seed);
    uniform_real_distribution<float> float_dist(0.f, 1.f);

    for (int i = 0; i < dim; i++) {
        x_vector[i] = float_dist(gen);
    }

    return true;
}

int main (int argc, char** argv)
{
    struct pb_TimerSet timers;
    struct pb_Parameters *parameters;
    
    printf ("CPU-based sparse matrix vector multiplication****\n");
    printf ("Original version by Li-Wen Chang <lchang20@illinois.edu> and Shengzhao Wu<wu14@illinois.edu>\n");
    printf ("This version maintained by Chris Rodrigues  ***********\n");

    parameters = pb_ReadParameters (&argc, argv);
    if (parameters->inpFiles[0] == NULL) {
        fprintf (stderr, "Expecting one input filename.\n");
        exit (-1);
    }

    pb_InitializeTimerSet (&timers);
    pb_SwitchToTimer (&timers, pb_TimerID_COMPUTE);

    // Parameters declaration
    //
    int len;
    int depth;
    int dim;
    int pad=1;
    int nzcnt_len;

    // Host memory allocation
    // Matrix
    //
    float *h_data;
    int *h_indices;
    int *h_ptr;
    int *h_perm;
    int *h_nzcnt;

    // Vector
    //
    float *h_Ax_vector;
    float *h_x_vector;

    // Load matrix from files
    //
    pb_SwitchToTimer (&timers, pb_TimerID_IO);

    int col_count;
    coo_to_jds(
        parameters->inpFiles[0], // bcsstk32.mtx, fidapm05.mtx, jgl009.mtx
        1, // row padding
        pad, // warp size
        1, // pack size
        1, // debug level [0:2]
        &h_data, &h_ptr, &h_nzcnt, &h_indices, &h_perm,
        &col_count, &dim, &len, &nzcnt_len, &depth
    );

    h_Ax_vector = new float[dim];
    h_x_vector = new float[dim];

    uint32_t seed = parameters->seed == 0 ? DEFAULT_SEED : parameters->seed;

    if (!generate_vector(h_x_vector, dim, seed)) {
        fprintf(stderr, "Failed to generate dense vector.\n");
        exit(-1);
    }

    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

    // Main execution
    //
    for (int p = 0; p < 50; p++) {
    #pragma omp parallel for
        for (int i = 0; i < dim; i++) {
            int k;
            float sum = 0.0f;
            int bound = h_nzcnt[i];
            
            for (k = 0; k < bound; k++) {
                int j = h_ptr[k] + i;
                int in = h_indices[j];

                float d = h_data[j];
                float t = h_x_vector[in];

                sum += d * t;
            }
            //  #pragma omp critical
            h_Ax_vector[h_perm[i]] = sum;
        }
    }

    if (parameters->outFile) {
        pb_SwitchToTimer(&timers, pb_TimerID_IO);
        outputData(parameters->outFile, h_Ax_vector, dim);
    }
    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

    delete[] h_data;
    delete[] h_indices;
    delete[] h_ptr;
    delete[] h_perm;
    delete[] h_nzcnt;
    delete[] h_Ax_vector;
    delete[] h_x_vector;

    pb_SwitchToTimer(&timers, pb_TimerID_NONE);
    pb_PrintTimerSet(&timers);

    pb_FreeParameters(parameters);

    return 0;
}