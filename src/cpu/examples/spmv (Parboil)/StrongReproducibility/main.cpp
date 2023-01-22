/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include "parboil.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <random>

#include "file.h"
#include "convert_dataset.h"

#include <omp.h>
#include "LongAccumulator.h"

using namespace std;

constexpr uint32_t DEFAULT_SEED = 1549813198;
constexpr uint32_t DEFAULT_NUMBER_OF_RUNS = 50;

constexpr char* input_files[] = { "bcsstk32.mtx", "fidapm05.mtx", "jgl009.mtx" };

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

bool diff(int dim, float *h_Ax_vector_1, float *h_Ax_vector_2)
{
    for (int i = 0; i < dim; i++)
        if (h_Ax_vector_1[i] != h_Ax_vector_2[i])
            return true;
    return false;
}

void spmv_seq (bool reproducible, int dim, int *h_nzcnt, int *h_ptr, int *h_indices, float *h_data,
               float *h_x_vector, int *h_perm, float *h_Ax_vector)
{
    float sum = 0.0f;
    LongAccumulator sum_acc;

    // Consider creating a random map by creating an array 0..dim - 1 and randomly shuffling it
    // for each execution. This should provide required randomness given the order of operations
    // is sequential at the moment.
    //
    for (int i = 0; i < dim; i++) {
        if (reproducible) {
            sum_acc = 0.0f;
        } else {
            sum = 0.0f;
        }

        int bound = h_nzcnt[i];

        for (int k = 0; k < bound; k++) {
            int j = h_ptr[k] + i;
            int in = h_indices[j];

            float d = h_data[j];
            float t = h_x_vector[in];

            if (reproducible) {
                sum_acc += d * t;
            } else {
                sum += d * t;
            }
        }

        if (reproducible) {
            h_Ax_vector[h_perm[i]] = sum_acc();
        } else {
            h_Ax_vector[h_perm[i]] = sum;
        }
    }
}

void spmv_omp (bool reproducible, int dim, int *h_nzcnt, int *h_ptr, int *h_indices, float *h_data,
               float *h_x_vector, int *h_perm, float *h_Ax_vector)
{
#pragma omp parallel
{
    float sum = 0.0f;
    LongAccumulator sum_acc;

    // Consider creating a random map by creating an array 0..dim - 1 and randomly shuffling it
    // for each execution. This should provide required randomness given the order of operations
    // is sequential at the moment.
    //
#pragma omp for
    for (int i = 0; i < dim; i++) {
        if (reproducible) {
            sum_acc = 0.0f;
        } else {
            sum = 0.0f;
        }

        int bound = h_nzcnt[i];

        for (int k = 0; k < bound; k++) {
            int j = h_ptr[k] + i;
            int in = h_indices[j];

            float d = h_data[j];
            float t = h_x_vector[in];

            if (reproducible) {
                sum_acc += d * t;
            } else {
                sum += d * t;
            }
        }

        if (reproducible) {
            h_Ax_vector[h_perm[i]] = sum_acc();
        } else {
            h_Ax_vector[h_perm[i]] = sum;
        }
    }
}
}

void execute (uint32_t nruns, bool parallel, bool reproducible, int dim, int *h_nzcnt, int *h_ptr, int *h_indices, float *h_data,
              float *h_x_vector, int *h_perm, float *h_Ax_vector, double &time)
{
    time = 0.0f;

    float *tmp_h_Ax_vector = new float[dim];

    for (int i = 0; i < nruns; ++i) {
        if (i == 0)
            time = omp_get_wtime();
        if (parallel)
            spmv_omp(reproducible, dim, h_nzcnt, h_ptr, h_indices, h_data, h_x_vector, h_perm, tmp_h_Ax_vector);
        else
            spmv_seq(reproducible, dim, h_nzcnt, h_ptr, h_indices, h_data, h_x_vector, h_perm, tmp_h_Ax_vector);
        if (i == 0) {
            time = omp_get_wtime() - time;
            cout << fixed << setprecision(10) << (float) time / 1000.0 << '\t';
            memcpy (h_Ax_vector, tmp_h_Ax_vector, dim * sizeof (float));
        } else if (diff(dim, h_Ax_vector, tmp_h_Ax_vector)) {
            printf("%s (%sreproducible) implementation not reproducible after %d runs!\n",
                    parallel ? "Parallel" : "Sequential", reproducible ? "" : "non-", i);
            break;
        }
    }

    delete[] tmp_h_Ax_vector;
}

int main (int argc, char** argv)
{
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
    float *h_Ax_vector_seq, *h_Ax_vector_seq_rep, *h_Ax_vector_omp, *h_Ax_vector_omp_rep;
    float *h_x_vector;

    double time_seq, time_omp, time_seq_rep, time_omp_rep;

    const int exe_path_len = strrchr(argv[0], '/') - argv[0] + 1;
    char exe_path[256];
    strncpy(exe_path, argv[0], exe_path_len);
    exe_path[exe_path_len] = '\0';

    char input_file_path[256];

    for (int i = 0; i < 3; ++i)
    {
        strncpy(input_file_path, exe_path, exe_path_len + 1);
        strcat(input_file_path, "data/");
        strcat(input_file_path, input_files[i]);

        cout << input_files[i] << "\n\n";

        int col_count;
        coo_to_jds(
            input_file_path,
            1, // row padding
            pad, // warp size
            1, // pack size
            0, // debug level [0:2]
            &h_data, &h_ptr, &h_nzcnt, &h_indices, &h_perm,
            &col_count, &dim, &len, &nzcnt_len, &depth
        );

        h_x_vector = new float[dim];

        if (!generate_vector(h_x_vector, dim, DEFAULT_SEED)) {
            fprintf(stderr, "Failed to generate dense vector.\n");
            exit(-1);
        }

        h_Ax_vector_seq = new float[dim];
        h_Ax_vector_seq_rep = new float[dim];
        h_Ax_vector_omp = new float[dim];
        h_Ax_vector_omp_rep = new float[dim];


        cout << "\nseq\t";
        
        for (int run = 0; run < 3; ++run) execute (1, false, false, dim, h_nzcnt, h_ptr, h_indices, h_data, h_x_vector, h_perm, h_Ax_vector_seq, time_seq);
        cout << '\n';
        
        for (int thread_count = 1; thread_count <= 128; thread_count <<= 1)
        {
            omp_set_dynamic(0);                 // Explicitly disable dynamic teams
            omp_set_num_threads(thread_count);  // Use  thread_count threads for all consecutive parallel regions

            #pragma omp parallel
            #pragma omp single
            {
                cout << omp_get_num_threads() << '\t';
            }

            for (int run = 0; run < 3; ++run) execute (1, true, false, dim, h_nzcnt, h_ptr, h_indices, h_data, h_x_vector, h_perm, h_Ax_vector_omp, time_omp);
            cout << '\n';
        }
        
        cout << "\nreproducible\n\nseq\t";

        for (int run = 0; run < 3; ++run) execute (1, false, true, dim, h_nzcnt, h_ptr, h_indices, h_data, h_x_vector, h_perm, h_Ax_vector_seq_rep, time_seq_rep);
        cout << '\n';
        
        for (int thread_count = 1; thread_count <= 128; thread_count <<= 1)
        {
            omp_set_dynamic(0);                 // Explicitly disable dynamic teams
            omp_set_num_threads(thread_count);  // Use  thread_count threads for all consecutive parallel regions

            #pragma omp parallel
            #pragma omp single
            {
                cout << omp_get_num_threads() << '\t';
            }

            for (int run = 0; run < 3; ++run) execute (1, true, true, dim, h_nzcnt, h_ptr, h_indices, h_data, h_x_vector, h_perm, h_Ax_vector_omp_rep, time_omp_rep);
            cout << '\n';
        }

        cout << '\n';

        delete[] h_data;
        delete[] h_indices;
        delete[] h_ptr;
        delete[] h_perm;
        delete[] h_nzcnt;
        delete[] h_Ax_vector_seq;
        delete[] h_Ax_vector_seq_rep;
        delete[] h_Ax_vector_omp;
        delete[] h_Ax_vector_omp_rep;
        delete[] h_x_vector;
    }

    return 0;
}