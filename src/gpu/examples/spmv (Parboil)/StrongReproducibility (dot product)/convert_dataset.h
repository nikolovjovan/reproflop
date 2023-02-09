#ifndef _CONVERT_DATASET_H
#define _CONVERT_DATASET_H

typedef struct _mat_entry {
    int row, col; /* i,j */
    float val;
} mat_entry;

void read_coo_from_mtx_file (
    char* mtx_filename,
    int* dim,
    int* rows,
    int* cols,
    int* nz,
    mat_entry** entries);

int coo_to_jds (
    char* mtx_filename,
    int pad_rows,
    int warp_size,
    int pack_size,
    int debug_level,
    float** data,
    int** data_row_ptr,
    int** nz_count,
    int** data_col_index,
    int** data_row_map,
    int* data_cols,
    int* dim,
    int* len,
    int* nz_count_len,
    int* data_ptr_len);

int coo_to_jds (
    int rows,
    int cols,
    int nz,
    mat_entry* entries,
    int pad_rows,
    int warp_size,
    int pack_size,
    int debug_level,
    float** data,
    int** data_row_ptr,
    int** nz_count,
    int** data_col_index,
    int** data_row_map,
    int* data_cols,
    int* len,
    int* nz_count_len,
    int* data_ptr_len);

int coo_to_csr (
    char* mtx_filename,
    int debug_level,
    int* dim,
    float** data,
    int** data_row_ptr,
    int** data_col_index);

int coo_to_csr (
    int rows,
    int cols,
    int nz,
    mat_entry* entries,
    int debug_level,
    float** data,
    int** data_row_ptr,
    int** data_col_index);

#endif