#define WORKGROUP_SIZE  (WARP_COUNT * WARP_SIZE)

__kernel __attribute__((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
void sum(
    const uint N,
    __global float *arr,
    __global float *res)
{
//     if (get_global_id(0) == 0)
//     {
//         for (uint i = 0; i < N && i < get_num_groups(0); ++i)
//         {
//             res[i] = arr[i];
//         }
//     }

    __local float local_sums[WORKGROUP_SIZE];

    // Initialize local sum.
    //
    local_sums[get_local_id(0)] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = get_global_id(0); i < N; i += get_global_size(0)) {
        // Fetch element.
        //
        float f = arr[i];

        // Add to thread-local sum.
        //
        local_sums[get_local_id(0)] += f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
        if (get_local_id(0) < stride) {
            local_sums[get_local_id(0)] += local_sums[get_local_id(0) + stride];
            local_sums[get_local_id(0) + stride] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (get_local_id(0) == 0) {
        // res[get_group_id(0)] = 0;
        // for (uint i = 1; i < get_local_size(0); ++i) {
        //     local_sums[0] += local_sums[i];
        // }
        res[get_group_id(0)] = local_sums[0];
    }

//     if (get_global_id(0) < get_num_groups(0)) {
//         res[get_global_id(0)] = local_sums[get_local_id(0)];
//     }
}