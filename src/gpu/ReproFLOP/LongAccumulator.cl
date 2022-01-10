#define WORKGROUP_SIZE (WARP_COUNT * WARP_SIZE)

__kernel __attribute__((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
void LongAccumulator ()
{
}

__kernel __attribute__((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
void LongAccumulatorComplete ()
{
}

__kernel __attribute__((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
void LongAccumulatorRound ()
{
}