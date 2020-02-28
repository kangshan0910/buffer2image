#define TILE_N 8
#define TILE_M 8
__attribute__((reqd_work_group_size(8, 1, 1)))
__kernel void blockread_test(
    __read_only image2d_t src)
{
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
	
	int2    coord = (int2)( group_x * TILE_N  * sizeof(uint), group_y * TILE_M );
	float8  block = as_float8( intel_sub_group_block_read8( src, coord ) );
	
	printf("group_xy(%d,%d) local_xy(%02d,%02d) data=%.2v8hlf\n", group_x, group_y, local_x, local_y, block);

}
