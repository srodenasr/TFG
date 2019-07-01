#include "kernel.h"
#include "tools.h"
#include "main.h"


////////////////////////////////////////////////////////////////////////////////////////////////////
// UTILITIES
////////////////////////////////////////////////////////////////////////////////////////////////////

void get_gpu_info(char *url)
{
	_get_gpu_info(url);
}

void get_error_codes(char *url)
{
	_get_error_codes(url);
}

void get_cuda_available_devices(int *num_cuda_devices, int *error_code)
{
	_get_cuda_available_devices(num_cuda_devices, error_code);
}

void set_device(int device_id, int *error_code)
{
	_set_device(device_id, error_code);
}

void reset_device(int device_id, int* error_code)
{
	_reset_device(device_id, error_code);
}

void reset_all_devices(int* error_code)
{
	_reset_all_devices(error_code);
}



////////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA MEMORY MANAGEMENT
////////////////////////////////////////////////////////////////////////////////////////////////////

void set_img_info(int width, int height)
{
	_set_img_info(width, height);
}

void copy_img_to_gpu(byte *hst_src_image, int *error_code)
{
	_copy_img_to_gpu(hst_src_image, error_code);
}

void copy_img_to_cpu(byte *hst_dst_image, int *error_code)
{
	_copy_img_to_cpu(hst_dst_image, error_code);
}

void cuda_free(int *error_code)
{
	_cuda_free(error_code);
}



////////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA ALGORITHMS
////////////////////////////////////////////////////////////////////////////////////////////////////

void threshold(int min, int max, int _num_blocks_th, int _num_threads_th)
{
	_threshold(min, max, _num_blocks_th, _num_threads_th);
}

void border(int radio, int _num_block_x_db, int _num_block_y_db, int _num_threads_x_db, int _num_threads_y_db, int _num_blocks_ab, int _num_threads_ab)
{
	_border(radio, _num_block_x_db, _num_block_y_db, _num_threads_x_db, _num_threads_y_db, _num_blocks_ab, _num_threads_ab);
}

void reverse(int _num_blocks_r, int _num_threads_r)
{
	_reverse(_num_blocks_r, _num_threads_r);
}

void erode(int radio, int _num_block_x_ed, int _num_block_y_ed, int _num_threads_x_ed, int _num_threads_y_ed)
{
	_erode(radio, _num_block_x_ed, _num_block_y_ed, _num_threads_x_ed, _num_threads_y_ed);
}

void dilate(int radio, int _num_block_x_ed, int _num_block_y_ed, int _num_threads_x_ed, int _num_threads_y_ed)
{
	_dilate(radio, _num_block_x_ed, _num_block_y_ed, _num_threads_x_ed, _num_threads_y_ed);
}

void open(int radio, int _num_block_x_ed, int _num_block_y_ed, int _num_threads_x_ed, int _num_threads_y_ed)
{
	_open(radio, _num_block_x_ed, _num_block_y_ed, _num_threads_x_ed, _num_threads_y_ed);
}

void close(int radio, int _num_block_x_ed, int _num_block_y_ed, int _num_threads_x_ed, int _num_threads_y_ed)
{
	_close(radio, _num_block_x_ed, _num_block_y_ed, _num_threads_x_ed, _num_threads_y_ed);
}