#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "definitions.h"
#include "kernel.h"
#include <Windows.h>


 ////////////////////////////////////////////////////////////////////////////////////////////////////
 // GLOBAL VARIABLES
 ////////////////////////////////////////////////////////////////////////////////////////////////////

// Device images (GPU)
byte *dev_src_image;
byte *dev_dst_image;

// Information about the image to be processed
int width;
int height;
int array_size;

// Error management
cudaError_t error;

// Threshold variables (also Border y Reverse)
int num_threads_th;
int num_blocks_th;
int step_th;

// Erode & Dilate variables (also Border)
int num_block_x_ed;
int num_block_y_ed;
int num_threads_x_ed;
int num_threads_y_ed;
dim3 grid_dim_ed;
dim3 block_dim_ed;



////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU UTILITY
////////////////////////////////////////////////////////////////////////////////////////////////////

// Pointers swap
void swap_buffers(byte **a, byte **b)
{
	byte *aux = *a;
	*a = *b;
	*b = aux;
}



////////////////////////////////////////////////////////////////////////////////////////////////////
// THRESHOLD [GRAYSCALE TO BINARY]
////////////////////////////////////////////////////////////////////////////////////////////////////

// Threshold kernel
__global__ void threshold_kernel(byte *dev_src_image, byte *dev_dst_image, int min, int max, int array_size, int num_threads_th, int step_th)
{
	// Calculate the position of the thread in the grid
	int pos = blockIdx.x * num_threads_th + threadIdx.x;

	// Do the operation
	for (; pos < array_size; pos += step_th)
		dev_dst_image[pos] = (dev_src_image[pos] >= min && dev_src_image[pos] <= max) ? 1 : 0;
}

// Public call to Threshold kernel
void _threshold(int min, int max, int _num_blocks_th, int _num_threads_th)
{
	// If the user doesn't indicate blocks and threads, these are calculated automatically
	if (_num_threads_th == 0 && _num_blocks_th == 0)
		_auto_threshold_parameters();
	else
	{
		num_threads_th = _num_threads_th;
		num_blocks_th = _num_blocks_th;
		step_th = _num_blocks_th * _num_threads_th;
	}

	// Threshold kernel call
	threshold_kernel << <num_blocks_th, num_threads_th >> > (dev_src_image, dev_dst_image, min, max, array_size, num_threads_th, step_th);
	swap_buffers(&dev_src_image, &dev_dst_image);

	cudaDeviceSynchronize();
}



////////////////////////////////////////////////////////////////////////////////////////////////////
// ERODE [BINARY]
////////////////////////////////////////////////////////////////////////////////////////////////////

// Erode kernel
__global__ void erode_kernel(byte *dev_src_image, byte *dev_dst_image, int height, int width, int radio)
{
	// Calculate the position of the thread in the grid
	int posx = threadIdx.x + blockIdx.x * blockDim.x;
	int posy = threadIdx.y + blockIdx.y * blockDim.y;

	// Do the operation
	if (posx + (posy * width) <= (width * height))
	{
		// Calculate the mask limit
		unsigned int start_i = max(posy - radio, 0);
		unsigned int end_i = min(height - 1, posy + radio);
		unsigned int start_j = max(posx - radio, 0);
		unsigned int end_j = min(width - 1, posx + radio);

		int _min = 1;

		// Write the minimum value
		for (int i = start_i; i <= end_i; i++)
			for (int j = start_j; j <= end_j; j++)
				_min = min(_min, dev_src_image[i*width + j]);

		dev_dst_image[posy * width + posx] = _min;
	}
}

// Public call to Erode kernel
void _erode(int radio, int _num_block_x_ed, int _num_block_y_ed, int _num_threads_x_ed, int _num_threads_y_ed)
{
	// If the user doesn't indicate blocks and threads, these are calculated automatically
	_set_erode_dilate_parameters(_num_block_x_ed, _num_block_y_ed, _num_threads_x_ed, _num_threads_y_ed);

	// Erode kernel call
	erode_kernel << <grid_dim_ed, block_dim_ed >> > (dev_src_image, dev_dst_image, height, width, radio);
	swap_buffers(&dev_src_image, &dev_dst_image);

	cudaDeviceSynchronize();
}



////////////////////////////////////////////////////////////////////////////////////////////////////
// DILATE [BINARY]
////////////////////////////////////////////////////////////////////////////////////////////////////

// Dilate kernel
__global__ void dilate_kernel(byte *dev_src_image, byte *dev_dst_image, int height, int width, int radio)
{
	// Calculate the position of the thread in the grid
	int posx = threadIdx.x + blockIdx.x * blockDim.x;
	int posy = threadIdx.y + blockIdx.y * blockDim.y;

	// Do the operation
	if (posx + (posy * width) <= (width * height))
	{
		// Calculate the mask limit
		unsigned int start_i = max(posy - radio, 0);
		unsigned int end_i = min(height - 1, posy + radio);
		unsigned int start_j = max(posx - radio, 0);
		unsigned int end_j = min(width - 1, posx + radio);

		int _max = 0;

		// Write the maximum value
		for (int i = start_i; i <= end_i; i++)
			for (int j = start_j; j <= end_j; j++)
				_max = max(_max, dev_src_image[i*width + j]);

		dev_dst_image[posy * width + posx] = _max;
	}
}

// Public call to Dilate kernel
void _dilate(int radio, int _num_block_x_ed, int _num_block_y_ed, int _num_threads_x_ed, int _num_threads_y_ed)
{
	// If the user doesn't indicate blocks and threads, these are calculated automatically
	_set_erode_dilate_parameters(_num_block_x_ed, _num_block_y_ed, _num_threads_x_ed, _num_threads_y_ed);

	// Dilate kernel call
	dilate_kernel << <grid_dim_ed, block_dim_ed >> > (dev_src_image, dev_dst_image, height, width, radio);
	swap_buffers(&dev_src_image, &dev_dst_image);

	cudaDeviceSynchronize();
}



////////////////////////////////////////////////////////////////////////////////////////////////////
// BORDER [BINARY]
////////////////////////////////////////////////////////////////////////////////////////////////////

// Border kernel
__global__ void border_kernel(byte *dev_src_image, byte *dev_dst_image, int array_size, int num_threads_th, int step_th)
{
	// Calculate the position of the thread in the grid
	int pos = blockIdx.x * num_threads_th + threadIdx.x;

	// Do the operation
	for (; pos < array_size; pos += step_th)
		dev_dst_image[pos] = (dev_dst_image[pos] == dev_src_image[pos]) ? 0 : 1;
}

// Public call to Border kernel
void _border(int radio, int _num_block_x_db, int _num_block_y_db, int _num_threads_x_db, int _num_threads_y_db, int _num_blocks_ab, int _num_threads_ab)
{
	// If the user doesn't indicate blocks and threads, these are calculated automatically
	_set_erode_dilate_parameters(_num_block_x_db, _num_block_y_db, _num_threads_x_db, _num_threads_y_db);

	// Dilate kernel call
	dilate_kernel << <grid_dim_ed, block_dim_ed >> > (dev_src_image, dev_dst_image, height, width, radio);
	swap_buffers(&dev_src_image, &dev_dst_image);

	cudaDeviceSynchronize();

	// If the user doesn't indicate blocks and threads, these are calculated automatically
	if (_num_threads_ab == 0 && _num_blocks_ab == 0)
		_auto_threshold_parameters();
	else
	{
		num_threads_th = _num_threads_ab;
		num_blocks_th = _num_blocks_ab;
		step_th = _num_blocks_ab * _num_threads_ab;
	}

	// Border kernel call
	border_kernel << <num_blocks_th, num_threads_th >> > (dev_src_image, dev_dst_image, array_size, num_threads_th, step_th);
	swap_buffers(&dev_src_image, &dev_dst_image);

	cudaDeviceSynchronize();
}



////////////////////////////////////////////////////////////////////////////////////////////////////
// REVERSE [BINARY]
////////////////////////////////////////////////////////////////////////////////////////////////////

// Reverse kernel
__global__ void reverse_kernel(byte *dev_src_image, byte *dev_dst_image, int array_size, int num_threads_th, int step_th)
{
	// Calculate the position of the thread in the grid
	int pos = blockIdx.x * num_threads_th + threadIdx.x;

	// Do the operation
	for (; pos < array_size; pos += step_th)
		dev_dst_image[pos] = (dev_src_image[pos] == 0) ? 1 : 0;
}

// Public call to Reverse kernel
void _reverse(int _num_blocks_r, int _num_threads_r)
{
	// If the user doesn't indicate blocks and threads, these are calculated automatically
	if (_num_threads_r == 0 && _num_blocks_r == 0)
		_auto_threshold_parameters();
	else
	{
		num_threads_th = _num_threads_r;
		num_blocks_th = _num_blocks_r;
		step_th = _num_blocks_r * _num_threads_r;
	}

	// Reverse kernel call
	reverse_kernel << <num_blocks_th, num_threads_th >> > (dev_src_image, dev_dst_image, array_size, num_threads_th, step_th);
	swap_buffers(&dev_src_image, &dev_dst_image);

	cudaDeviceSynchronize();
}



////////////////////////////////////////////////////////////////////////////////////////////////////
// OPEN & CLOSE [BINARY]
////////////////////////////////////////////////////////////////////////////////////////////////////

// Public call to Open
void _open(int radio, int _num_block_x_ed, int _num_block_y_ed, int _num_threads_x_ed, int _num_threads_y_ed)
{
	// If the user doesn't indicate blocks and threads, these are calculated automatically
	_set_erode_dilate_parameters(_num_block_x_ed, _num_block_y_ed, _num_threads_x_ed, _num_threads_y_ed);

	// Erode kernel call
	erode_kernel << <grid_dim_ed, block_dim_ed >> > (dev_src_image, dev_dst_image, height, width, radio);
	swap_buffers(&dev_src_image, &dev_dst_image);

	cudaDeviceSynchronize();

	// Dilate kernel call
	dilate_kernel << <grid_dim_ed, block_dim_ed >> > (dev_src_image, dev_dst_image, height, width, radio);
	swap_buffers(&dev_src_image, &dev_dst_image);

	cudaDeviceSynchronize();
}

// Public call to Open
void _close(int radio, int _num_block_x_ed, int _num_block_y_ed, int _num_threads_x_ed, int _num_threads_y_ed)
{
	// If the user doesn't indicate blocks and threads, these are calculated automatically
	_set_erode_dilate_parameters(_num_block_x_ed, _num_block_y_ed, _num_threads_x_ed, _num_threads_y_ed);

	// Dilate kernel call
	dilate_kernel << <grid_dim_ed, block_dim_ed >> > (dev_src_image, dev_dst_image, height, width, radio);
	swap_buffers(&dev_src_image, &dev_dst_image);

	cudaDeviceSynchronize();

	// Erode kernel call
	erode_kernel << <grid_dim_ed, block_dim_ed >> > (dev_src_image, dev_dst_image, height, width, radio);
	swap_buffers(&dev_src_image, &dev_dst_image);

	cudaDeviceSynchronize();
}



////////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA MEMORY MANAGEMENT
////////////////////////////////////////////////////////////////////////////////////////////////////

// Reserve GPU memory and copy image from CPU to GPU
void _copy_img_to_gpu(byte *hst_src_image, int *error_code)
{
	// Reserve src image memory on GPU
	error = cudaMalloc(&dev_src_image, array_size);

	if (error != cudaSuccess)
	{
		*error_code = CODE_ERROR_200;
		exit(EXIT_FAILURE);
	}

	// Reserve dst image memory on GPU
	error = cudaMalloc(&dev_dst_image, array_size);

	if (error != cudaSuccess)
	{
		*error_code = CODE_ERROR_201;
		exit(EXIT_FAILURE);
	}

	// Copy src image memory from CPU to GPU
	error = cudaMemcpy(dev_src_image, hst_src_image, array_size, cudaMemcpyHostToDevice);

	if (error != cudaSuccess)
	{
		*error_code = CODE_ERROR_202;
		exit(EXIT_FAILURE);
	}

	cudaDeviceSynchronize();
}

// Copy image from GPU to CPU
void _copy_img_to_cpu(byte *hst_dst_image, int *error_code)
{
	cudaDeviceSynchronize();

	// Copy src image memory from GPU to CPU
	error = cudaMemcpy(hst_dst_image, dev_src_image, array_size, cudaMemcpyDeviceToHost);

	if (error != cudaSuccess)
	{
		*error_code = CODE_ERROR_203;
		exit(EXIT_FAILURE);
	}

	cudaDeviceSynchronize();
}

// Free cuda memory
void _cuda_free(int *error_code)
{
	// Free src image memory
	error = cudaFree(dev_src_image);

	if (error != cudaSuccess)
	{
		*error_code = CODE_ERROR_204;
		exit(EXIT_FAILURE);
	}

	//Free dst image memory
	error = cudaFree(dev_dst_image);

	if (error != cudaSuccess)
	{
		*error_code = CODE_ERROR_205;
		exit(EXIT_FAILURE);
	}
}



////////////////////////////////////////////////////////////////////////////////////////////////////
// PARAMETRIZACIÓN ALGORITMIA E IMÁGENES
////////////////////////////////////////////////////////////////////////////////////////////////////

// Set image to be processed parameters
void _set_img_info(int _width, int _height)
{
	width = _width;
	height = _height;
	array_size = _width * _height;
}

// Automatic block and thread assignment if the user does't indicate anything
void _set_erode_dilate_parameters(int _num_block_x_ed, int _num_block_y_ed, int _num_threads_x_ed, int _num_threads_y_ed)
{
	if (_num_block_x_ed == 0 && _num_block_y_ed == 0 && _num_threads_x_ed == 0 && _num_threads_y_ed == 0)
	{
		// Divide by multiple of 2. 16 is good option
		num_block_x_ed = width / 16;
		if (num_block_x_ed * 16 < width)
			num_block_x_ed++;

		num_block_y_ed = height / 16;
		if (num_block_y_ed * 16 < height)
			num_block_y_ed++;

		num_threads_x_ed = 16;
		num_threads_y_ed = 16;

		grid_dim_ed.x = num_block_x_ed;
		grid_dim_ed.y = num_block_y_ed;
		block_dim_ed.x = num_threads_x_ed;
		block_dim_ed.y = num_threads_y_ed;
	}
	else
	{
		num_block_x_ed = _num_block_x_ed;
		num_block_y_ed = _num_block_y_ed;
		num_threads_x_ed = _num_threads_x_ed;
		num_threads_y_ed = _num_threads_y_ed;

		grid_dim_ed.x = _num_block_x_ed;
		grid_dim_ed.y = _num_block_y_ed;
		block_dim_ed.x = _num_threads_x_ed;
		block_dim_ed.y = _num_threads_y_ed;
	}
}

// Automatic block and thread assignment if the user does't indicate anything
void _auto_threshold_parameters()
{
	num_threads_th = 1024; // It's the maximum threads per block
	num_blocks_th = array_size / num_threads_th;

	if (num_blocks_th * num_threads_th < array_size) // Add one more block if odd
		num_blocks_th++;

	step_th = num_blocks_th * num_threads_th;
}
