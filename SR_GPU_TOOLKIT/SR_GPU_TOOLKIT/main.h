#pragma once

#include "definitions.h"

extern "C"
{
	// Utilidades
	void __declspec(dllexport) get_gpu_info(char *url);
	void __declspec(dllexport) get_error_codes(char *url);
	void __declspec(dllexport) get_cuda_available_devices(int * num_cuda_devices, int * error_code);
	void __declspec(dllexport) set_device(int device_id, int * error_code);
	void __declspec(dllexport) reset_device(int device_id, int * error_code);
	void __declspec(dllexport) reset_all_devices(int * error_code);

	// Gestión CUDA
	void __declspec(dllexport) set_img_info(int width, int height);
	void __declspec(dllexport) copy_img_to_gpu(byte * hst_src_image, int * error_code);
	void __declspec(dllexport) copy_img_to_cpu(byte * hst_dst_image, int * error_code);
	void __declspec(dllexport) cuda_free(int * error_code);

	// Algoritmia CUDA
	void __declspec(dllexport) threshold(int min, int max, int _num_blocks_th, int _num_threads_th);
	void __declspec(dllexport) border(int radio, int _num_block_x_db, int _num_block_y_db, int _num_threads_x_db, int _num_threads_y_db, int _num_blocks_ab, int _num_threads_ab);
	void __declspec(dllexport) reverse(int _num_blocks_r, int _num_threads_r);
	void __declspec(dllexport) erode(int radio, int _num_block_x_ed, int _num_block_y_ed, int _num_threads_x_ed, int _num_threads_y_ed);
	void __declspec(dllexport) dilate(int radio, int _num_block_x_ed, int _num_block_y_ed, int _num_threads_x_ed, int _num_threads_y_ed);
	void __declspec(dllexport) open(int radio, int _num_block_x_ed, int _num_block_y_ed, int _num_threads_x_ed, int _num_threads_y_ed);
	void __declspec(dllexport) close(int radio, int _num_block_x_ed, int _num_block_y_ed, int _num_threads_x_ed, int _num_threads_y_ed);
}


