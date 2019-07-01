#pragma once

#include "definitions.h"

// Gestión CUDA
void _copy_img_to_gpu(byte * hst_src_image, int *error_code);
void _copy_img_to_cpu(byte * hst_dst_image, int *error_code);
void _cuda_free(int *error_code);

// Algoritmos CUDA
void _threshold(int min, int max, int _num_blocks_th, int _num_threads_th);
void _border(int radio, int _num_block_x_db, int _num_block_y_db, int _num_threads_x_db, int _num_threads_y_db, int _num_blocks_ab, int _num_threads_ab);
void _reverse(int _num_blocks_r, int _num_threads_r);
void _erode(int radio, int _num_block_x_ed, int _num_block_y_ed, int _num_threads_x_ed, int _num_threads_y_ed);
void _dilate(int radio, int _num_block_x_ed, int _num_block_y_ed, int _num_threads_x_ed, int _num_threads_y_ed);
void _open(int radio, int _num_block_x_ed, int _num_block_y_ed, int _num_threads_x_ed, int _num_threads_y_ed);
void _close(int radio, int _num_block_x_ed, int _num_block_y_ed, int _num_threads_x_ed, int _num_threads_y_ed);

// Parametrización automática algoritmos
void _set_img_info(int _width, int _height);
void _set_erode_dilate_parameters(int _num_block_x_ed, int _num_block_y_ed, int _num_threads_x_ed, int _num_threads_y_ed);
void _auto_threshold_parameters();

