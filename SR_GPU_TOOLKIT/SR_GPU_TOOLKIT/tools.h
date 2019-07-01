#pragma once

#include "definitions.h"

// Utilidades
void _get_gpu_info(char *url);
void _get_error_codes(char *url);
void _get_cuda_available_devices(int *num_cuda_devices, int *error_code);
void _set_device(int device_id, int *error_code);
void _reset_device(int device_id, int* error_code);
void _reset_all_devices(int* error_code);