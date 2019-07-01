#pragma once

#include<string> 
typedef unsigned char byte;
using namespace std;

// General errors
#define CODE_OK 0			// All Right
#define CODE_ERROR -1		// Uncontrolled error

// Utility errors
#define CODE_ERROR_100 100	// The indicated device (GPU) doesn't exist
#define CODE_ERROR_101 101	// The indicated GPU exists, but there has been a problem assigning it
#define CODE_ERROR_102 102	// Error to get the number of available GPUs
#define CODE_ERROR_103 103	// The indicated GPU exists, but there has been a problem restarting it
#define CODE_ERROR_104 104	// Error assigning the indicated GPU to later restart it
#define CODE_ERROR_105 105	// There are no compatible CUDA devices to restart

// CUDA errors
#define CODE_ERROR_200 200	// Error reserving memory for the src image of the device (GPU)
#define CODE_ERROR_201 201	// Error reserving memory for the dst image of the device (GPU)
#define CODE_ERROR_202 202	// Error to copy the src image from the host (CPU) to the device (GPU)
#define CODE_ERROR_203 203	// Error to copy the src image from the device (GPU) to the host (CPU)
#define CODE_ERROR_204 204	// Error to free memory from the src image of the device (GPU)
#define CODE_ERROR_205 205	// Error to free memory from the dst image of the device (GPU)