#include <fstream>
#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "tools.h"


// Write in a file the information about GPUs compatible with CUDA
void _get_gpu_info(char *url)
{
	string s_url = url;
	ofstream data;

	data.open(s_url + "gpu_info.txt");

	int device_count = 0;
	cudaError_t error = cudaGetDeviceCount(&device_count);

	if (error != cudaSuccess)
	{
		data << "cudaGetDeviceCount devolvió: " << error << "\n-> " << cudaGetErrorString(error) << "\n";
		data << "Resultado = FALLIDO\n";
		return;
	}

	if (device_count == 0)
		data << "No hay dispositivos disponibles que soporten CUDA\n";
	else
		data << "Detectado(s) " << device_count << " dispositivo(s) compatible(s) con CUDA\n";

	int dev, driver_version = 0, runtime_version = 0;

	for (dev = 0; dev < device_count; ++dev)
	{
		cudaSetDevice(dev);
		cudaDeviceProp device_prop;
		cudaGetDeviceProperties(&device_prop, dev);

		data << "\nDispositivo " << dev << ": \"" << device_prop.name << "\"\n";

		cudaDriverGetVersion(&driver_version);
		cudaRuntimeGetVersion(&runtime_version);
		data << "   - CUDA Driver Version / Runtime Version " << driver_version / 1000 << "." << (driver_version % 100) / 10 << " / " << runtime_version / 1000 << "." << (runtime_version % 100) / 10 << "\n";
		data << "   - CUDA Capability Major/Minor version number: " << device_prop.major << "." << device_prop.minor << "\n";
		data << "   - Total amount of global memory: " << device_prop.totalGlobalMem / 1048576.0f << " MBytes (" << device_prop.totalGlobalMem << " bytes)\n";
		data << "   - (" << device_prop.multiProcessorCount << ") Multiprocessors, (" << _ConvertSMVer2Cores(device_prop.major, device_prop.minor) << ") CUDA Cores/MP: " << _ConvertSMVer2Cores(device_prop.major, device_prop.minor) * device_prop.multiProcessorCount << " CUDA Cores\n";
		data << "   - Total amount of constant memory: " << device_prop.totalConstMem << " bytes\n";
		data << "   - Total amount of shared memory per block: " << device_prop.sharedMemPerBlock << " bytes\n";
		data << "   - Total number of registers available per block: " << device_prop.regsPerBlock << "\n";
		data << "   - Warp size: " << device_prop.warpSize << "\n";
		data << "   - Maximum number of threads per multiprocessor: " << device_prop.maxThreadsPerMultiProcessor << "\n";
		data << "   - Maximum number of threads per block: " << device_prop.maxThreadsPerBlock << "\n";
		data << "   - Max dimension size of a thread block (x,y,z): (" << device_prop.maxThreadsDim[0] << ", " << device_prop.maxThreadsDim[1] << ", " << device_prop.maxThreadsDim[2] << ")\n";
		data << "   - Max dimension size of a grid size (x,y,z): (" << device_prop.maxGridSize[0] << ", " << device_prop.maxGridSize[1] << ", " << device_prop.maxGridSize[2] << ")\n";
		data << "   - memory pitch: " << device_prop.memPitch << " bytes\n";
	}

	data << "\nResultado = EXITOSO\n";

	data.close();
}

// Write information about custom error codes in a file
void _get_error_codes(char *url)
{
	string s_url = url;
	ofstream data;

	//data.open("error_codes.txt");
	data.open(s_url + "error_codes.txt");

	data << "CÓDIGOS ERRORES GENERALES\n";
	data << "======================================\n";
	data << "0      Todo correcto\n";
	data << "-1     Error no controlado\n";

	data << "\nCÓDIGOS ERRORES UTILIDADES\n";
	data << "======================================\n";
	data << "100     No existe el dispositivo (GPU) indicado\n";
	data << "101     La GPU indicada existe, pero ha habido un problema al asignarla\n";
	data << "102     Error al obtener el número de GPUs disponibles\n";
	data << "103     La GPU indicada existe, pero ha habido un problema al reiniciarla\n";
	data << "104     Error al asignar la GPU indicada para posteriormente reiniciarla\n";
	data << "105     No hay dispositivos compatibles con CUDA para reiniciar\n";

	data << "\nCÓDIGOS ERRORES CUDA\n";
	data << "======================================\n";
	data << "200     Error al reservar memoria para la imagen src del device\n";
	data << "201     Error al reservar memoria para la imagen dst del device\n";
	data << "202     Error al copiar la imagen src del host al device\n";
	data << "203     Error al copiar la imagen src del device al dst del host\n";
	data << "204     Error al liberar memoria de la imagen src del device\n";
	data << "205     Error al liberar memoria de la imagen dst del device\n";

	data.close();
}

// Returns the number of GPU devices supported by CUDA
void _get_cuda_available_devices(int *num_cuda_devices, int *error_code)
{
	cudaError_t error = cudaGetDeviceCount(num_cuda_devices);

	if (error != cudaSuccess)
	{
		*error_code = CODE_ERROR_102;
		exit(EXIT_FAILURE);
	}
	else
		*error_code = CODE_OK;
}

// Assign the indicated GPU device to use in the CUDA program
void _set_device(int device_id, int *error_code)
{
	int num_devices;

	_get_cuda_available_devices(&num_devices, error_code);

	if (device_id < 0 || device_id >= num_devices)
		*error_code = CODE_ERROR_100;
	else
	{
		cudaError_t error = cudaSetDevice(device_id);

		if (error != cudaSuccess)
		{
			*error_code = CODE_ERROR_101;
			exit(EXIT_FAILURE);
		}
		else
			*error_code = CODE_OK;
	}
}

// Restart the indicated GPU device, first assign it and then restart it
void _reset_device(int device_id, int* error_code)
{
	int num_devices;

	_get_cuda_available_devices(&num_devices, error_code);

	if (device_id < 0 || device_id >= num_devices)
		*error_code = CODE_ERROR_100;
	else
	{
		cudaError_t error = cudaSetDevice(device_id);

		// Check that the device assignment is correct
		if (error != cudaSuccess)
		{
			*error_code = CODE_ERROR_104;
			exit(EXIT_FAILURE);
		}
		else
		{
			error = cudaDeviceReset();

			// Check that the device restart is correct
			if (error != cudaSuccess)
			{
				*error_code = CODE_ERROR_103;
				exit(EXIT_FAILURE);
			}
			else
				*error_code = CODE_OK;
		}
	}
}

// Restart all available GPU devices and assign the first
void _reset_all_devices(int* error_code)
{
	int num_devices;

	_get_cuda_available_devices(&num_devices, error_code);

	if (num_devices < 1)
		*error_code = CODE_ERROR_105;
	else
	{
		for (int dev = 0; dev < num_devices; dev++)
			_reset_device(dev, error_code);
		_set_device(0, error_code);

		*error_code = CODE_OK;
	}
}