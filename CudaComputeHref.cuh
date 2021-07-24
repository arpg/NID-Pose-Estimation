#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void CudaComputeHref(double* im0, double* points3d, double* pose, double* camera_intrincis, int bin_num, int bs_degree, int cell_num, int rows, int cols, double* bs_value, int* bs_index, int* bs_counter, double* Href);

