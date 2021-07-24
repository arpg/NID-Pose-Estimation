#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

namespace g2o{

void CudaComputeH(bool calculate_der, double* im0, double*im1, double* points3d, int* bs_counter, double* bs_ref, int* bs_index_ref, double* pose, double* camera_intrincis, int bin_num, int bs_degree, int cell_num, int rows, int cols, double* Href, double* pro_target, double* pro_joint, double* Htarget, double* Hjoint, double* der);

}