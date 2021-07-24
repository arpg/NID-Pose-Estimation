#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


void Calculate3Dpoint(double* depth, double* pose_c2w, double* points_3d, double* camera_intrincis, int rows, int cols);
