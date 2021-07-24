#include "CudaPoints3d.cuh"
#include <sys/time.h>
#include <iostream>

__global__ void Calculate3DpointKernel(double* depth, double* pose, double* points_3d, double* d_in, int thread_work, int rows, int cols){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    int id = thread_work * (idy * blockDim.x * gridDim.x + idx);
    for(int i = 0; i<thread_work; i++){

        if(depth[id] <0.01 || depth[id] > 100){
            points_3d[3*id] = NAN;
            points_3d[3*id + 1] = NAN;
            points_3d[3*id + 2] = NAN;
            id++;
            continue;
        }

        int row = id / cols;
        int col = id % cols;
        double x0 = depth[id] * (col - d_in[2]) / d_in[0];
        double y0 = depth[id] * (row - d_in[3]) / d_in[1];
        double z0 = depth[id];

        points_3d[3*id]     = pose[0] * x0 + pose[4] * y0 + pose[8] * z0 + pose[12];
        points_3d[3*id + 1] = pose[1] * x0 + pose[5] * y0 + pose[9] * z0 + pose[13];
        points_3d[3*id + 2] = pose[2] * x0 + pose[6] * y0 + pose[10]* z0 + pose[14];

        id++;
    }
}

//points_3d is from the unifide memory, no need to copy
void Calculate3Dpoint(double* depth, double* pose_c2w, double* points_3d, double* camera_intrincis, int rows, int cols){
    //check device property
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    
    int block_dim, thread_work;
    block_dim = devProp.multiProcessorCount * devProp.maxThreadsPerMultiProcessor / 1024;
    //make sure we maximize the use of each thread and in all they can cover all points. 
    thread_work = rows * cols / (block_dim * 1024);

    while(thread_work * block_dim *1024 < rows * cols)
        thread_work++;

    //device variable, allocation memory
    //note you cannot define pointer like thie int* a,b;
    double* d_depth;
    double* d_in;
    double* d_c2w;
    cudaMalloc((void**)&d_depth, rows*cols*sizeof(double));
    cudaMalloc((void**)&d_in, 5*sizeof(double));
    cudaMalloc((void**)&d_c2w, 16*sizeof(double));

    //copy data from the host to the device
    cudaMemcpy(d_depth, depth, rows*cols*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in, camera_intrincis, 5*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c2w, pose_c2w, 16*sizeof(double), cudaMemcpyHostToDevice);

    //block dimension
    dim3 blockNumber(block_dim,1);
    dim3 threadNumber(32,32);

    Calculate3DpointKernel<<<blockNumber, threadNumber>>>(d_depth, d_c2w, points_3d, d_in, thread_work, rows, cols);

    cudaDeviceSynchronize();
    
    //clear
    cudaFree(d_depth);
    cudaFree(d_in);
    cudaFree(d_c2w);
};