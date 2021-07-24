#include "CudaComputeHref.cuh"
#include <sys/time.h>
#include <iostream>

__device__ double Bspline(int index, int order, double u, double* knots){
    double coef1, coef2;
    if ( order == 1 )
    {
      if ( index == 0 ) if ( ( knots[index] <= u ) && ( u <= knots[index+1] ) ) return 1.0;
      if ( ( knots[index] < u ) && ( u <= knots[index+1] ) ) return 1.0;
      else return 0.0;
    }
    else
    {
      if ( knots[index + order - 1] == knots[index] ) 
      {
          if ( u == knots[index] ) coef1 = 1;
          else coef1 = 0;
      }
      else coef1 = (u - knots[index])/(knots[index + order - 1] - knots[index]);
  
      if ( knots[index + order] == knots[index+1] )
      {
          if ( u == knots[index + order] ) coef2 = 1;
          else coef2 = 0;
      }
      else coef2 = (knots[index + order] - u)/(knots[index + order] - knots[index+1]);
      
      return ( coef1 * Bspline(index, order-1, u, knots) + coef2 * Bspline(index+1,order-1 ,u, knots));
    }
};

__global__ void CalculateHrefKernel(double* im0, double* points3d, double* pose, double* in, int thread_work, int bin_num, int bs_degree, int rows, int cols, int cell, double* bs_value, int* bs_index, double* d_pro, int* d_bs_counter){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    int id = thread_work * (idy * blockDim.x * gridDim.x + idx);

    //6 bins, 4 order
    double knots6[10] = { 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0};

    //8 bin, 4 order
    double knots8[12] = { 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0};

    //10 bins, 4 order
    double knots10[14] = { 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0, 7.0};

    //12 bins, 4 order
    double knots12[16] = { 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.0, 9.0, 9.0};

    //14 bins, 4 order
    double knots14[18] = { 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 11.0, 11.0, 11.0};

    double* knots;
    switch(bin_num){
        case 6:
            knots = &knots6[0];
            break;
        case 8:
            knots = &knots8[0];
            break;
        case 10:
            knots = &knots10[0];
            break;
        case 12:
            knots = &knots12[0];
            break;
        case 14:
            knots = &knots14[0];
            break;
        default:
            knots = &knots10[0];
            break;
    }

    for(int i = id; i<thread_work+id; i++){
        double x0,y0,z0,x1,y1,z1;
        x0 = points3d[3*i];
        y0 = points3d[3*i + 1];
        z0 = points3d[3*i + 2];
        if(isnan(x0) || isnan(y0) || isnan(z0)){
            bs_index[i] = NAN;
            bs_value[4*i] = NAN;
            bs_value[4*i + 1] = NAN;
            bs_value[4*i + 2] = NAN;
            bs_value[4*i + 3] = NAN;
            continue;
        }

        //printf("pixel 3d point and intensity %f, %f, %f, %f", i/cols, i%cols, im0[i]);
        //pose is world to cam1
        x1 = pose[0] * x0 + pose[4] * y0 + pose[8] * z0 + pose[12];
        y1 = pose[1] * x0 + pose[5] * y0 + pose[9] * z0 + pose[13];
        z1 = pose[2] * x0 + pose[6] * y0 + pose[10]* z0 + pose[14];

        //make sure after we mapping to another image we still have the pixel in the frame
        double u = in[0] * x1 / z1 + in[2];
        double v = in[1] * y1 / z1 + in[3];

        //bilinear interporlation of pixel. DSO getInterpolatedElement33() function, bilinear interpolation
        if(u >= 0 && u+3 <= cols && v >= 0 && v+3 <= rows){
            if(im0[i] >= 255)
                im0[i] = 254.999;
            if(im0[i] < 0)
                im0[i] = 0;
            double bin_pos_ref =  im0[i] * (bin_num- bs_degree)/255.0;
            double pos_cubic_ref = bin_pos_ref * bin_pos_ref * bin_pos_ref;
            double pos_qua_ref  = bin_pos_ref * bin_pos_ref;
            int bins_index_ref = floor(bin_pos_ref);

            bs_index[i] = bins_index_ref;
            bs_value[4*i]     = Bspline(bins_index_ref, bs_degree+1, bin_pos_ref, knots);
            bs_value[4*i + 1] = Bspline(bins_index_ref+1, bs_degree+1, bin_pos_ref, knots);
            bs_value[4*i + 2] = Bspline(bins_index_ref+2, bs_degree+1, bin_pos_ref, knots);
            bs_value[4*i + 3] = Bspline(bins_index_ref+3, bs_degree+1, bin_pos_ref, knots);

            int cell_id = (i/cols)/(rows/cell) * cell + (i%cols)/(cols/cell);

            atomicAdd(&d_bs_counter[cell_id],1);
            atomicAdd(&d_pro[cell_id*bin_num+bins_index_ref], bs_value[4*i]);
            atomicAdd(&d_pro[cell_id*bin_num+bins_index_ref+1], bs_value[4*i+1]);
            atomicAdd(&d_pro[cell_id*bin_num+bins_index_ref+2], bs_value[4*i+2]);
            atomicAdd(&d_pro[cell_id*bin_num+bins_index_ref+3], bs_value[4*i+3]);
        }
        else{
            bs_index[i] = NAN;
            bs_value[4*i] = NAN;
            bs_value[4*i + 1] = NAN;
            bs_value[4*i + 2] = NAN;
            bs_value[4*i + 3] = NAN;
        }

    }

};



void CudaComputeHref(double* im0, double* points3d, double* pose, double* camera_intrincis, int bin_num, int bs_degree, int cell_num, int rows, int cols, double* bs_value, int* bs_index, int* bs_counter, double* Href){
    //check device property
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

    //make sure we maximize the use of each thread and in all they can cover all points. 
    int block_dim, thread_work;
    block_dim = devProp.multiProcessorCount * devProp.maxThreadsPerMultiProcessor / 1024;
    thread_work = rows * cols / (block_dim * 1024);
    while(thread_work * block_dim *1024 < rows * cols)
        thread_work++;
    

    //device variable, allocation memory
    //note you cannot define pointer like thie int* a,b;
    double* d_bs_value;
    double* d_in;
    double* d_pose;
    int* d_bs_index;
    cudaMalloc((void**)&d_pose, 4*4*sizeof(double));
    cudaMalloc((void**)&d_bs_value, 4*rows*cols*sizeof(double));
    cudaMalloc((void**)&d_bs_index, rows*cols*sizeof(int));
    cudaMalloc((void**)&d_in, 5*sizeof(double));

    //test
    double* pro = (double*)malloc(bin_num*cell_num*cell_num*sizeof(double));
    //int* bs_counter = (int*)malloc(cell_num*cell_num*sizeof(int));
    double* d_pro;
    int* d_bs_counter;
    cudaMalloc((void**)&d_pro, bin_num*cell_num*cell_num*sizeof(double));
    cudaMalloc((void**)&d_bs_counter, cell_num*cell_num*sizeof(int));
    //set initial value as 0
    cudaMemset((void**)&d_pro, 0, bin_num*cell_num*cell_num*sizeof(double));
    cudaMemset((void**)&d_bs_counter, 0, cell_num*cell_num*sizeof(int));

    //copy data from the host to the device
    cudaMemcpy(d_in, camera_intrincis, 5*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pose, pose, 4*4*sizeof(double), cudaMemcpyHostToDevice);

    //block dimension
    dim3 block_num(block_dim,4);
    dim3 thread_num(16,16);

    CalculateHrefKernel<<<block_num, thread_num>>>(im0, points3d, d_pose, d_in, thread_work, bin_num, bs_degree, rows, cols, cell_num, d_bs_value, d_bs_index, d_pro, d_bs_counter);

    //copy data
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error 0\"%s\".\n", cudaGetErrorString(cudaerr));

    cudaMemcpy(bs_value, d_bs_value, sizeof(double) * rows * cols * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(bs_index, d_bs_index, sizeof(int) * rows * cols, cudaMemcpyDeviceToHost); 

    //test
    cudaMemcpy(bs_counter, d_bs_counter, sizeof(int) * cell_num * cell_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(pro, d_pro, sizeof(double) * cell_num * cell_num * bin_num, cudaMemcpyDeviceToHost); 


    cudaFree(d_in);
    cudaFree(d_bs_value);
    cudaFree(d_bs_index);

    //test
    cudaFree(d_bs_counter);
    cudaFree(d_pro);

    double sigma = 1e-30;
    //std::cout<<"the Href or the counter is "<<std::endl;
    for(int i = 0; i<cell_num*cell_num; i++){
        if(bs_counter[i]<300){
            Href[i] = NAN;
            continue;
        }
        for(int j = 0; j < bin_num ; j++){
            pro[i*bin_num+j] /= (double)bs_counter[i];
        }
        for(int j = 0; j < bin_num ; j++){
            if(pro[i*bin_num+j] < sigma)
              continue;
            Href[i] -= pro[i*bin_num+j] * log2(pro[i*bin_num+j]);
        }
    }

    free(pro);
}