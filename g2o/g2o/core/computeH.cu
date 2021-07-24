#include "computeH.cuh"

#include <chrono>

  // get a gray scale value from reference image (bilinear interpolated)
  __device__ double get_interpolated_pixel_value0 ( double x, double y, double* im1, int cols)
  {
    int ix = (int)x;
    int iy = (int)y;
    
    double dx = x - ix;
    double dy = y - iy;
    double dxdy = dx*dy; 

    double xx = x - floor ( x );
    double yy = y - floor ( y );

    return double (
      dxdy * im1[(iy+1)*cols + ix+1]
      + (dy - dxdy) * im1[(iy+1)*cols + ix]
      + (dx - dxdy) * im1[iy*cols + ix + 1]
      + (1 - dx - dy + dxdy) * im1[iy*cols + ix]
    );
  }


__device__ double Bspline0(int index, int order, double u, double* knots){
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
      
      return ( coef1 * Bspline0(index, order-1, u, knots) + coef2 * Bspline0(index+1,order-1 ,u, knots));
    }
};

__device__ double BsplineDer0(int index, int order, double u, double* knots){
    double coef1, coef2, coef3, coef4;
    if ( order == 1 )
    {
      return 0.0;
    }
    else
    {
      if ( knots[index + order - 1] == knots[index] ) 
      {
        if ( u == knots[index] ) coef1 = 1;
        else coef1 = 0;
  
        coef3 = 0.0;
      }
      else {
        coef1 = (u - knots[index])/(knots[index + order - 1] - knots[index]);
        coef3 = 1.0 / (knots[index + order - 1] - knots[index]);
      }
  
      if ( knots[index + order] == knots[index+1] )
      {
        if ( u == knots[index + order] ) coef2 = 1;
        else coef2 = 0;
  
        coef4 = 0.0;
      }
      else {
        coef2 = (knots[index + order] - u)/(knots[index + order] - knots[index+1]);
        coef4 = -1.0/(knots[index + order] - knots[index+1]);
      }
      
      return ( coef1 * BsplineDer0(index, order-1, u, knots) + coef2 * BsplineDer0(index+1, order-1 ,u, knots) + coef3 * Bspline0(index, order-1, u, knots) + coef4 * Bspline0(index+1, order-1 ,u,knots) );
    }
};



__global__ void CalculateProKernel(bool calculate_der, double* im0, double* im1, double* points3d, double* bs_value_ref, int*bs_index_ref, double* pose, double* in, int thread_work, int bin_num, int bs_degree, int rows, int cols, int cell, double* d_d_sum_bs_pose, double* d_d_sum_joint_bs_pose, double* d_pro_target, double* d_pro_joint){
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

    int bin_col = bin_num * cell * cell;



    for(int i = id; i<thread_work+id; i++){
        double x0,y0,z0,x1,y1,z1;
        x0 = points3d[3*i];
        y0 = points3d[3*i + 1];
        z0 = points3d[3*i + 2];
        if(isnan(x0) || isnan(y0) || isnan(z0)){
            continue;
        }

        double d_mi_i = (bin_num- bs_degree)/255.0;

        //pose is world to cam1
        x1 = pose[0] * x0 + pose[4] * y0 + pose[8] * z0 + pose[12];
        y1 = pose[1] * x0 + pose[5] * y0 + pose[9] * z0 + pose[13];
        z1 = pose[2] * x0 + pose[6] * y0 + pose[10]* z0 + pose[14];

        //make sure after we mapping to another image we still have the pixel in the frame
        double u = in[0] * x1 / z1 + in[2];
        double v = in[1] * y1 / z1 + in[3];


        double jacobian_pixel_uv[2];
        double jacobian_uv_ksai[12];//2*6
        //bilinear interporlation of pixel. DSO getInterpolatedElement33() function, bilinear interpolation
        if(u >= 0 && u+3 <= cols && v >= 0 && v+3 <= rows){
            double ic = get_interpolated_pixel_value0(u,v,im1, cols);//current intensity

            if(ic >= 255)
                ic = 254.999;
            if(ic < 0)
                ic = 0;
            double bin_pos_target =  ic * (bin_num- bs_degree)/255.0;
            int bins_index_target = floor(bin_pos_target);

            double d_i_pose[6];
            double d_bs_mi[4] = {0,0,0,0};
            if(calculate_der){
                double invz = 1.0/z1;
                double invz_2 = invz*invz;
                jacobian_pixel_uv[0] = ( get_interpolated_pixel_value0 ( u+1,v, im1, cols )-get_interpolated_pixel_value0 ( u-1,v, im1, cols ) ) /2;
                jacobian_pixel_uv[1] = ( get_interpolated_pixel_value0 ( u,v+1, im1, cols )-get_interpolated_pixel_value0 ( u,v-1, im1, cols ) ) /2;

                jacobian_uv_ksai[0] = - x1*y1*invz_2 *in[0];
                jacobian_uv_ksai[1] = ( 1+ ( x1*x1*invz_2 ) ) *in[0];
                jacobian_uv_ksai[2] = - y1*invz *in[0];
                jacobian_uv_ksai[3] = invz *in[0];
                jacobian_uv_ksai[4] = 0;
                jacobian_uv_ksai[5] = -x1*invz_2 * in[0];
        
                jacobian_uv_ksai[6] = - ( 1+y1*y1*invz_2 ) *in[1];
                jacobian_uv_ksai[7] = x1*y1*invz_2 *in[1];
                jacobian_uv_ksai[8] = x1*invz *in[1];
                jacobian_uv_ksai[9] = 0;
                jacobian_uv_ksai[10] = invz *in[1];
                jacobian_uv_ksai[11] = -y1*invz_2 *in[1];
            

                //d_i_pose = jacobian_pixel_uv * jacobian_uv_ksai;
                d_i_pose[0] = jacobian_pixel_uv[0] * jacobian_uv_ksai[0] + jacobian_pixel_uv[1] * jacobian_uv_ksai[6];
                d_i_pose[1] = jacobian_pixel_uv[0] * jacobian_uv_ksai[1] + jacobian_pixel_uv[1] * jacobian_uv_ksai[7];
                d_i_pose[2] = jacobian_pixel_uv[0] * jacobian_uv_ksai[2] + jacobian_pixel_uv[1] * jacobian_uv_ksai[8];
                d_i_pose[3] = jacobian_pixel_uv[0] * jacobian_uv_ksai[3] + jacobian_pixel_uv[1] * jacobian_uv_ksai[9];
                d_i_pose[4] = jacobian_pixel_uv[0] * jacobian_uv_ksai[4] + jacobian_pixel_uv[1] * jacobian_uv_ksai[10];
                d_i_pose[5] = jacobian_pixel_uv[0] * jacobian_uv_ksai[5] + jacobian_pixel_uv[1] * jacobian_uv_ksai[11];

                d_bs_mi[0]     = BsplineDer0(bins_index_target, bs_degree+1, bin_pos_target, knots);
                d_bs_mi[1] = BsplineDer0(bins_index_target+1, bs_degree+1, bin_pos_target, knots);
                d_bs_mi[2] = BsplineDer0(bins_index_target+2, bs_degree+1, bin_pos_target, knots);
                d_bs_mi[3] = BsplineDer0(bins_index_target+3, bs_degree+1, bin_pos_target, knots);
            }

            //accidently I use bins_index_target before for the third parameters and get a better performance for some images, which is wired...
            double bs_value_target[4] = {0,0,0,0};
            bs_value_target[0]     = Bspline0(bins_index_target, bs_degree+1, bin_pos_target, knots);
            bs_value_target[1] = Bspline0(bins_index_target+1, bs_degree+1, bin_pos_target, knots);
            bs_value_target[2] = Bspline0(bins_index_target+2, bs_degree+1, bin_pos_target, knots);
            bs_value_target[3] = Bspline0(bins_index_target+3, bs_degree+1, bin_pos_target, knots);



            int cell_id = (i/cols)/(rows/cell) * cell + (i%cols)/(cols/cell);

            atomicAdd(&d_pro_target[cell_id*bin_num+bins_index_target],   bs_value_target[0]);
            atomicAdd(&d_pro_target[cell_id*bin_num+bins_index_target+1], bs_value_target[1]);
            atomicAdd(&d_pro_target[cell_id*bin_num+bins_index_target+2], bs_value_target[2]);
            atomicAdd(&d_pro_target[cell_id*bin_num+bins_index_target+3], bs_value_target[3]);


            if(calculate_der){
                for(int m = 0; m<bs_degree+1; m++)
                    for(int n = 0; n<6; n++){
        
                    atomicAdd(&d_d_sum_bs_pose[6*bin_num*cell_id + 6*(bins_index_target + m) + n], d_bs_mi[m] * d_mi_i * d_i_pose[n]);
                    
                };
            }

            int bins_index_ref = bs_index_ref[i];
            for(int m = 0 ; m<bs_degree+1; m++)
               for(int n = 0; n<bs_degree+1; n++){
                double tmp = bs_value_ref[4*i+m] * bs_value_target[n];
                
                atomicAdd(&d_pro_joint[(cell_id*bin_num+bins_index_ref+m) * bin_col + cell_id*bin_num+bins_index_target+n], tmp); 
            }

            if(calculate_der){
                for(int k = 0; k<bs_degree+1; k++)
                    for(int m = 0; m<bs_degree+1; m++){
                        for(int n = 0; n<6; n++){
                            double tmp = bs_value_ref[4*i+k] * d_bs_mi[m] * d_mi_i * d_i_pose[n];
                    //3D to 1D. Tricky
                            atomicAdd(&d_d_sum_joint_bs_pose[6*((bin_num*cell_id + bins_index_ref + k) * bin_col + cell_id * bin_num + bins_index_target + m) + n], tmp);
                    }
                }
            }
        }

    }

};

__global__ void CalculateHKernel(int bin_num, int bs_degree, int rows, int cols, int cell, double* pro_target, double* pro_joint, int* bs_counter, double* Htarget, double* Hjoint){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    int bin_col = bin_num * cell * cell;

    int i = idy * blockDim.x * gridDim.x + idx;

    double sigma = 1e-30;

    if(bs_counter[i]<300){
        Htarget[i] = NAN;
        Hjoint[i]  = NAN;
        return;
    }

    for(int j = 0; j < bin_num ; j++){
        pro_target[i*bin_num+j] /= (double)bs_counter[i];
    }
    
    for(int j = 0; j < bin_num ; j++){
        if(pro_target[i*bin_num+j] < sigma){
            continue;
        }
        Htarget[i] -= pro_target[i*bin_num+j] * log2(pro_target[i*bin_num+j]);
    }
    
    for(int m = 0; m < bin_num ; m++)
        for(int n = 0; n < bin_num ; n++){
            pro_joint[(i*bin_num+m)*bin_col + i*bin_num+n] /= (double)bs_counter[i];
    }

    for(int m = 0; m < bin_num ; m++)
        for(int n = 0; n < bin_num ; n++){
            if(pro_joint[(i*bin_num+m)*bin_col + i*bin_num+n]<sigma){
                continue;
            }
            Hjoint[i] -= pro_joint[(i*bin_num+m)*bin_col + i*bin_num+n] * log2(pro_joint[(i*bin_num+m)*bin_col + i*bin_num+n]);
        }
};


__global__ void CalculateDerKernel(int bin_num, int bs_degree, int rows, int cols, int cell, int* bs_counter, double* d_d_sum_bs_pose, double* d_d_sum_joint_bs_pose, double* d_pro_target, double* d_pro_joint, double* d_htarget, double* d_hjoint, double* d_href, double* der){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    int bin_col = bin_num * cell * cell;

    int i = idy * blockDim.x * gridDim.x + idx;

    double sigma = 1e-30;

    if(bs_counter[i]<300){
        //Htarget[i] = NAN;
        //Hjoint[i]  = NAN;
        der[6*i]   = NAN;
        der[6*i+1] = NAN;
        der[6*i+2] = NAN;
        der[6*i+3] = NAN;
        der[6*i+4] = NAN;
        der[6*i+5] = NAN;
        return;
    }

    for(int m = 0; m<bin_num; m++)
        for(int n = 0; n<6; n++){
            d_d_sum_bs_pose[6*(i*bin_num+m)+n] /= bs_counter[i];
    }

    for(int m = 0; m<bin_num; m++)
        for(int n = 0; n<bin_num; n++)
            for(int k = 0; k<6; k++){
            d_d_sum_joint_bs_pose[6*((i*bin_num+m)*bin_col + i*bin_num+n)+k] /= bs_counter[i];
      }

    double d_hj_p[6] = {0,0,0,0,0,0};
    double d_hl_p[6] = {0,0,0,0,0,0};

    for(int k = 0; k < 6; k++){
        double tmp = 0.0;
        for(int m = 0; m<bin_num; m++){
          for(int n = 0; n<bin_num; n++){
            if(d_pro_joint[(i*bin_num+m)*bin_col + i*bin_num + n]< sigma)
              continue;
            tmp -= (1.0 + log2(d_pro_joint[(i*bin_num+m)*bin_col + i*bin_num + n])) * d_d_sum_joint_bs_pose[6*((i*bin_num+m)*bin_col + i*bin_num+n)+k];
          }
        }
        d_hj_p[k] = tmp;
    } 
    
    for(int k = 0; k < 6; k++)
    for(int m = 0; m < bin_num; m++){
      if(d_pro_target[i*bin_num+m] < sigma)
        continue;
      d_hl_p[k] -= (1.0 + log2(d_pro_target[i*bin_num+m])) * d_d_sum_bs_pose[6*(i*bin_num+m)+k];
    }

    double inv_square_hj = 1.0/(d_hjoint[i] * d_hjoint[i]);


    der[6*i]   = (d_hj_p[0] * (d_htarget[i] + d_href[i]) - d_hl_p[0] * d_hjoint[i]) * inv_square_hj;
    der[6*i+1] = (d_hj_p[1] * (d_htarget[i] + d_href[i]) - d_hl_p[1] * d_hjoint[i]) * inv_square_hj;
    der[6*i+2] = (d_hj_p[2] * (d_htarget[i] + d_href[i]) - d_hl_p[2] * d_hjoint[i]) * inv_square_hj;
    der[6*i+3] = (d_hj_p[3] * (d_htarget[i] + d_href[i]) - d_hl_p[3] * d_hjoint[i]) * inv_square_hj;
    der[6*i+4] = (d_hj_p[4] * (d_htarget[i] + d_href[i]) - d_hl_p[4] * d_hjoint[i]) * inv_square_hj;
    der[6*i+5] = (d_hj_p[5] * (d_htarget[i] + d_href[i]) - d_hl_p[5] * d_hjoint[i]) * inv_square_hj;
    
}


namespace g2o{

    void CudaComputeH(bool calculate_der, double* im0, double*im1, double* points3d, int* bs_counter, double* bs_value_ref, int* bs_index_ref, double* pose, double* camera_intrincis, int bin_num, int bs_degree, int cell_num, int rows, int cols, double* Href, double* pro_target, double* pro_joint, double* Htarget, double* Hjoint, double* der){
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    
        //check device property
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, 0);
    
        //make sure we maximize the use of each thread and in all they can cover all points. 
        int block_dim, thread_work;
        block_dim = devProp.multiProcessorCount * devProp.maxThreadsPerMultiProcessor / 1024;
        thread_work = rows * cols / (block_dim * 1024);
        while(thread_work * block_dim *1024 < rows * cols)
            thread_work++;

        double* d_htarget;
        double* d_hjoint;
        double* d_href;
        double* d_in;
        double* d_pose;
        double* d_der;
        int* d_bs_counter;
        double* d_bs_value_ref;
        int* d_bs_index_ref;
        cudaMalloc((void**)&d_pose, 4*4*sizeof(double));
        cudaMalloc((void**)&d_in, 5*sizeof(double));
        cudaMalloc((void**)&d_bs_counter, cell_num*cell_num*sizeof(int));
        cudaMalloc((void**)&d_bs_value_ref, cols * rows * 4 * sizeof(double));
        cudaMalloc((void**)&d_bs_index_ref, cols * rows * sizeof(int));
        cudaMalloc((void**)&d_htarget, cell_num*cell_num*sizeof(double));
        cudaMalloc((void**)&d_hjoint, cell_num*cell_num*sizeof(double));
        cudaMalloc((void**)&d_href, cell_num*cell_num*sizeof(double));

        if(calculate_der)
            cudaMalloc((void**)&d_der, 6*cell_num*cell_num*sizeof(double));
    
        //test
        int bin_cell = bin_num*cell_num*cell_num;
        double* d_pro_target;
        double* d_pro_joint;

        cudaMalloc((void**)&d_pro_joint, bin_cell*bin_cell*sizeof(double));
        cudaMalloc((void**)&d_pro_target, bin_cell*sizeof(double));
        //set initial value as 0
        cudaMemset(d_pro_joint, 0, bin_cell*bin_cell*sizeof(double));
        cudaMemset(d_pro_target, 0, bin_cell*sizeof(double));
    
        //copy data from the host to the device
        cudaMemcpy(d_in, camera_intrincis, 5*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pose, pose, 4*4*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bs_value_ref, bs_value_ref, cols * rows * 4 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bs_counter, bs_counter, cell_num*cell_num*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bs_index_ref, bs_index_ref, cols * rows * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_htarget, Htarget, cell_num*cell_num*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_hjoint, Hjoint, cell_num*cell_num*sizeof(double), cudaMemcpyHostToDevice);

        if(calculate_der)
            cudaMemcpy(d_href, Href, cell_num*cell_num*sizeof(double), cudaMemcpyHostToDevice);

        //test jacobian calculation
        double* d_d_sum_bs_pose;
        double* d_d_sum_joint_bs_pose;

        if(calculate_der){
            cudaMalloc((void**)&d_d_sum_bs_pose, 6*bin_cell*sizeof(double));
            cudaMemset(d_d_sum_bs_pose, 0, 6*bin_cell*sizeof(double));
            cudaMalloc((void**)&d_d_sum_joint_bs_pose, 6*bin_cell*bin_cell*sizeof(double));
            cudaMemset(d_d_sum_joint_bs_pose, 0, 6*bin_cell*bin_cell*sizeof(double));
        }
    
        //block dimension
        dim3 block_num0(block_dim,4);
        dim3 thread_num0(16,16);


        size_t limit = 2048;

        cudaDeviceSetLimit(cudaLimitStackSize, limit);

        CalculateProKernel<<<block_num0, thread_num0>>>(calculate_der, im0, im1, points3d, d_bs_value_ref, d_bs_index_ref, d_pose, d_in, thread_work, bin_num, bs_degree, rows, cols, cell_num, d_d_sum_bs_pose, d_d_sum_joint_bs_pose, d_pro_target, d_pro_joint);

        //copy data
        cudaError_t cudaerr0 = cudaDeviceSynchronize();
        if (cudaerr0 != cudaSuccess)
            printf("kernel launch failed with error 1\"%s\".\n", cudaGetErrorString(cudaerr0));

        dim3 block_num1(1,1);
        dim3 thread_num1(cell_num,cell_num);

        CalculateHKernel<<<block_num1, thread_num1>>>(bin_num, bs_degree, rows, cols, cell_num, d_pro_target, d_pro_joint, d_bs_counter, d_htarget, d_hjoint);

        cudaError_t cudaerr1 = cudaDeviceSynchronize();
        if (cudaerr1 != cudaSuccess)
            printf("kernel launch failed with error 2\"%s\".\n", cudaGetErrorString(cudaerr1));


        if(calculate_der){
            CalculateDerKernel<<<block_num1, thread_num1>>>(bin_num, bs_degree, rows, cols, cell_num, d_bs_counter, d_d_sum_bs_pose, d_d_sum_joint_bs_pose, d_pro_target, d_pro_joint, d_htarget, d_hjoint, d_href, d_der);

            cudaError_t cudaerr2 = cudaDeviceSynchronize();
            if (cudaerr2 != cudaSuccess)
                    printf("kernel launch failed with error 3\"%s\".\n", cudaGetErrorString(cudaerr2));
        }        
    
        //test
        cudaMemcpy(Htarget, d_htarget, cell_num*cell_num*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(Hjoint, d_hjoint, cell_num*cell_num*sizeof(double), cudaMemcpyDeviceToHost);

        if(calculate_der)
            cudaMemcpy(der, d_der, 6*cell_num*cell_num*sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_in);
        cudaFree(d_pose);
        cudaFree(d_pro_joint);
        cudaFree(d_pro_target);
        cudaFree(d_bs_counter);
        cudaFree(d_bs_value_ref);
        cudaFree(d_bs_index_ref);
        cudaFree(d_htarget);
        cudaFree(d_hjoint);

        if(calculate_der){
            cudaFree(d_der);
            cudaFree(d_d_sum_bs_pose);
            cudaFree(d_d_sum_joint_bs_pose);
        }
    
    
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> time_span = t2 - t1;
    };
}
