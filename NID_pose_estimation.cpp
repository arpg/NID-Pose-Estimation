#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <iostream>
#include <memory>

#include "g2o/g2o/core/block_solver.h"
#include "g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/g2o/solvers/linear_solver_eigen.h"
#include "g2o/g2o/types/types_six_dof_expmap.h"
#include "g2o/g2o/core/robust_kernel_impl.h"
#include "g2o/g2o/solvers/linear_solver_dense.h"


int cell = 16;
int bin_num = 8;
int bs_degree = 3;

struct Intrinsics{
    double fx;
    double fy;
    double cx;
    double cy;
};

std::vector<double> knots(bin_num+bs_degree+1,0.0);
//8 bins, 4 order knot vector example
//static double knots[12] = { 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0};

void Get3dPointAndIntensity(int cell_size, int cell_row_id, int cell_col_id, const cv::Mat& image, const cv::Mat& depth, std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >& points_3d, Eigen::VectorXd& intensity, const Eigen::Matrix4d& T_wc, Intrinsics in);

std::vector<cv::Mat> ReadGroundtruth(std::string add, std::string dataset);

g2o::SE3Quat toSE3Quat(const cv::Mat &cvT)
{
    Eigen::Matrix<double,3,3> R;
    R << cvT.at<double>(0,0), cvT.at<double>(0,1), cvT.at<double>(0,2),
         cvT.at<double>(1,0), cvT.at<double>(1,1), cvT.at<double>(1,2),
         cvT.at<double>(2,0), cvT.at<double>(2,1), cvT.at<double>(2,2);

    Eigen::Matrix<double,3,1> t(cvT.at<double>(0,3), cvT.at<double>(1,3), cvT.at<double>(2,3));

    return g2o::SE3Quat(R,t);
}

int main(int argc, char** argv){
    if(argc!=2){
        std::cout<<"usage './program path_to_config.yaml', image1's timestamp should be smaller than image2"<<std::endl;
        exit(0);
    }

    /*Assign knot vector*/
    for(int i = 0 ; i < bin_num+bs_degree+1; i++){
      if(i<bs_degree+1)
        knots[i] = 0;
      else if(i<bin_num && i>= bs_degree+1)
        knots[i] = i - bs_degree;
      else if(i>=bin_num)
        knots[i] = bin_num - bs_degree;
    }

    std::vector<std::vector<std::vector<double> > > bs;
    std::vector<std::vector<std::vector<double> > > bs_der;
    
    std::string config_add = argv[1];
    cv::FileStorage fc;
    fc.open(config_add, cv::FileStorage::READ);
    std::string type0 = fc["image0_type"] , type1 = fc["image1_type"], id0 = fc["image0_id"], id1 = fc["image1_id"], use_gt = fc["use_groundtruth"], dataset = fc["dataset"], im_add = fc["im_address"];
    double depth_factor = 1.0/(int)fc["depth_factor"], fx = fc["fx"], fy = fc["fy"], cx = fc["cx"], cy = fc["cy"];
    
    cv::Mat im0,im1;
    std::string rgb0_add, rgb1_add;
    if(dataset == "eth_cvg"){
      rgb0_add = im_add + type0 + "/" + id0 + ".png";
      rgb1_add = im_add + type1 + "/" + id1 + ".png";
    }
    else{
      std::cout<<"currently only support testing on eth_cvg dataset"<<std::endl;
      exit(0);
    }

    cv::Mat im_rgb0 = cv::imread(rgb0_add,CV_LOAD_IMAGE_UNCHANGED);//use cv::IMREAD_GRAYSCALE, the result will be different
    im0 = im_rgb0;
    cvtColor(im0,im0,CV_RGB2GRAY);

    cv::Mat im_rgb1 = cv::imread(rgb1_add,CV_LOAD_IMAGE_UNCHANGED);
    im1 = im_rgb1;
    cvtColor(im1,im1,CV_RGB2GRAY);

    std::cout<<"image size "<<im0.size()<<","<<im1.size()<<std::endl;

    //read corresponding depth
    cv::Mat depth0;
    if(dataset == "eth_cvg"){
      std::string depth_add = im_add + "depth/" + id0 + ".png";
      depth0 = cv::imread(depth_add, CV_LOAD_IMAGE_UNCHANGED);
      depth0.convertTo(depth0,CV_64F,depth_factor);
    }

    std::string pose_gt_add;
    if(dataset == "eth_cvg"){
      pose_gt_add = im_add + "groundtruth.txt";
    }
    std::vector<cv::Mat> gt = ReadGroundtruth(pose_gt_add, dataset);
    int pose_id0 = stoi(id0);
    int pose_id1 = stoi(id1);

    cv::Mat T_wc0_cv,T_wc1_cv, T_cw0_cv, T_cw1_cv;

    //pose id output from slam needs to corresponds to the id input of this program
    cv::FileStorage fp0, fp1;
    if(use_gt == "0"){
      if(fp0.open(std::to_string(pose_id0) + ".xml", cv::FileStorage::READ) && fp1.open(std::to_string(pose_id1) + ".xml", cv::FileStorage::READ)){
        std::cout<<"use the pose from etimation"<<std::endl;
        fp0["pose"]>>T_cw0_cv;
        fp1["pose"]>>T_cw1_cv;
        T_cw0_cv.convertTo(T_cw0_cv, CV_64F);
        T_cw1_cv.convertTo(T_cw1_cv, CV_64F);
        T_wc0_cv = T_cw0_cv.inv();
        T_wc1_cv = T_cw1_cv.inv();
      }
      else{
        std::cout<<"no correspoding pose, exit "<<std::endl;
        exit(0);
      }
    }
    else if(use_gt == "1"){
        std::cout<<"use groundtruth pose "<<std::endl;
        T_wc0_cv = gt[pose_id0];
        T_wc1_cv = gt[pose_id1];
    }
    else{
      std::cout<<"you must have pose provided by groundtruth or SLAM"<<std::endl;
    }

    Intrinsics in;
    in.fx = fx;
    in.fy = fy;
    in.cx = cx;
    in.cy = cy;

    Eigen::Matrix4d T_wc0, T_wc1;
    T_wc0<<T_wc0_cv.ptr<double>(0)[0], T_wc0_cv.ptr<double>(0)[1], T_wc0_cv.ptr<double>(0)[2], T_wc0_cv.ptr<double>(0)[3],
          T_wc0_cv.ptr<double>(1)[0], T_wc0_cv.ptr<double>(1)[1], T_wc0_cv.ptr<double>(1)[2], T_wc0_cv.ptr<double>(1)[3],
          T_wc0_cv.ptr<double>(2)[0], T_wc0_cv.ptr<double>(2)[1], T_wc0_cv.ptr<double>(2)[2], T_wc0_cv.ptr<double>(2)[3],
          0                              , 0                              , 0                              , 1;
    T_wc1<<T_wc1_cv.ptr<double>(0)[0], T_wc1_cv.ptr<double>(0)[1], T_wc1_cv.ptr<double>(0)[2], T_wc1_cv.ptr<double>(0)[3],
          T_wc1_cv.ptr<double>(1)[0], T_wc1_cv.ptr<double>(1)[1], T_wc1_cv.ptr<double>(1)[2], T_wc1_cv.ptr<double>(1)[3],
          T_wc1_cv.ptr<double>(2)[0], T_wc1_cv.ptr<double>(2)[1], T_wc1_cv.ptr<double>(2)[2], T_wc1_cv.ptr<double>(2)[3],
          0                              , 0                              , 0                              , 1;

    /*set g2o relates */
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_X::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_X::PoseMatrixType>();

    g2o::BlockSolver_6_X * solver_ptr = new g2o::BlockSolver_6_X(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    optimizer.setAlgorithm(solver);

    optimizer.setVerbose(true);


    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();

    //std::cout<<"before add disturbance the T_wc1 is \n"<<T_wc1<<std::endl;
    std::cout<<"before add disturbance the T_wc1 inverse is \n"<<T_wc1.inverse()<<std::endl;
    const Eigen::Matrix4d SE3_ori= T_wc1;
    Eigen::Vector3d t_dist;
    double t_offset = 0.02; 
    double r_offset = 0.005;

    t_dist<<0.5*t_offset, -t_offset, t_offset;

    Eigen::Matrix3d rotation_dist;
    rotation_dist = Eigen::AngleAxisd(-r_offset*M_PI, Eigen::Vector3d::UnitX())
    * Eigen::AngleAxisd(r_offset*M_PI,  Eigen::Vector3d::UnitY())
    * Eigen::AngleAxisd(-r_offset*M_PI, Eigen::Vector3d::UnitZ());

    Eigen::Matrix4d dist = Eigen::Matrix4d::Zero();
    dist.block<3,3>(0,0) = rotation_dist;
    dist(3,3) = 1.0;

    T_wc1 = SE3_ori * dist;

    Eigen::Matrix3d r_cw1;
    Eigen::Vector3d t_cw1;
    r_cw1 = T_wc1.block<3,3>(0,0).transpose();
    t_cw1 = -r_cw1 * T_wc1.block<3,1>(0,3);
    
    vSE3->setEstimate(g2o::SE3Quat(r_cw1,t_cw1));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);
    std::cout<<"original matrix to be optimized \n"<<vSE3->estimate().to_homogeneous_matrix()<<std::endl;

    int cell_size = cell * cell;
    std::vector<g2o::EdgeSE3ProjectIntensityOnlyPoseNID*> vpEdges;
    std::vector<size_t> vnIndexEdge;
    std::vector<bool> inlier;
    inlier.reserve(cell_size);
    vpEdges.reserve(cell_size);
    vnIndexEdge.reserve(cell_size);

    const double deltaNID = sqrt(0.95);
    //calculate entropy of each cell
    double total_nid = 0.0;
    for(int i = 0; i<cell; i++){
        for(int j = 0; j<cell; j++){
            std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > points_3d;
            Eigen::VectorXd intensity;
            Get3dPointAndIntensity(cell, i, j, im0, depth0, points_3d, intensity, T_wc0, in);

            g2o::EdgeSE3ProjectIntensityOnlyPoseNID* e = new g2o::EdgeSE3ProjectIntensityOnlyPoseNID();

            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));

            e->image1_ = im1;
            e->fx_ = in.fx;
            e->fy_ = in.fy;
            e->cx_ = in.cx;
            e->cy_ = in.cy;
            e->x_world_set_ = points_3d;
            e->knots_ = knots;

            e->setMeasurement(intensity);

            //must first set Measurement then set b spline relates
            e->set_bspline_relates(bs_degree, bin_num);

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(deltaNID);

            Eigen::MatrixXd info = Eigen::MatrixXd::Identity(1,1);
            e->setInformation(info);

            e->computeHref();

            optimizer.addEdge(e);

            optimizer.addEdge(e);
            vpEdges.push_back(e);
            vnIndexEdge.push_back(j+cell*i);
            inlier.push_back(true);

        }
    }

    const float chi2threshold[1]={1.0};
    const int its[1]={10};

    vSE3->setEstimate(g2o::SE3Quat(r_cw1,t_cw1));
    
    optimizer.initializeOptimization(0);
    optimizer.optimize(its[0]);

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    g2o::Vector6d min_vec =  SE3quat_recov.toMinimalVector();
    auto pose = SE3quat_recov.to_homogeneous_matrix();

    std::cout<<"pose optimized \n"<<pose<<std::endl;
    return 0;

}

void Get3dPointAndIntensity(int cell_size, int cell_row_id, int cell_col_id, const cv::Mat& image, const cv::Mat& depth, std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >& points_3d, Eigen::VectorXd& intensity, const Eigen::Matrix4d& T_wc, Intrinsics in){
    int row_block = image.rows / cell_size;
    int col_block = image.cols / cell_size;
    int row_start = row_block * cell_row_id;
    int col_start = col_block * cell_col_id;
    int row_end = row_block * (cell_row_id + 1);
    int col_end = col_block * (cell_col_id + 1);
    int counter = 0;
    std::vector<double> intensity_v;
    for(int i = row_start; i < row_end; i++)
        for(int j = col_start; j < col_end; j++){
            double z_p = depth.ptr<double>(i)[j];
            counter++;
            if(z_p <0.01 || z_p > 100)
                continue;

            double x_p = z_p * (j - in.cx) / in.fx;
            double y_p = z_p * (i - in.cy) / in.fy;

            Eigen::Vector3d p_world = (T_wc*Eigen::Vector4d(x_p,y_p,z_p,1)).head(3);

            points_3d.push_back(p_world);
            intensity_v.push_back(image.ptr<uchar>(i)[j]);
            
        }
    
    intensity = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(intensity_v.data(), intensity_v.size());

    return;
};

std::vector<cv::Mat> ReadGroundtruth(std::string add, std::string dataset){
    std::vector<cv::Mat> all_gt;

    std::ifstream groundtruth_file(add.c_str());
    if(!groundtruth_file){
        printf("cannot find the file that contains groundtruth \n");
        return all_gt;
    }

    int counter = 0;
    std::string one_row_gt;
  
    while(getline(groundtruth_file,one_row_gt)){
        std::istringstream temp_one_row_gt(one_row_gt);
        std::string string_gt;
        Eigen::Matrix3d rm;
        double px,py,pz;
        int sequence = 0;
        if (dataset == "eth_cvg"){
            double qx,qy,qz,qw;
            while(getline(temp_one_row_gt, string_gt, ' ')){
                switch(sequence){
                    case 0:{
                        int msg_timestamp = atoi(string_gt.c_str());
                        sequence++;
                    }
                        break;
                    case 1:{
                        //the case 1,2,3 is position x,y,z
                        px = (double)atof(string_gt.c_str());
                        sequence++;
                    }
                        break;
                    case 2:{
                        py = (double)atof(string_gt.c_str());
                        sequence++;
                    }
                        break;
                    case 3:{
                        pz = (double)atof(string_gt.c_str());
                        sequence++;
                    }
                        break;
                    case 4:{
                        //the case 4,5,6,7 is quternion x,y,z,w
                        qx = (double)atof(string_gt.c_str());
                        sequence++;
                    }
                        break;
                    case 5:{
                        qy = (double)atof(string_gt.c_str());
                        sequence++;
                    }
                        break;
                    case 6:{
                        qz = (double)atof(string_gt.c_str());
                        sequence++;
                    }
                        break;
                    case 7:{
                        qw = (double)atof(string_gt.c_str());
                        sequence++;
                    }
                        break;
                    default:
                        break;
                }
            }
            Eigen::Quaterniond qt(qw,qx,qy,qz);
            rm = qt.toRotationMatrix();
        }
    
        cv::Mat one_gt(4,4,CV_64FC1);
        one_gt.ptr<double>(0)[0] = rm(0,0);
        one_gt.ptr<double>(0)[1] = rm(0,1);
        one_gt.ptr<double>(0)[2] = rm(0,2);
        one_gt.ptr<double>(1)[0] = rm(1,0);
        one_gt.ptr<double>(1)[1] = rm(1,1);
        one_gt.ptr<double>(1)[2] = rm(1,2);
        one_gt.ptr<double>(2)[0] = rm(2,0);
        one_gt.ptr<double>(2)[1] = rm(2,1);
        one_gt.ptr<double>(2)[2] = rm(2,2);
        one_gt.ptr<double>(3)[0] = 0.0;
        one_gt.ptr<double>(3)[1] = 0.0;
        one_gt.ptr<double>(3)[2] = 0.0;
        
        one_gt.ptr<double>(0)[3] = px;
        one_gt.ptr<double>(1)[3] = py;
        one_gt.ptr<double>(2)[3] = pz;
        one_gt.ptr<double>(3)[3] = 1.0;

        all_gt.push_back(one_gt);

  }

  return all_gt;
};







