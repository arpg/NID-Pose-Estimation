#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <iostream>
#include <memory>


int cell = 16;
int bin_num = 8;

struct Intrinsics{
    double fx;
    double fy;
    double cx;
    double cy;
};

void Get3dPointAndIntensity(int cell_size, int cell_row_id, int cell_col_id, const cv::Mat& image, const cv::Mat& depth, std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >& points_3d, Eigen::VectorXd& intensity, const Eigen::Matrix4d& T_wc, Intrinsics in);

std::vector<cv::Mat> ReadGroundtruth(std::string add, std::string dataset);

class NID{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void set_member_size();
    void ComputeHref();
    void ComputeH();

    int cell_ = 16;
    int bin_num_ ;
    Eigen::VectorXd intensity0_;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > pw0_;
    Eigen::Matrix4d tf_;
    cv::Mat image1_;
    double fx_;
    double fy_;
    double cx_;
    double cy_;
    std::vector<double> pro_ref_, pro_current_;
    std::vector<std::vector<double> > pro_joint_;
    double sigma_ = 1e-30;
    double H_ref_ = 0.0, H_current_ = 0.0, H_joint_ = 0.0;
    Eigen::VectorXd intensity_current_;
    double nid_, MI_;

    inline double get_interpolated_pixel_value ( double x, double y )
    {
        int ix = (int)x;
        int iy = (int)y;

        double dx = x - ix;
        double dy = y - iy;
        double dxdy = dx*dy; 

        double xx = x - floor ( x );
        double yy = y - floor ( y );

        return double (
        dxdy * image1_.ptr<uchar>(iy+1)[ix+1] 
        + (dy - dxdy) * image1_.ptr<uchar>(iy+1)[ix]
        + (dx - dxdy) * image1_.ptr<uchar>(iy)[ix+1]
        + (1 - dx - dy + dxdy) * image1_.ptr<uchar>(iy)[ix]
        );
    }
};

int main(int argc, char** argv){
    if(argc!=2){
        std::cout<<"usage './program path_to_config.yaml', image1's timestamp should be smaller than image2"<<std::endl;
        exit(0);
    }
    
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

    //std::cout<<"pose id is "<<pose_id<<std::endl;
    Intrinsics in;
    in.fx = fx;
    in.fy = fy;
    in.cx = cx;
    in.cy = cy;

    std::vector<std::shared_ptr<NID> > nid_vec;
    Eigen::Matrix4d T_wc0, T_wc1;
    T_wc0<<T_wc0_cv.ptr<double>(0)[0], T_wc0_cv.ptr<double>(0)[1], T_wc0_cv.ptr<double>(0)[2], T_wc0_cv.ptr<double>(0)[3],
          T_wc0_cv.ptr<double>(1)[0], T_wc0_cv.ptr<double>(1)[1], T_wc0_cv.ptr<double>(1)[2], T_wc0_cv.ptr<double>(1)[3],
          T_wc0_cv.ptr<double>(2)[0], T_wc0_cv.ptr<double>(2)[1], T_wc0_cv.ptr<double>(2)[2], T_wc0_cv.ptr<double>(2)[3],
          0                              , 0                              , 0                              , 1;
    T_wc1<<T_wc1_cv.ptr<double>(0)[0], T_wc1_cv.ptr<double>(0)[1], T_wc1_cv.ptr<double>(0)[2], T_wc1_cv.ptr<double>(0)[3],
          T_wc1_cv.ptr<double>(1)[0], T_wc1_cv.ptr<double>(1)[1], T_wc1_cv.ptr<double>(1)[2], T_wc1_cv.ptr<double>(1)[3],
          T_wc1_cv.ptr<double>(2)[0], T_wc1_cv.ptr<double>(2)[1], T_wc1_cv.ptr<double>(2)[2], T_wc1_cv.ptr<double>(2)[3],
          0                              , 0                              , 0                              , 1;

    std::ofstream of("nid_test.csv", std::ofstream::out | std::ofstream::app);

    const double px_ori = T_wc0(0,3);
    const Eigen::Matrix4d SE3_ori= T_wc0;

    //calculate entropy of each cell
    double total_nid = 0.0;
    for(int i = 0; i<cell; i++){
        for(int j = 0; j<cell; j++){
            std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > points_3d;
            Eigen::VectorXd intensity;
            Get3dPointAndIntensity(cell, i, j, im0, depth0, points_3d, intensity, T_wc0, in);
            std::shared_ptr<NID> nid = std::make_shared<NID>();
            nid->cell_ = cell;
            nid->bin_num_ = bin_num;
            nid->image1_ = im1;
            nid->fx_ = in.fx;
            nid->fy_ = in.fy;
            nid->cx_ = in.cx;
            nid->cy_ = in.cy;
            nid->pw0_ = points_3d;
            nid->intensity0_ = intensity;
            nid->tf_ = T_wc1;
            nid->set_member_size();
            nid->ComputeHref();
            nid->ComputeH();
            total_nid += nid->nid_ * nid->nid_;
            //exit(0);
        }
    }

    std::cout<<"final nid is "<<sqrt(total_nid)<<std::endl;
    

    

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

  //std::cout<<"size of all gt "<<all_gt.size()<<std::endl;

  return all_gt;
};

void NID::set_member_size(){
    pro_current_ = std::vector<double>(bin_num_,0.0);
    pro_ref_ = std::vector<double>(bin_num_,0.0);
    pro_joint_ = std::vector<std::vector<double> >(bin_num_);
    for(int i = 0; i < bin_num_; i++){
        pro_joint_[i] = std::vector<double>(bin_num_, 0.0);
    }
    intensity_current_ = Eigen::VectorXd::Zero(intensity0_.rows());
}

void NID::ComputeHref(){
  
  Eigen::Matrix4d T_cw1 = tf_.inverse();
  int ob = 0;
  for(int i = 0 ; i<intensity0_.rows(); i++){

    //decide if there is a valid mapping first, or we won't count this pixel
    Eigen::Vector4d pw;
    pw << pw0_[i](0), pw0_[i](1), pw0_[i](2), 1;
    Eigen::Vector4d p_c = T_cw1 * pw;

    //2d pixel position in current frame
    double u = fx_ * p_c(0,0) / p_c(2,0) + cx_;
    double v = fy_ * p_c(1,0) / p_c(2,0) + cy_;

    //bilinear interporlation of pixel. DSO getInterpolatedElement33() function, bilinear interpolation
    if(u >= 0 && u+3 <= image1_.cols && v >= 0 && v+3 <= image1_.rows){
      intensity_current_(i,0) = get_interpolated_pixel_value(u,v);
    }
    else{
      ob++;
      continue;
    }

    //current frame mutual information probability
    if(intensity0_(i,0) >= 255)
      intensity0_(i,0) = 254.999;
    if(intensity0_(i,0) < 0)
      intensity0_(i, 0) = 0.0;

    double bin_pos_ref =  intensity0_(i,0) * bin_num_/255.0;
    
    int bins_index_ref = floor(bin_pos_ref);

    if(bins_index_ref < 0 || bins_index_ref>bin_num_ -1)
        std::cout<<"the intensity is \n"<<intensity0_<<std::endl;

    pro_ref_[bins_index_ref] += 1.0;

  }

  for(int i = 0; i < bin_num_ ; i++)
    pro_ref_[i] /=  (intensity0_.rows()-ob);
  
  for(int i = 0; i < bin_num_ ; i++){
    if(pro_ref_[i] < sigma_)
      continue;
    H_ref_ -= pro_ref_[i] * log2(pro_ref_[i]);
  }
  
};


void NID::ComputeH(){
  int ob = 0;

  Eigen::Matrix4d T_cw1 = tf_.inverse();
  for(int i = 0 ; i<intensity0_.rows(); i++){

    Eigen::Vector4d pw;
    pw << pw0_[i](0), pw0_[i](1), pw0_[i](2), 1;
    Eigen::Vector4d p_c = T_cw1 * pw;

    if(intensity0_(i,0) >= 255)
      intensity0_(i,0) = 254.999;
    if(intensity0_(i,0) < 0)
      intensity0_(i, 0) = 0.0;

    double bin_pos_ref =  intensity0_(i,0)* bin_num_/255.0;
    int bins_index_ref = floor(bin_pos_ref);
    
    //2d pixel position in current frame
    double u = fx_ * p_c(0,0) / p_c(2,0) + cx_;
    double v = fy_ * p_c(1,0) / p_c(2,0) + cy_;

    //bilinear interporlation of pixel. DSO getInterpolatedElement33() function, bilinear interpolation
    if(u >= 0 && u+3 <= image1_.cols && v >= 0 && v+3 <= image1_.rows){
      intensity_current_(i,0) = get_interpolated_pixel_value(u,v);
    }
    else{
      ob++;
      continue;
    }

    if(intensity_current_(i,0) >= 255)
      intensity_current_(i,0) = 254.999;
    if(intensity_current_(i,0) < 0)
      intensity_current_(i, 0) = 0.0;

    double bin_pos_current = intensity_current_(i,0)* bin_num_/255.0;
    double pos_cubic_current = bin_pos_current * bin_pos_current * bin_pos_current;
    double pos_qua_current  = bin_pos_current * bin_pos_current;
    double bins_index_current = floor(bin_pos_current);

    pro_current_[bins_index_current] += 1.0;

    pro_joint_[bins_index_ref][bins_index_current] += 1.0;
    
    
  }


  //if too many points are out of image boundary, we don't use this edge
  if(intensity0_.rows() - ob < 300){
    //std::cout<<"no enough points in the image boundary"<<std::endl;
    return;
  }

  //pro_last_.size() = bin_num
  for(int i = 0; i < bin_num_ ; i++){
    pro_current_[i] /= (intensity0_.rows() - ob);
  }

  for(int i = 0; i < bin_num_; i++)
    for(int j = 0; j<bin_num_; j++){
      pro_joint_[i][j] /= (intensity0_.rows() - ob); 
    }
  
  for(int i = 0; i < bin_num_ ; i++){
    if(pro_current_[i] < sigma_)
      continue;
    H_current_ -= pro_current_[i] * log2(pro_current_[i]);
  }

  for(int i = 0; i < bin_num_; i++)
    for(int j = 0; j<bin_num_; j++){
      if(pro_joint_[i][j] < sigma_)
        continue;
      H_joint_ -= pro_joint_[i][j] * log2(pro_joint_[i][j]);
    }

    nid_ = (2*H_joint_ - H_ref_ - H_current_)/H_joint_;
    MI_ = H_ref_ + H_current_ - H_joint_;
    
    //if all points from one image is mapped into only one bin so that one of the pro_current_[i] == 1, then H_currrent and H_joint will be 0;
    //For example, if we have 3 bins, pro_ref is 0 1 0, pro_current is 0 1 0 (or 0 0 1, 1 0 0), then H_ref and current will be 0. Notice we can totally infer one image's state from another. For this perfect matching, we should count the cost as 0
    if(H_ref_ == 0.0 && H_current_ == 0.0 && H_joint_ == 0.0){
      MI_ = 0.0;
      nid_ = 0.0;
    }

    std::cout<<"Href, current, joint from standard method is "<<H_ref_<<","<<H_current_<<","<<H_joint_<<", MI "<<MI_<<", NID "<<nid_<<std::endl;

}




