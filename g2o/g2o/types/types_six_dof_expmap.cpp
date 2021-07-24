// g2o - General Graph Optimization
// Copyright (C) 2011 H. Strasdat
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "types_six_dof_expmap.h"

#include "../core/factory.h"
#include "../stuff/macros.h"

namespace g2o {

using namespace std;


Vector2d project2d(const Vector3d& v)  {
  Vector2d res;
  res(0) = v(0)/v(2);
  res(1) = v(1)/v(2);
  return res;
}

Vector3d unproject2d(const Vector2d& v)  {
  Vector3d res;
  res(0) = v(0);
  res(1) = v(1);
  res(2) = 1;
  return res;
}

VertexSE3Expmap::VertexSE3Expmap() : BaseVertex<6, SE3Quat>() {
}

bool VertexSE3Expmap::read(std::istream& is) {
  Vector7d est;
  for (int i=0; i<7; i++)
    is  >> est[i];
  SE3Quat cam2world;
  cam2world.fromVector(est);
  setEstimate(cam2world.inverse());
  return true;
}

bool VertexSE3Expmap::write(std::ostream& os) const {
  SE3Quat cam2world(estimate().inverse());
  for (int i=0; i<7; i++)
    os << cam2world[i] << " ";
  return os.good();
}


EdgeSE3ProjectXYZ::EdgeSE3ProjectXYZ() : BaseBinaryEdge<2, Vector2d, VertexSBAPointXYZ, VertexSE3Expmap>() {
}

bool EdgeSE3ProjectXYZ::read(std::istream& is){
  for (int i=0; i<2; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeSE3ProjectXYZ::write(std::ostream& os) const {

  for (int i=0; i<2; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}


void EdgeSE3ProjectXYZ::linearizeOplus() {
  VertexSE3Expmap * vj = static_cast<VertexSE3Expmap *>(_vertices[1]);
  SE3Quat T(vj->estimate());
  VertexSBAPointXYZ* vi = static_cast<VertexSBAPointXYZ*>(_vertices[0]);
  Vector3d xyz = vi->estimate();
  Vector3d xyz_trans = T.map(xyz);

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double z = xyz_trans[2];
  double z_2 = z*z;

  Matrix<double,2,3> tmp;
  tmp(0,0) = fx;
  tmp(0,1) = 0;
  tmp(0,2) = -x/z*fx;

  tmp(1,0) = 0;
  tmp(1,1) = fy;
  tmp(1,2) = -y/z*fy;

  _jacobianOplusXi =  -1./z * tmp * T.rotation().toRotationMatrix();

  _jacobianOplusXj(0,0) =  x*y/z_2 *fx;
  _jacobianOplusXj(0,1) = -(1+(x*x/z_2)) *fx;
  _jacobianOplusXj(0,2) = y/z *fx;
  _jacobianOplusXj(0,3) = -1./z *fx;
  _jacobianOplusXj(0,4) = 0;
  _jacobianOplusXj(0,5) = x/z_2 *fx;

  _jacobianOplusXj(1,0) = (1+y*y/z_2) *fy;
  _jacobianOplusXj(1,1) = -x*y/z_2 *fy;
  _jacobianOplusXj(1,2) = -x/z *fy;
  _jacobianOplusXj(1,3) = 0;
  _jacobianOplusXj(1,4) = -1./z *fy;
  _jacobianOplusXj(1,5) = y/z_2 *fy;
}

Vector2d EdgeSE3ProjectXYZ::cam_project(const Vector3d & trans_xyz) const{
  Vector2d proj = project2d(trans_xyz);
  Vector2d res;
  res[0] = proj[0]*fx + cx;
  res[1] = proj[1]*fy + cy;
  return res;
}


Vector3d EdgeStereoSE3ProjectXYZ::cam_project(const Vector3d & trans_xyz, const double &bf) const{
  const double invz = 1.0f/trans_xyz[2];
  Vector3d res;
  res[0] = trans_xyz[0]*invz*fx + cx;
  res[1] = trans_xyz[1]*invz*fy + cy;
  res[2] = res[0] - bf*invz;
  return res;
}

EdgeStereoSE3ProjectXYZ::EdgeStereoSE3ProjectXYZ() : BaseBinaryEdge<3, Vector3d, VertexSBAPointXYZ, VertexSE3Expmap>() {
}

bool EdgeStereoSE3ProjectXYZ::read(std::istream& is){
  for (int i=0; i<=3; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<=2; i++)
    for (int j=i; j<=2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeStereoSE3ProjectXYZ::write(std::ostream& os) const {

  for (int i=0; i<=3; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<=2; i++)
    for (int j=i; j<=2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}

void EdgeStereoSE3ProjectXYZ::linearizeOplus() {
  VertexSE3Expmap * vj = static_cast<VertexSE3Expmap *>(_vertices[1]);
  SE3Quat T(vj->estimate());
  VertexSBAPointXYZ* vi = static_cast<VertexSBAPointXYZ*>(_vertices[0]);
  Vector3d xyz = vi->estimate();
  Vector3d xyz_trans = T.map(xyz);

  const Matrix3d R =  T.rotation().toRotationMatrix();

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double z = xyz_trans[2];
  double z_2 = z*z;

  _jacobianOplusXi(0,0) = -fx*R(0,0)/z+fx*x*R(2,0)/z_2;
  _jacobianOplusXi(0,1) = -fx*R(0,1)/z+fx*x*R(2,1)/z_2;
  _jacobianOplusXi(0,2) = -fx*R(0,2)/z+fx*x*R(2,2)/z_2;

  _jacobianOplusXi(1,0) = -fy*R(1,0)/z+fy*y*R(2,0)/z_2;
  _jacobianOplusXi(1,1) = -fy*R(1,1)/z+fy*y*R(2,1)/z_2;
  _jacobianOplusXi(1,2) = -fy*R(1,2)/z+fy*y*R(2,2)/z_2;

  _jacobianOplusXi(2,0) = _jacobianOplusXi(0,0)-bf*R(2,0)/z_2;
  _jacobianOplusXi(2,1) = _jacobianOplusXi(0,1)-bf*R(2,1)/z_2;
  _jacobianOplusXi(2,2) = _jacobianOplusXi(0,2)-bf*R(2,2)/z_2;

  _jacobianOplusXj(0,0) =  x*y/z_2 *fx;
  _jacobianOplusXj(0,1) = -(1+(x*x/z_2)) *fx;
  _jacobianOplusXj(0,2) = y/z *fx;
  _jacobianOplusXj(0,3) = -1./z *fx;
  _jacobianOplusXj(0,4) = 0;
  _jacobianOplusXj(0,5) = x/z_2 *fx;

  _jacobianOplusXj(1,0) = (1+y*y/z_2) *fy;
  _jacobianOplusXj(1,1) = -x*y/z_2 *fy;
  _jacobianOplusXj(1,2) = -x/z *fy;
  _jacobianOplusXj(1,3) = 0;
  _jacobianOplusXj(1,4) = -1./z *fy;
  _jacobianOplusXj(1,5) = y/z_2 *fy;

  _jacobianOplusXj(2,0) = _jacobianOplusXj(0,0)-bf*y/z_2;
  _jacobianOplusXj(2,1) = _jacobianOplusXj(0,1)+bf*x/z_2;
  _jacobianOplusXj(2,2) = _jacobianOplusXj(0,2);
  _jacobianOplusXj(2,3) = _jacobianOplusXj(0,3);
  _jacobianOplusXj(2,4) = 0;
  _jacobianOplusXj(2,5) = _jacobianOplusXj(0,5)-bf/z_2;
}


//Only Pose

bool EdgeSE3ProjectXYZOnlyPose::read(std::istream& is){
  for (int i=0; i<2; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeSE3ProjectXYZOnlyPose::write(std::ostream& os) const {

  for (int i=0; i<2; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}


void EdgeSE3ProjectXYZOnlyPose::linearizeOplus() {
  VertexSE3Expmap * vi = static_cast<VertexSE3Expmap *>(_vertices[0]);
  Vector3d xyz_trans = vi->estimate().map(Xw);

  //std::cout<<"to be estimated matrix \n"<<vi->estimate().to_homogeneous_matrix()<<std::endl;

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double invz = 1.0/xyz_trans[2];
  double invz_2 = invz*invz;

  _jacobianOplusXi(0,0) =  x*y*invz_2 *fx;
  _jacobianOplusXi(0,1) = -(1+(x*x*invz_2)) *fx;
  _jacobianOplusXi(0,2) = y*invz *fx;
  _jacobianOplusXi(0,3) = -invz *fx;
  _jacobianOplusXi(0,4) = 0;
  _jacobianOplusXi(0,5) = x*invz_2 *fx;

  _jacobianOplusXi(1,0) = (1+y*y*invz_2) *fy;
  _jacobianOplusXi(1,1) = -x*y*invz_2 *fy;
  _jacobianOplusXi(1,2) = -x*invz *fy;
  _jacobianOplusXi(1,3) = 0;
  _jacobianOplusXi(1,4) = -invz *fy;
  _jacobianOplusXi(1,5) = y*invz_2 *fy;

  //std::cout<<"the jacobian value, x,y,z only pose is \n"<<_jacobianOplusXi<<std::endl;
}

Vector2d EdgeSE3ProjectXYZOnlyPose::cam_project(const Vector3d & trans_xyz) const{
  Vector2d proj = project2d(trans_xyz);
  Vector2d res;
  res[0] = proj[0]*fx + cx;
  res[1] = proj[1]*fy + cy;
  return res;
}


Vector3d EdgeStereoSE3ProjectXYZOnlyPose::cam_project(const Vector3d & trans_xyz) const{
  const double invz = 1.0f/trans_xyz[2];
  Vector3d res;
  res[0] = trans_xyz[0]*invz*fx + cx;
  res[1] = trans_xyz[1]*invz*fy + cy;
  res[2] = res[0] - bf*invz;
  return res;
}


bool EdgeStereoSE3ProjectXYZOnlyPose::read(std::istream& is){
  for (int i=0; i<=3; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<=2; i++)
    for (int j=i; j<=2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeStereoSE3ProjectXYZOnlyPose::write(std::ostream& os) const {

  for (int i=0; i<=3; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<=2; i++)
    for (int j=i; j<=2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}

void EdgeStereoSE3ProjectXYZOnlyPose::linearizeOplus() {
  VertexSE3Expmap * vi = static_cast<VertexSE3Expmap *>(_vertices[0]);
  Vector3d xyz_trans = vi->estimate().map(Xw);

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double invz = 1.0/xyz_trans[2];
  double invz_2 = invz*invz;

  _jacobianOplusXi(0,0) =  x*y*invz_2 *fx;
  _jacobianOplusXi(0,1) = -(1+(x*x*invz_2)) *fx;
  _jacobianOplusXi(0,2) = y*invz *fx;
  _jacobianOplusXi(0,3) = -invz *fx;
  _jacobianOplusXi(0,4) = 0;
  _jacobianOplusXi(0,5) = x*invz_2 *fx;

  _jacobianOplusXi(1,0) = (1+y*y*invz_2) *fy;
  _jacobianOplusXi(1,1) = -x*y*invz_2 *fy;
  _jacobianOplusXi(1,2) = -x*invz *fy;
  _jacobianOplusXi(1,3) = 0;
  _jacobianOplusXi(1,4) = -invz *fy;
  _jacobianOplusXi(1,5) = y*invz_2 *fy;

  _jacobianOplusXi(2,0) = _jacobianOplusXi(0,0)-bf*y*invz_2;
  _jacobianOplusXi(2,1) = _jacobianOplusXi(0,1)+bf*x*invz_2;
  _jacobianOplusXi(2,2) = _jacobianOplusXi(0,2);
  _jacobianOplusXi(2,3) = _jacobianOplusXi(0,3);
  _jacobianOplusXi(2,4) = 0;
  _jacobianOplusXi(2,5) = _jacobianOplusXi(0,5)-bf*invz_2;
}

/***Add by zhaozhong chen**/
bool EdgeSE3ProjectIntensityOnlyPoseNID::read(std::istream& is){
  return true;
}

bool EdgeSE3ProjectIntensityOnlyPoseNID::write(std::ostream& os) const {

  return os.good();
}


void EdgeSE3ProjectIntensityOnlyPoseNID::linearizeOplus() {

  if(use_CPU_){
    VertexSE3Expmap * vi = static_cast<VertexSE3Expmap *>(_vertices[0]);
    Eigen::VectorXd obs(_measurement);
    
    std::vector<std::vector<double> > d_sum_bs_pose(bin_num_, std::vector<double>(6, 0.0));

    //bin_num * bin_num * 6 dimension
    std::vector<std::vector<std::vector<double> > > d_sum_joint_bs_pose(bin_num_, std::vector<std::vector<double> >(bin_num_, std::vector<double>(6,0.0)));

    //derivative, mappde intensity to intensity to 
    double d_mi_i = (bin_num_- bs_degree_)/255.0;

    for(int i = 0 ; i<_measurement.rows(); i++){

      Eigen::Vector3d p_c = vi->estimate().map ( x_world_set_[i] );

      if(obs(i,0) >= 255)
        obs(i,0) = 254.999;
      if(obs(i,0) < 0)
        obs(i, 0) = 0.0;
      
      double bin_pos_ref =  obs(i,0)* (bin_num_- bs_degree_)/255.0;
      int bins_index_ref = floor(bin_pos_ref);

      double u_c =  p_c(0,0)/p_c(2,0);
      double v_c =  p_c(1,0)/p_c(2,0);

      double x = p_c(0,0);
      double y = p_c(1,0);
      double invz = 1.0/p_c(2,0);
      double invz_2 = invz*invz;

      // jacobian from se3 to u,v
      // NOTE that in g2o the Lie algebra is (\omega, \epsilon), where \omega is so(3) and \epsilon the translation
      Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;


      //2d pixel position in l frame
      double u = fx_ * u_c + cx_;
      double v = fy_ * v_c + cy_;

      double bin_pos_current =  intensity_current_(i,0)* (bin_num_-3.0)/255.0;
      double pos_cubic_current = bin_pos_current * bin_pos_current * bin_pos_current;
      double pos_qua_current  = bin_pos_current * bin_pos_current;
      int bins_index_current = floor(bin_pos_current);

      double gradient_x, gradient_y;

      Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;
      //calculate the four corner pixel's gradient and do bilinear inerpolation
      if(u >= 0 && u+3 <= image1_.cols - 1 && v >= 0 && v+3 <= image1_.rows){
        jacobian_pixel_uv ( 0,0 ) = ( get_interpolated_pixel_value ( u+1,v )-get_interpolated_pixel_value ( u-1,v ) ) /2;
        jacobian_pixel_uv ( 0,1 ) = ( get_interpolated_pixel_value ( u,v+1 )-get_interpolated_pixel_value ( u,v-1 ) ) /2;
        //std::cout<<"gradient x and y "<<gradient_x<<", "<<gradient_y<<std::endl;

        jacobian_uv_ksai ( 0,0 ) = - x*y*invz_2 *fx_;
        jacobian_uv_ksai ( 0,1 ) = ( 1+ ( x*x*invz_2 ) ) *fx_;
        jacobian_uv_ksai ( 0,2 ) = - y*invz *fx_;
        jacobian_uv_ksai ( 0,3 ) = invz *fx_;
        jacobian_uv_ksai ( 0,4 ) = 0;
        jacobian_uv_ksai ( 0,5 ) = -x*invz_2 *fx_;

        jacobian_uv_ksai ( 1,0 ) = - ( 1+y*y*invz_2 ) *fy_;
        jacobian_uv_ksai ( 1,1 ) = x*y*invz_2 *fy_;
        jacobian_uv_ksai ( 1,2 ) = x*invz *fy_;
        jacobian_uv_ksai ( 1,3 ) = 0;
        jacobian_uv_ksai ( 1,4 ) = invz *fy_;
        jacobian_uv_ksai ( 1,5 ) = -y*invz_2 *fy_;
      }
      else{
        jacobian_pixel_uv ( 0,0 ) = 0.0;
        jacobian_pixel_uv ( 0,1 ) = 0.0;
        jacobian_uv_ksai = Eigen::Matrix<double, 2, 6>::Zero();
        continue;
      }
      
      Eigen::Matrix<double, 1, 6> d_i_pose;
      d_i_pose = jacobian_pixel_uv * jacobian_uv_ksai;


      std::vector<double> d_bs_mi(bin_num_, 0.0);

      //Checked when bin position in (0,1) (2,3) (3,4), all correct
      //std::cout<<"measurement is "<<hg_intensity_last_(i,0)<<", bin position is "<<bin_pos_last<<std::endl;
      d_bs_mi[bins_index_current]     = BsplineDer(bins_index_current, bs_degree_+1, bin_pos_current);//bins_index_current
      d_bs_mi[bins_index_current + 1] = BsplineDer(bins_index_current+1, bs_degree_+1, bin_pos_current);
      d_bs_mi[bins_index_current + 2] = BsplineDer(bins_index_current+2, bs_degree_+1, bin_pos_current);
      d_bs_mi[bins_index_current + 3] = BsplineDer(bins_index_current+3, bs_degree_+1, bin_pos_current);


      for(int m = 0; m<bs_degree_+1; m++)
        for(int n = 0; n<6; n++){
          d_sum_bs_pose[bins_index_current + m][n] += d_bs_mi[bins_index_current + m] * d_mi_i * d_i_pose(0,n);
        }

      for(int k = 0; k<bs_degree_+1; k++)
        for(int m = 0; m<bs_degree_+1; m++)
          for(int n = 0; n<6; n++){
            d_sum_joint_bs_pose[bins_index_ref + k][bins_index_current + m][n] += bs_value_ref_(i,k) * d_bs_mi[bins_index_current + m] * d_mi_i * d_i_pose[n];
          }
    }

    
    for(int m = 0; m<bin_num_; m++)
      for(int n = 0; n<6; n++){
        d_sum_bs_pose[m][n] /= (_measurement.rows() - ob);
    }

    for(int m = 0; m<bin_num_; m++)
      for(int n = 0; n<bin_num_; n++)
        for(int k = 0; k<6; k++){
          d_sum_joint_bs_pose[m][n][k] /= (_measurement.rows() - ob);
        }

    //derivative H_joint to pose
    std::vector<double> d_hj_p(6, 0.0);
    for(int i = 0; i < 6; i++){
      double tmp = 0.0;
      for(int m = 0; m<bin_num_; m++){
        for(int n = 0; n<bin_num_; n++){
          if(pro_joint_[m][n] < sigma_)
            continue;
          tmp -= (1.0 + log2(pro_joint_[m][n])) * d_sum_joint_bs_pose[m][n][i];
        }
      }
      d_hj_p[i] = tmp;
    }  

    //derivative H_last_ to pose
    std::vector<double> d_hl_p(6, 0.0);
    for(int i = 0; i < 6; i++){
      for(int j = 0; j < bin_num_; j++){
        if(pro_current_[j] < sigma_)
          continue;
        d_hl_p[i] -= (1.0 + log2(pro_current_[j])) * d_sum_bs_pose[j][i];
      }
    }

    double inv_square_hj = 1.0/(H_joint_ * H_joint_);
    
    _jacobianOplusXi(0,0) = (d_hj_p[0] * (H_current_ + H_ref_) - d_hl_p[0] * H_joint_) * inv_square_hj;
    _jacobianOplusXi(0,1) = (d_hj_p[1] * (H_current_ + H_ref_) - d_hl_p[1] * H_joint_) * inv_square_hj;
    _jacobianOplusXi(0,2) = (d_hj_p[2] * (H_current_ + H_ref_) - d_hl_p[2] * H_joint_) * inv_square_hj; 
    _jacobianOplusXi(0,3) = (d_hj_p[3] * (H_current_ + H_ref_) - d_hl_p[3] * H_joint_) * inv_square_hj; 
    _jacobianOplusXi(0,4) = (d_hj_p[4] * (H_current_ + H_ref_) - d_hl_p[4] * H_joint_) * inv_square_hj; 
    _jacobianOplusXi(0,5) = (d_hj_p[5] * (H_current_ + H_ref_) - d_hl_p[5] * H_joint_) * inv_square_hj;
  }
  else{
    //GPU has already calculated the required value
    _jacobianOplusXi(0,0) = j0_;
    _jacobianOplusXi(0,1) = j1_;
    _jacobianOplusXi(0,2) = j2_;
    _jacobianOplusXi(0,3) = j3_;
    _jacobianOplusXi(0,4) = j4_;
    _jacobianOplusXi(0,5) = j5_;
  }
  

}


void EdgeSE3ProjectIntensityOnlyPoseNID::ComputeH(){

  const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
  Eigen::VectorXd obs(_measurement);

  for(int i = 0 ; i<obs.rows(); i++){
    //current frame mutual information probabilit
    Eigen::Vector3d p_c = v1->estimate().map ( x_world_set_[i] );

    if(obs(i,0) >= 255)
      obs(i,0) = 254.999;
    if(obs(i,0) < 0)
      obs(i, 0) = 0.0;

    double bin_pos_ref =  obs(i,0)* (bin_num_- bs_degree_)/255.0;
    int bins_index_ref = floor(bin_pos_ref);
    
    //2d pixel position in current frame
    double u = fx_ * p_c(0,0) / p_c(2,0) + cx_;
    double v = fy_ * p_c(1,0) / p_c(2,0) + cy_;

    if(u >= 0 && u+3 <= image1_.cols && v >= 0 && v+3 <= image1_.rows){
      intensity_current_(i,0) = get_interpolated_pixel_value(u,v);
    }
    else{
      continue;
    }

    if(intensity_current_(i,0) >= 255)
      intensity_current_(i,0) = 254.999;
    if(intensity_current_(i,0) < 0)
      intensity_current_(i, 0) = 0.0;


    //last frame mutual information probability
    double bin_pos_current = intensity_current_(i,0)* (bin_num_-3.0)/255.0;
    double pos_cubic_current = bin_pos_current * bin_pos_current * bin_pos_current;
    double pos_qua_current  = bin_pos_current * bin_pos_current;
    double bins_index_current = floor(bin_pos_current);

    double bs_value0_current = Bspline(bins_index_current, bs_degree_+1, bin_pos_current);
    bs_value_current_(i, 0 ) = bs_value0_current;
    double bs_value1_current = Bspline(bins_index_current+1, bs_degree_+1, bin_pos_current);
    bs_value_current_(i, 1 ) = bs_value1_current;
    double bs_value2_current = Bspline(bins_index_current+2, bs_degree_+1, bin_pos_current);
    bs_value_current_(i, 2) = bs_value2_current;
    double bs_value3_current = Bspline(bins_index_current+3, bs_degree_+1, bin_pos_current);
    bs_value_current_(i, 3) = bs_value3_current;

    pro_current_[bins_index_current]   += bs_value0_current;
    pro_current_[bins_index_current+1] += bs_value1_current;
    pro_current_[bins_index_current+2] += bs_value2_current;
    pro_current_[bins_index_current+3] += bs_value3_current;
    
    //(bs_degree + 1) * (bs_degree + 1) combination
    for(int m = 0 ; m<bs_degree_+1; m++)
      for(int n = 0; n<bs_degree_+1; n++){
        pro_joint_[bins_index_ref+m][bins_index_current+n] += bs_value_ref_(i,m)*bs_value_current_(i,n);
      }
    
  }

  //std::cout<<"final ob is "<<ob<<std::endl;

  //if too many points are out of image boundary, we don't use this edge
  if(obs.rows() - ob < 300){
    this->setLevel(1);
    return;
  }
  
  //pro_last_.size() = bin_num
  for(int i = 0; i < bin_num_ ; i++){
    pro_current_[i] /= (obs.rows() - ob);
  }

  for(int i = 0; i < bin_num_; i++)
    for(int j = 0; j<bin_num_; j++){
      pro_joint_[i][j] /= (obs.rows() - ob); 
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

}

void EdgeSE3ProjectIntensityOnlyPoseNID::set_bspline_relates(int bs_degree, int bin_num){
  bs_degree_ = bs_degree;
  bin_num_ = bin_num;

  pro_current_ = std::vector<double>(bin_num_,0.0);
  pro_ref_ = std::vector<double>(bin_num_,0.0);
  pro_joint_ = std::vector<std::vector<double> >(bin_num_);
  for(int i = 0; i < bin_num_; i++){
    pro_joint_[i] = std::vector<double>(bin_num_, 0.0);
  }

  bs_value_ref_ = Eigen::MatrixXd::Zero(_measurement.rows(),4);
  bs_value_current_ = Eigen::MatrixXd::Zero(_measurement.rows(),4);
  intensity_current_ = Eigen::VectorXd::Zero(_measurement.rows());
};

void EdgeSE3ProjectIntensityOnlyPoseNID::computeHref(){
  const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);

  Eigen::VectorXd obs(_measurement);

  ob = 0;
  
  for(int i = 0 ; i<obs.rows(); i++){

    Eigen::Vector3d p_c = v1->estimate().map ( x_world_set_[i] );


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
    if(obs(i,0) >= 255)
      obs(i,0) = 254.999;
    if(obs(i,0) < 0)
      obs(i, 0) = 0.0;

    double bin_pos_ref =  obs(i,0) * (bin_num_- bs_degree_)/255.0;
    double pos_cubic_ref = bin_pos_ref * bin_pos_ref * bin_pos_ref;
    double pos_qua_ref  = bin_pos_ref * bin_pos_ref;
    int bins_index_ref = floor(bin_pos_ref);

    double bs_value0_ref = Bspline(bins_index_ref, bs_degree_+1, bin_pos_ref);
    bs_value_ref_(i,0) = bs_value0_ref;
    double bs_value1_ref = Bspline(bins_index_ref+1, bs_degree_+1, bin_pos_ref);
    bs_value_ref_(i,1) = bs_value1_ref;
    double bs_value2_ref = Bspline(bins_index_ref+2, bs_degree_+1, bin_pos_ref);
    bs_value_ref_(i,2) = bs_value2_ref;
    double bs_value3_ref = Bspline(bins_index_ref+3, bs_degree_+1, bin_pos_ref);
    bs_value_ref_(i,3) = bs_value3_ref;

    if(bins_index_ref < 0 || bins_index_ref>bin_num_ -1)
        std::cout<<"the intensity is \n"<<obs(i,0)<<std::endl;

    pro_ref_[bins_index_ref]   += bs_value0_ref;
    pro_ref_[bins_index_ref+1] += bs_value1_ref;
    pro_ref_[bins_index_ref+2] += bs_value2_ref;
    pro_ref_[bins_index_ref+3] += bs_value3_ref;

  }

  if(obs.rows() - ob < 300){
    this->setLevel(1);
    return;
  }

  for(int i = 0; i < bin_num_ ; i++){
    pro_ref_[i] /=  (obs.rows() - ob);
  }
  
  for(int i = 0; i < bin_num_ ; i++){
    if(pro_ref_[i] < sigma_)
      continue;
    H_ref_ -= pro_ref_[i] * log2(pro_ref_[i]);
  }
  
};

void EdgeSE3ProjectIntensityOnlyPoseNID::ClearPrevH(){
  pro_current_ = std::vector<double>(bin_num_,0.0);
  pro_ref_ = std::vector<double>(bin_num_,0.0);
  pro_joint_ = std::vector<std::vector<double> >(bin_num_);
  for(int i = 0; i < bin_num_; i++){
    pro_joint_[i] = std::vector<double>(bin_num_, 0.0);
  }
  H_joint_ = 0.0;
  H_current_ = 0.0;  
};

double EdgeSE3ProjectIntensityOnlyPoseNID::Bspline(int index, int order, double u){
  double coef1, coef2;
  if ( order == 1 )
  {
    if ( index == 0 ) if ( ( knots_[index] <= u ) && ( u <= knots_[index+1] ) ) return 1.0;
    if ( ( knots_[index] < u ) && ( u <= knots_[index+1] ) ) return 1.0;
    else return 0.0;
  }
  else
  {
    if ( knots_[index + order - 1] == knots_[index] ) 
    {
        if ( u == knots_[index] ) coef1 = 1;
        else coef1 = 0;
    }
    else coef1 = (u - knots_[index])/(knots_[index + order - 1] - knots_[index]);

    if ( knots_[index + order] == knots_[index+1] )
    {
        if ( u == knots_[index + order] ) coef2 = 1;
        else coef2 = 0;
    }
    else coef2 = (knots_[index + order] - u)/(knots_[index + order] - knots_[index+1]);
    
    return ( coef1 * Bspline(index, order-1, u) + coef2 * Bspline(index+1,order-1 ,u) );
  }
};

double EdgeSE3ProjectIntensityOnlyPoseNID::BsplineDer(int index, int order, double u){
  double coef1, coef2, coef3, coef4;
  if ( order == 1 )
  {
    return 0.0;
  }
  else
  {
    if ( knots_[index + order - 1] == knots_[index] ) 
    {
      if ( u == knots_[index] ) coef1 = 1;
      else coef1 = 0;

      coef3 = 0.0;
    }
    else {
      coef1 = (u - knots_[index])/(knots_[index + order - 1] - knots_[index]);
      coef3 = 1.0 / (knots_[index + order - 1] - knots_[index]);
    }

    if ( knots_[index + order] == knots_[index+1] )
    {
      if ( u == knots_[index + order] ) coef2 = 1;
      else coef2 = 0;

      coef4 = 0.0;
    }
    else {
      coef2 = (knots_[index + order] - u)/(knots_[index + order] - knots_[index+1]);
      coef4 = -1.0/(knots_[index + order] - knots_[index+1]);
    }
    
    return ( coef1 * BsplineDer(index, order-1, u) + coef2 * BsplineDer(index+1, order-1 ,u) + coef3 * Bspline(index, order-1, u) + coef4 * Bspline(index+1, order-1 ,u) );
  }
};

bool EdgeSE3ProjectIntensityOnlyPoseNID::setHref(int cell_size, int cell_row_id, int cell_col_id, int rows, int cols, double* bs_value, int* bin_index){
    int row_block = rows / cell_size;
    int col_block = cols / cell_size;
    int row_start = row_block * cell_row_id;
    int col_start = col_block * cell_col_id;
    int row_end = row_block * (cell_row_id + 1);
    int col_end = col_block * (cell_col_id + 1);

    int counter = 0;

    pro_ref_ = std::vector<double>(bin_num_,0.0);
    H_ref_ = 0.0;
    
    for(int i = row_start; i < row_end; i++)
        for(int j = col_start; j < col_end; j++){
          int id = i * cols + j;
          int bi = bin_index[id];
          if(!isnan(bs_value[4*id])){
            counter++;
            pro_ref_[bi] += bs_value[4*id];
            pro_ref_[bi+1] += bs_value[4*id+1];
            pro_ref_[bi+2] += bs_value[4*id+2];
            pro_ref_[bi+3] += bs_value[4*id+3];
          }
        }

    if(counter < 300){
      this->setLevel(1);
      return false;
    }

    for(int i = 0; i < bin_num_ ; i++){
      pro_ref_[i] /= counter;
    }
  
    for(int i = 0; i < bin_num_ ; i++){
      if(pro_ref_[i] < sigma_)
        continue;
      H_ref_ -= pro_ref_[i] * log2(pro_ref_[i]);
    }
};

void EdgeSE3ProjectDirectOnlyPose::linearizeOplus(){
    if ( level() == 1 )
    {
        _jacobianOplusXi = Eigen::Matrix<double, 1, 6>::Zero();
        return;
    }
    VertexSE3Expmap* vtx = static_cast<VertexSE3Expmap*> ( _vertices[0] );
    Eigen::Vector3d xyz_trans = vtx->estimate().map ( x_world_ );   // q in book

    double x = xyz_trans[0];
    double y = xyz_trans[1];
    double invz = 1.0/xyz_trans[2];
    double invz_2 = invz*invz;

    float u = x*fx_*invz + cx_;
    float v = y*fy_*invz + cy_;

    // jacobian from se3 to u,v
    // NOTE that in g2o the Lie algebra is (\omega, \epsilon), where \omega is so(3) and \epsilon the translation
    Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;

    jacobian_uv_ksai ( 0,0 ) = - x*y*invz_2 *fx_;
    jacobian_uv_ksai ( 0,1 ) = ( 1+ ( x*x*invz_2 ) ) *fx_;
    jacobian_uv_ksai ( 0,2 ) = - y*invz *fx_;
    jacobian_uv_ksai ( 0,3 ) = invz *fx_;
    jacobian_uv_ksai ( 0,4 ) = 0;
    jacobian_uv_ksai ( 0,5 ) = -x*invz_2 *fx_;

    jacobian_uv_ksai ( 1,0 ) = - ( 1+y*y*invz_2 ) *fy_;
    jacobian_uv_ksai ( 1,1 ) = x*y*invz_2 *fy_;
    jacobian_uv_ksai ( 1,2 ) = x*invz *fy_;
    jacobian_uv_ksai ( 1,3 ) = 0;
    jacobian_uv_ksai ( 1,4 ) = invz *fy_;
    jacobian_uv_ksai ( 1,5 ) = -y*invz_2 *fy_;

    Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;

    jacobian_pixel_uv ( 0,0 ) = ( get_interpolated_pixel_value ( u+1,v )-get_interpolated_pixel_value ( u-1,v ) ) /2;
    jacobian_pixel_uv ( 0,1 ) = ( get_interpolated_pixel_value ( u,v+1 )-get_interpolated_pixel_value ( u,v-1 ) ) /2;

    _jacobianOplusXi = jacobian_pixel_uv*jacobian_uv_ksai;  
};

} // end namespace
