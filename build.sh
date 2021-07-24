echo "Configuring and building g2o ..."

#Use Relase in g2o here will lead to a problem.
#const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0])
#std::cout<<"to be estimated matrix \n"<<v1->estimate().to_homogeneous_matrix()<<std::endl;
#will give a different matrix as the initialization in main
#vSE3->setEstimate(g2o::SE3Quat(r_cw1,t_cw1));

#Don't use Eigen3.1.0
#Compile error 
#static assertion failed: YOU_CALLED_A_FIXED_SIZE_METHOD_ON_A_DYNAMIC_SIZE_MATRIX_OR_VECTOR
#will show up
cd g2o
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Relase
make -j

cd ../../

echo "Configuring and building main_program ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Relase
make -j
