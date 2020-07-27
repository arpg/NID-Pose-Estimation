echo "Configuring and building g2o ..."

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
