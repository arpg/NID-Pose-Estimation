This repo is an implementation of NID based pose estimation. Implementtion detial can be seen in paper `Robust Dense Visual SLAM Using Cell Based Normalized Information Distance`. If you use this work, please kindly consider citing
```
TO BE ADDED
```

### Requirement
Opencv > 3.0  
Eigen > 3.3  
You have to use the G2O along with the code because NID jacobian estimation is written in the this G2O 

### Install
```
git clone https://github.com/arpg/NID-Pose-Estimation.git
cd NID-Pose-Estimation
./buid.sh
```

### Use 
Downlaod the dataset here https://cvg.ethz.ch/research/illumination-change-robust-dslam/  
In the `config_eth_cvg.yaml` you must specify where the dataset is first. Then can sepcify which two images you want to use. Then simply use the following to run the code. You need wait for a while to see the result because currently it is implemented on CPU.
```
cd bin
./NID_pose_estimation ../config_eth_cvg.yaml
```
In the code, a noise is added to the groundtruth, then you can see the original pose, pose with noise and pose estimated by NID.  
You may also be interested in seeing what's the property of NID estimation without B spline, simply run
```
./NID_standard_property ../config_eth_cvg.yaml
```
