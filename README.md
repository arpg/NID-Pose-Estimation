This repo is an implementation of NID based pose estimation. Implementtion detial can be seen in paper `Robust Dense Visual SLAM Using Cell Based Normalized Information Distance`. If you use this work, please kindly consider citing
```
@inproceedings{chen2021robust,
  title={Robust Pose Estimation Based on Normalized Information Distance},
  author={Chen, Zhaozhong and Heckman, Christoffer},
  booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={2217--2223},
  year={2021},
  organization={IEEE}
}

```

### Requirement
Opencv >= 3.0  
Eigen >= 3.1
You have to use the G2O along with the code because NID jacobian estimation is written in the this G2O  
If you want to use GPU version, make sure you have CUDA > 8  
GPU architecture > `Pascal`  

Test on Ubuntu 18.04, Opecv 3 pass  
Test on Ubuntu 20.04 , Opencv 4 failed  
### Install
```
git clone https://github.com/arpg/NID-Pose-Estimation.git
cd NID-Pose-Estimation
./buid.sh
```

### Use 
Downlaod the dataset here https://cvg.ethz.ch/research/illumination-change-robust-dslam/  
In the `config_eth_cvg.yaml` you must specify where the dataset is first. Then can sepcify which two images you want to use. Finally you specify you want to run it on CPU or GPU. Then simply use the following to run the code. You need wait for a while to see the result if you run it on CPU. It should be around 100~300 times faster on different GPU.
```
cd bin
./NID_pose_estimation ../config_eth_cvg.yaml
```

In the code, a noise is added to the groundtruth, then you can see the original pose, pose with noise and pose estimated by NID.  
You may also be interested in seeing what's the property of NID estimation without B spline, simply run
```
./NID_standard_property ../config_eth_cvg.yaml
```
