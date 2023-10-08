# VISLAM

- Place the Visual-Inertial-SLAM_final folder inside the ECE276A_PR3(downloaded from the course website) directory.

- Place the data inside ECE276A_PR3/data 

To run the script run:
- First run `image_utils.py` to preprocess the stereo camera features to 3D coordinates in IMU frame(Used to initialize the landmarks when first seen). Has one argument `--dataset` which specifies whether the dataset is 3 or 10.
```
cd Visual-Inertial-SLAM_final
python image_utils.py --dataset 10    
# This should create 3 files
#for eg if dataset = 3 -> coordinates_03.npy, valid_landmark_indices_03.npy, visibility_matrix_03.npy
```

- For EKF Mapping, we run `main_mapping.py`
```
cd Visual-Inertial-SLAM_final
python image_utils.py --dataset 10 # If not already run
python main_mapping.py --dataset 10 --feats 200
```
There are two arguments
    - dataset - which dataset to run 3 or 10
    - feats - Number of feats to use, the features are evenly skipped such that the final feature count is this.

- For VI EKF SLAM, we run `main.py`
```
cd Visual-Inertial-SLAM_final
python image_utils.py --dataset 10 # If not already run
python main.py --dataset 10 --feats 200
```
There are two arguments
    - dataset - which dataset to run 3 or 10
    - feats - Number of feats to use, the features are evenly skipped such that the final feature count is this.

These codes should first show the Dead Reckoning plots and then show the final result as plots. It should also save them as images.