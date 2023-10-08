import numpy as np
from pr3_utils import *
from ekf import ekf, ekf_landmark, combined_ekf
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Particle Filter')
parser.add_argument('--dataset', type=int,
                    help='Dataset to run on', default = 10)
parser.add_argument('--feats', type=int,
                    help='Number of Features to use', default = 200)



args = parser.parse_args()


if __name__ == '__main__':

    num_features = [args.feats]

    filenumber = args.dataset

    # imu_udown = np.array([[1, 0, 0, 0],
    #                       [0, np.cos(np.pi), -np.sin(np.pi), 0],
    #                       [0, np.sin(np.pi), np.cos(np.pi), 0],
    #                       [0, 0, 0, 1]])
    imu_udown = np.array([[1, 0, 0, 0],
                          [0, -1, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 0, 1]])
    
    rot_matrix = np.array([[1,0,0,0],
                           [0,-1,0,0],
                           [0,0,-1,0],
                           [0,0,0,1]])

    # Load the measurements
    filename = "../data/" + "{0:0=2d}".format(filenumber) + ".npz"
    t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)

    t = t - t[0][0]

    ekf_ins = ekf(np.zeros((6,6)), np.zeros((6,6)))

    # (a) IMU Localization via EKF Prediction
    imu_poses = []
    imu_poses.append(ekf_ins.mean_pose)
    for ind in range(len(t)-1):
        i = ind+1
        ekf_ins.predict(linangtou((linear_velocity[i]).reshape(1,-1), (angular_velocity[i]).reshape(1,-1)), t[i])
        imu_poses.append(ekf_ins.mean_pose)
    
    with open("DR_" + "{0:0=2d}".format(filenumber) + ".npy", "wb") as f:
        np.save(f, imu_poses)
    
    visualize_trajectory_2d_feats(np.transpose(np.array(imu_poses), (1,2,0)), None, fig_name = "DR_" + "{0:0=2d}".format(filenumber) + ".png" , path_name = "Dead Reckoning", show_ori = True)
    
    # (b) Landmark Mapping via EKF Update
    K_s = np.zeros((4,4))
    K_s[:2,:3] = K[:2,:3]
    K_s[2:,:3] = K[:2,:3]
    K_s[2,3] = -K[0,0]*b

    coordinates = np.load("coordinates_" + "{0:0=2d}".format(filenumber) + ".npy")
    visibility_matrix = np.load("visibility_matrix_" + "{0:0=2d}".format(filenumber) + ".npy")
    valid_landmark_indices = np.load("valid_landmark_indices_" + "{0:0=2d}".format(filenumber) + ".npy")
    visibility_matrix = visibility_matrix[:, valid_landmark_indices]
    features = features[:, valid_landmark_indices]

    # ekf_mapping = ekf_landmark(K_s, np.linalg.inv((imu_udown @ imu_T_cam)), np.eye(4)*4)

    # final_map = []

    # print("Coordinates", coordinates.shape)
    # print("Visibility Matrix", visibility_matrix.shape)

    # number_of_features_kept = 500
    # skip = features.shape[1] // number_of_features_kept

    # for i in tqdm(range(len(t))):
    #     ekf_mapping.compute_timestep(imu_poses, features[:, ::skip, :], coordinates[:, ::skip, :], visibility_matrix[:, ::skip], i)
    
    # final_map = ekf_mapping.mean_landmarks
    # print(final_map.shape)

    

    # with open("visual_mapping.npy", "wb") as f:
    #     np.save(f, final_map)

    # (c) Visual-Inertial SLAM

    # W = np.eye(6)
    # W[:3,:3] *= 0.2 # 0.1 # 0.2 # Best  # 0.1
    # W[3:,3:] *= 0.02 # 0.01 # 0.02 # Best

    for num in num_features: 
        W = np.eye(6)
        W[:3,:3] *= 0.2 # 0.1 # 0.2 # Best  # 0.1
        W[3:,3:] *= 0.02 # 0.01 # 0.02 # Best
        print("W", W)
        print(num)

        final_map = None
        ekf_vi = None

        # 0.1 # cov_landmark 10
        ekf_vi = combined_ekf(covariance_pose = np.eye(6)*0, covariance_landmark = np.eye(3)*10, K_s = K_s, o_T_i = np.linalg.inv((imu_udown @ imu_T_cam)), W =  W, V = np.eye(4)*25)

        number_of_features_kept = num
        skip = features.shape[1] // number_of_features_kept
        print("Skip", skip)

        # skip_after_2000 = features.shape[1] // 3000

        # indices_to_keep = []
        # indices_to_keep_after_2000 = []
        # for ind in tqdm(range(features.shape[1])):
        #     if ind%skip == 0:
        #         indices_to_keep.append(ind)
        # for ind in tqdm(range(features.shape[1])):
        #     if ind%skip == 0:
        #         indices_to_keep_after_2000.append(ind)
        #     if ind == 2000:
        #         skip = skip_after_2000

        imu_poses_vi = []
        imu_poses_vi.append(ekf_vi.mean_pose)
        print("Poses", imu_poses_vi)
        print("final_map", final_map)
        i=0
        # indices_to_keep = np.array(indices_to_keep).astype(np.int16)
        for ind in tqdm(range(len(t)-1)):
            i = ind+1
            ekf_vi.compute_timestep(features[:, ::skip, :], coordinates[:, ::skip, :], visibility_matrix[:, ::skip], linear_velocity, angular_velocity, t, i)
            # if ind<2000:
            #     ekf_vi.compute_timestep(features[:, indices_to_keep, :], coordinates[:, indices_to_keep, :], visibility_matrix[:, indices_to_keep], linear_velocity, angular_velocity, t, i)
            # else:
            #     ekf_vi.compute_timestep(features[:, indices_to_keep_after_2000, :], coordinates[:, indices_to_keep_after_2000, :], visibility_matrix[:, indices_to_keep_after_2000], linear_velocity, angular_velocity, t, i)
            imu_poses_vi.append(ekf_vi.mean_pose)
        
        final_map = ekf_vi.mean_landmarks

        with open("map_"+str(num) + "_" + "{0:0=2d}".format(filenumber) + ".npy", "wb") as f:
            np.save(f, final_map)
        
        with open("pose_"+str(num) + "_" + "{0:0=2d}".format(filenumber) + ".npy", "wb") as f:
            np.save(f, np.array(imu_poses_vi))

        # final_map = (rot_matrix @ final_map.T).T


        # You can use the function below to visualize the robot pose over time
        visualize_trajectory_2d_feats(np.transpose(np.array(imu_poses_vi), (1,2,0)), final_map, fig_name = "map_"+str(num) + "_" + "{0:0=2d}".format(filenumber) + ".png" ,path_name = "Trajectory "+ "{0:0=2d}".format(filenumber) + " Feats:" + str(num), show_ori = True)