import numpy as np
import matplotlib.pyplot as plt
from pr3_utils import *
import argparse

parser = argparse.ArgumentParser(description='Particle Filter')
parser.add_argument('--dataset', type=int,
                    help='Dataset to run on', default = 10)



args = parser.parse_args()


if __name__ == '__main__':

    # imu_udown = np.array([[1, 0, 0, 0],
    #                       [0, np.cos(np.pi), -np.sin(np.pi), 0],
    #                       [0, np.sin(np.pi), np.cos(np.pi), 0],
    #                       [0, 0, 0, 1]])
    imu_udown = np.array([[1, 0, 0, 0],
                          [0, -1, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 0, 1]])
    filenumber = args.dataset
    filename = "../data/" + "{0:0=2d}".format(filenumber) + ".npz"
    t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
    imu_T_cam =  imu_udown @ imu_T_cam
    disparity = features[:, :, 0] - features[:, :, 2]
    valid_landmark_indices = np.ones((disparity.shape[1]))
    visibility_matrix = np.ones((disparity.shape[0], disparity.shape[1]))

    for i in range(disparity.shape[0]):
        for j in range(disparity.shape[1]):
            current_landmark_features = features[i,j]
            if current_landmark_features[0] == -1:
                visibility_matrix[i,j] = 0
            else:
                if disparity[i,j] < 1:
                    valid_landmark_indices[j] = 0
    
    valid_landmark_indices = valid_landmark_indices.astype(bool)
    visibility_matrix = visibility_matrix.astype(bool)
    disparity = np.ones_like(features[:, valid_landmark_indices, 0])*2000
    valid_indices = visibility_matrix[:,valid_landmark_indices]

    first_timestep_visible_landmarks = features[:, valid_landmark_indices][0][valid_indices[0]]

    disparity[valid_indices] = features[:, valid_landmark_indices, 0][valid_indices] - features[:, valid_landmark_indices, 2][valid_indices]

    z_0 = np.ones_like(disparity)*2000
    x_0 = np.ones_like(disparity)*2000
    y_0 = np.ones_like(disparity)*2000
    z_0[valid_indices] = (K[0,0]*b)/(disparity[valid_indices])
    y_0[valid_indices] = (((features[:, valid_landmark_indices, 1] - K[1,2])*z_0)/K[1,1])[valid_indices]
    x_0[valid_indices] = (((features[:, valid_landmark_indices, 0] - K[0,2])*z_0)/K[0,0])[valid_indices]
    coords = np.stack((x_0, y_0, z_0), axis = 2)
    coords_homogenised = np.ones((coords.shape[0], coords.shape[1], coords.shape[2]+1))
    coords_homogenised[:,:,:3] = coords

    print(coords)

    print(np.tensordot(imu_T_cam, coords_homogenised, axes=[[1],[2]]).shape)
    coords = np.transpose(np.tensordot(imu_T_cam, coords_homogenised, axes=[[1],[2]]), (1,2,0))[:,:,:3]
    # coords = (imu_T_cam @ coords_homogenised.T).T[:,:3]

    with open("coordinates_" + "{0:0=2d}".format(filenumber) + ".npy", "wb") as f:
        np.save(f, coords)
    with open("visibility_matrix_" + "{0:0=2d}".format(filenumber) + ".npy", "wb") as f:
        np.save(f, visibility_matrix)
    with open("valid_landmark_indices_" + "{0:0=2d}".format(filenumber) + ".npy", "wb") as f:
        np.save(f, valid_landmark_indices)