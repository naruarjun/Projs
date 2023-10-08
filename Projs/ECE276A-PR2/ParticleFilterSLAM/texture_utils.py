import os
import numpy as np
import matplotlib
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

def normalize(img):
    max_ = img.max()
    min_ = img.min()
    return (img - min_)/(max_-min_)

def get_rgbijd(d):
    disparity = d.astype(np.float32)
    dd = (-0.00304 * disparity + 3.31)
    z = 1.03 / dd
    print("Got Depth")
    v_temp,u_temp = np.mgrid[0:disparity.shape[1],0:disparity.shape[2]]
    v = np.stack((v_temp for ind in range(disparity.shape[0])), axis = 0)
    u = np.stack((u_temp for ind in range(disparity.shape[0])), axis = 0)
    fx = 585.05108211
    fy = 585.05108211
    cx = 315.83800193
    cy = 242.94140713
    x = (u-cx) / fx * z
    print("Got x")
    y = (v-cy) / fy * z
    print("Got y")

    rgbu = np.round((u * 526.37 + dd*(-4.5*1750.46) + 19276.0)/fx)
    print("Got rgbu")
    rgbv = np.round((v * 526.37 + 16662.0)/fy)
    print("Got rgbu")
    valid = (rgbu>= 0)&(rgbu < disparity.shape[2])&(rgbv>=0)&(rgbv<disparity.shape[1])

    return rgbu, rgbv, x, y, z, valid

def convert_grid_to_pc(coords, colors, valid):
    new_coords = coords[valid].reshape(-1,3)
    new_colors = colors[valid].reshape(-1,3)
    return new_coords, new_colors

def convert_to_world_coordinates(robot_coordinates, x, y, z):
    coords_array = np.stack((x, y, z), axis = 2)
    cam_to_kinect_rotation = np.array([[0,0,1],
                                      [-1,0,0],
                                      [0,-1,0]])
    kinect_to_robot_rotation = np.array([[np.cos(0.314159), 0 , -np.sin(0.314159)],
                                         [0,1,0],
                                         [np.sin(0.314159), 0 , np.cos(0.314159)]])
    
    robot_to_world_rotation = np.array([[np.cos(robot_coordinates[2]), -np.sin(robot_coordinates[2]), 0],
            [np.sin(robot_coordinates[2]), np.cos(robot_coordinates[2]), 0],
            [0,0,1]
        ])
    x_translation = 0.33276
    final_coords = np.transpose(np.tensordot(cam_to_kinect_rotation, coords_array, axes = [[1],[2]]), (1,2,0))
    final_coords = np.transpose(np.tensordot(kinect_to_robot_rotation, final_coords, axes = [[1],[2]]), (1,2,0))
    final_coords = np.transpose(np.tensordot(robot_to_world_rotation, final_coords, axes = [[1],[2]]), (1,2,0))
    

    final_coords[:,:,0] += x_translation + robot_coordinates[0]
    final_coords[:,:,1] += x_translation + robot_coordinates[1]

    return final_coords

def read_disparity_rgb(disparity_images_path, rgb_images_path):
    image_path_list = []
    disparity_path_list = []

    for file in os.listdir(disparity_images_path):
        disparity_path_list.append(file)
    
    for file in os.listdir(rgb_images_path):
        image_path_list.append(file)

    disparity_path_list.sort(key = lambda x: int(x[12:-4]))
    image_path_list.sort(key = lambda x: int(x[6:-4]))

    disparity_images = []
    rgb_images = []
    for file in tqdm(disparity_path_list):
        imd = cv2.imread(os.path.join(disparity_images_path, file),cv2.IMREAD_UNCHANGED) #(480x640)
        disparity_images.append(imd)
    
    for file in tqdm(image_path_list):
        imc = cv2.imread(os.path.join(rgb_images_path, file))[...,::-1] #(480x640)
        rgb_images.append(imc)
    
    return np.array(disparity_images), np.array(rgb_images)

def their_get_rgbijd(disparity):
    disparity = disparity.astype(np.float32)
    dd = (-0.00304 * disparity + 3.31)
    z = 1.03 / dd
    v,u = np.mgrid[0:disparity.shape[0],0:disparity.shape[1]]
    #u,v = np.meshgrid(np.arange(disparity.shape[1]),np.arange(disparity.shape[0]))

    # get 3D coordinates 
    fx = 585.05108211
    fy = 585.05108211
    cx = 315.83800193
    cy = 242.94140713
    x = (u-cx) / fx * z
    y = (v-cy) / fy * z
    # calculate the location of each pixel in the RGB image
    rgbu = np.round((u * 526.37 + dd*(-4.5*1750.46) + 19276.0)/fx)
    rgbv = np.round((v * 526.37 + 16662.0)/fy)

    return rgbu, rgbv, z


if __name__ == "__main__":
    dataset = 20
    disparity_images_path = "../data/Disparity%d"%dataset
    rgb_images_path = "../data/RGB%d"%dataset

    # image_path_list = []
    # disparity_path_list = []

    # for file in os.listdir(disparity_images_path):
    #     disparity_path_list.append(file)
    
    # for file in os.listdir(rgb_images_path):
    #     image_path_list.append(file)

    # disparity_path_list.sort(key = lambda x: int(x[12:-4]))
    # image_path_list.sort(key = lambda x: int(x[6:-4]))

    # disparity_images = []
    # rgb_images = []
    # for file in tqdm(disparity_path_list):
    #     imd = cv2.imread(os.path.join(disparity_images_path, file),cv2.IMREAD_UNCHANGED) #(480x640)
    #     disparity_images.append(imd)
    
    # for file in tqdm(image_path_list):
    #     imc = cv2.imread(os.path.join(rgb_images_path, file))[...,::-1] #(480x640)
    #     rgb_images.append(imc)
    
    # disparity_images = np.array(disparity_images)

    disparity_images, rgb_images = read_disparity_rgb(disparity_images_path, rgb_images_path)
    disparity_images = disparity_images

    rgbu, rgbv, x, y, z, valid = get_rgbijd(disparity_images)
    data = {
        "rgbu" : rgbu,
        "rgbv" : rgbv,
        "x" : x,
        "y" : y,
        "z" : z,
        "valid" : valid
    }

    with open("rgdb%d.pkl"%dataset, "wb") as f:
        pickle.dump(data, f)

    # print("Shapes", rgbu.shape, rgbv.shape, z.shape)

    # for i in range(len(disparity_images)):
    #     rgbu2, rgbv2, z2 = their_get_rgbijd(disparity_images[i])
    #     print("Shapes", rgbu2.shape, rgbv2.shape, z2.shape)

    #     print(np.allclose(rgbu[i], rgbu2))
    #     print(np.allclose(rgbv[i], rgbv2))
    #     print(np.allclose(z[i], z2))
    
    # rgbu, rgbv, x, y, z = rgbu[0], rgbv[0], x[0], y[0], z[0]
    # imc = rgb_images[0]
    # imd = disparity_images[0]
    # disparity = imd.astype(np.float32)
    # valid = (rgbu>= 0)&(rgbu < disparity.shape[1])&(rgbv>=0)&(rgbv<disparity.shape[0])
    # # display valid RGB pixels
    # fig = plt.figure(figsize=(10, 13.3))
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(z[valid],-x[valid],-y[valid],c=imc[rgbv[valid].astype(int),rgbu[valid].astype(int)]/255.0)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.view_init(elev=0, azim=180)
    # plt.show()

    # # display disparity image
    # plt.imshow(normalize(imd), cmap='gray')
    # plt.show()
