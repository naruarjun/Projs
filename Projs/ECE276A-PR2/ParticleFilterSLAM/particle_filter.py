import numpy as np
from tqdm import tqdm
import tkinter
import matplotlib
import matplotlib.pyplot as plt
from pr2_utils import bresenham2D, effMapCorrelation, noteffMapCorrelation, perturbed_map_corr
from texture_utils import read_disparity_rgb, get_rgbijd, convert_grid_to_pc, convert_to_world_coordinates
import pickle
import argparse
# matplotlib.use('TkAgg')

def time_sync_dispartity(disparity_timestamps, timestamps):
    time_sync = np.searchsorted(timestamps, disparity_timestamps)
    return time_sync

def process_hokuyo(data):
    angles = np.linspace(data["angle_min"], data["angle_max"], data["ranges"].shape[1]).reshape(1,-1)
    x = data["ranges"] * np.cos(angles).reshape(1,-1) + 0.30183 - (0.3302/2)
    y = data["ranges"] * np.sin(angles).reshape(1,-1)

    points = np.stack((x,y), axis = 2)

    data["points"] = points

    return data


def encoder2vel(data):
    encoder_counts = data["counts"].T
    encoder_stamps = data["time_stamps"]

    time_diffs = encoder_stamps[1:] - encoder_stamps[:-1]
    right_velocities = (((encoder_counts[1:,0] + encoder_counts[1:,2])*0.0022)/2)/time_diffs
    left_velocities = (((encoder_counts[1:,1] + encoder_counts[1:,3])*0.0022)/2)/time_diffs

    new_data = {
        "velocity" : (right_velocities + left_velocities)/2.0,
        "timestamps" : data["time_stamps"][1:]
    }

    return new_data

def imu_encoder_time_sync(imu_data_dict, encoder_data_dict):
    imu_stamps = imu_data_dict["timestamps"]
    encoder_stamps = encoder_data_dict["timestamps"]

    time_sync_encoder = np.searchsorted(imu_stamps, encoder_stamps)
    time_sync_imu = np.searchsorted(encoder_stamps, imu_stamps)
    if imu_stamps.min() < encoder_stamps.min():
        min_time = imu_stamps.min()
    else:
        min_time = encoder_stamps.min()

    combined_data = {
        "velocity" : encoder_data_dict["velocity"],
        "angular_velocity" : imu_data_dict["angular_velocity"],
        "imu_stamps" :  imu_data_dict["timestamps"] - min_time,
        "encoder_stamps" : encoder_data_dict["timestamps"] - min_time,
        "encoder_sync" : time_sync_encoder,
        "imu_sync" : time_sync_imu,
        "min_time" : min_time
    }

    return combined_data

def encoder_hokuyo_sync(encoder_data, hokuyo_data):
    hokuyo_stamps = hokuyo_data["timestamps"]
    encoder_stamps = encoder_data["encoder_stamps"]

    time_sync_hokuyo = np.searchsorted(encoder_stamps, hokuyo_stamps)
    return time_sync_hokuyo

def get_closest_encoder(encoder_data, hokuyo_data, hokuyo_time_sync,  index):
        closest_index = hokuyo_time_sync[index]
        if encoder_data['encoder_stamps'][closest_index] > hokuyo_data["timestamps"][index]:
            closest_index = closest_index - 1
        return closest_index

def get_rgbij(d):
    dd = -0.00304*d + 3.31
    depth = 1.03 / dd
    i = np.linspace(0, d.shape[2], d.shape[2], dtype = np.int32).astype(np.float32)
    iss = np.stack((i for ind in range(d.shape[0])), axis = 0)
    j = np.linspace(0, d.shape[1], d.shape[1], dtype = np.int32).astype(np.float32)
    js = np.stack((j for ind in range(d.shape[0])), axis = 0)
    rgbi = (526.37 * iss + (-4.5 * 1750.66) * dd + 19276.0) / 585.051
    rgbj = (526.37 * js + 16662) / 585.051

    return rgbi, rgbj, depth

class ParticleFilter():
    def __init__(self, n = 500, dim_state = 3, map_size = (800, 800), resolution = 0.07, variance = None, initial_location = None):
        self.n = n
        self.dim_state = dim_state
        self.particles = np.zeros((n, dim_state))
        self.alphas = np.ones((n,))/n
        self.last_predict_timestamp = 0
        self.last_action_timestamp = 0
        self.current_timestamp = 0
        self.current_imu_index = 0
        self.current_encoder_index = 0 
        self.resolution = resolution
        self.map = np.zeros(map_size)
        self.log_odds = np.zeros(map_size)
        self.initial_grid_location = initial_location
        if self.initial_grid_location is None:
            self.initial_grid_location = np.array([400, 200])
        self.x_im = np.linspace(0, map_size[0] * self.resolution, map_size[0]) + 0.5*self.resolution - self.initial_grid_location[0]*self.resolution
        self.y_im = np.linspace(0, map_size[1] * self.resolution, map_size[1]) + 0.5*self.resolution - self.initial_grid_location[1]*self.resolution
        self.final_locations = np.meshgrid()
        self.variance = np.diag(variance)
        self.thresh = self.n+1
    
    def get_current_location(self):
        return (self.particles * self.alphas.reshape(-1,1)).sum(axis = 0).reshape(3,)

    
    def predict(self, data, sensor = "Encoder", noise = True):
        if sensor == "IMU":
            encoder_index = self.get_closest_encoder(data, self.current_imu_index)
            if encoder_index != -1:
                self.particles = self.motion_model(data['velocity'][encoder_index], data['angular_velocity'][self.current_imu_index], data['imu_stamps'][self.current_imu_index] - self.current_timestamp)
                self.current_timestamp = data['imu_stamps'][self.current_imu_index]
                self.current_encoder_index = encoder_index
            self.current_imu_index += 1
             
        if sensor == "Encoder":
            imu_index = self.get_closest_imu(data, self.current_encoder_index)
            old_imu_index = self.get_closest_imu(data, self.current_encoder_index - 1)
            if imu_index != -1 and old_imu_index!= -1:
                self.particles = self.motion_model(data['velocity'][self.current_encoder_index], data['angular_velocity'][old_imu_index :imu_index + 1].mean(), data['encoder_stamps'][self.current_encoder_index] - self.current_timestamp, noise)
                self.current_timestamp = data['encoder_stamps'][self.current_encoder_index]
                self.current_imu_index = imu_index
            self.current_encoder_index += 1
    
    def predict_wo_index_update(self, data, timestamp, sensor = "Encoder", update_time = True):
        if sensor == "IMU":
            self.particles = self.motion_model(data['velocity'][self.current_encoder_index], data['angular_velocity'][self.current_imu_index], timestamp - self.current_timestamp)
            self.current_timestamp = timestamp
        if sensor == "Encoder":
            self.particles = self.motion_model(data['velocity'][self.current_encoder_index], data['angular_velocity'][self.current_imu_index], timestamp - self.current_timestamp)
            self.current_timestamp = timestamp


    def update(self, data, index):
        correlations = noteffMapCorrelation(self.map, self.particles, data["points"][index], self.resolution, self.initial_grid_location)
        if correlations.sum() !=0:
            new_alphas = self.alphas * correlations
            self.alphas = new_alphas / new_alphas.sum()

    def get_closest_encoder(self, data, index):
        closest_index = data['imu_sync'][index]
        if closest_index == data['encoder_stamps'].shape[0]:
            closest_index = closest_index - 1
        if data['encoder_stamps'][closest_index] > data['imu_stamps'][index]:
            closest_index = closest_index - 1
        
        return closest_index
    
    def get_closest_imu(self, data, index):
        if index == -1:
            return 0
        closest_index = data['encoder_sync'][index]
        if closest_index == data['imu_stamps'].shape[0]:
            closest_index = closest_index - 1
        if data['imu_stamps'][closest_index] > data['encoder_stamps'][index]:
            closest_index = closest_index - 1
        return closest_index
    
    def motion_model(self, velocity, omega, tau, noise = True):
        const = omega*tau/2
        update_theta = self.particles[:,2] + const
        noise_add = np.random.multivariate_normal([0,0,0], self.variance, self.n)
        if noise == False:
            return self.particles + tau*np.stack((velocity * np.sinc(const) * np.cos(update_theta), velocity * np.sinc(const) * np.sin(update_theta), np.ones_like(update_theta)*omega), axis = 1)
        return self.particles + tau*np.stack((velocity * np.sinc(const) * np.cos(update_theta), velocity * np.sinc(const) * np.sin(update_theta), np.ones_like(update_theta)*omega), axis = 1) + noise_add

    def update_map(self, data, index):
        max_ind = np.argmax(self.alphas)
        current_position = self.particles[max_ind]
        
        robot_grid_index = current_position[:2][[1,0]]
        robot_grid_index[0] = - robot_grid_index[0]
        
        robot_grid_index = (((robot_grid_index)/self.resolution) + self.initial_grid_location).astype(int)

        rotation_matrix = np.array([[np.cos(current_position[2]), -np.sin(current_position[2]), current_position[0]],
            [np.sin(current_position[2]), np.cos(current_position[2]), current_position[1]],
            [0,0,1]
        ])
        
        
        current_imp_points = data["points"][index]#[np.linalg.norm(data["points"][index], axis = 1) <= 30]
        
        world_frame_coordinates = (rotation_matrix @ (np.stack((current_imp_points[:,0], current_imp_points[:,1], np.ones_like(current_imp_points[:,1])), axis = 1).T)).T
        indexes = world_frame_coordinates[:,0:2][:,[1,0]]
        indexes[:,0] = - indexes[:,0]
        indexes = ((indexes) / self.resolution).astype(np.int32)
        
        indexes = (indexes +self.initial_grid_location.reshape(1,2)).astype(np.int32)
        
        for i in range(len(indexes)):
            cells = bresenham2D(robot_grid_index[0], robot_grid_index[1], indexes[i][0], indexes[i][1]).T
            self.log_odds[indexes[i][0], indexes[i][1]] += 2*np.log(4)
            self.log_odds[cells[:,0].astype(np.int32), cells[:,1].astype(np.int32)] -= np.log(4)
        
        self.map[self.log_odds>0] = 1
    
    def resample(self):
        neff = 1/((self.alphas**2).sum())
        if neff < self.thresh:
            indices = np.arange(self.n)
            self.particles = self.particles[np.random.choice(indices, self.n, p = self.alphas)]
            self.alphas = np.ones((self.n,))/self.n
    
    # def get_initial_map_size_estimate(self):

    def xyz2grid(self, points, colours, z_range):
        valid_colours = colours[points[:,2] > z_range[0]]
        valid_points = points[points[:,2] > z_range[0]]
        valid_colours = valid_colours[valid_points[:,2] < z_range[1]]
        valid_points = valid_points[valid_points[:,2] < z_range[1]]
        indexes = valid_points[:,0:2][:,[1,0]]
        indexes[:,0] = - indexes[:,0]
        indexes = ((indexes) / self.resolution).astype(np.int32)
        
        indexes = (indexes +self.initial_grid_location.reshape(1,2)).astype(np.int32)
        return indexes, valid_colours
    # def evaluate_corr(self, data, index):
    #     correlations = effMapCorrelation(self.map, self.particles, data["points"][index], self.resolution, self.initial_grid_location)

parser = argparse.ArgumentParser(description='Particle Filter')
parser.add_argument('--dataset', type=int,
                    help='Dataset to run on', default = 20)
parser.add_argument('--skip', type=int,
                    help='Number of LiDAR frames to skip, 1 for no skipping', default = 1)
parser.add_argument('--particles', type=int,
                    help='Number of particles in the filter', default = 20)


args = parser.parse_args()

if __name__ == '__main__':
    dataset = args.dataset
    print(dataset)
    encoder_data_dict = {}
    imu_data_dict = {}
    hokuyo_data_dict = {}
  
    with np.load("../data/Encoders%d.npz"%dataset) as data:
        encoder_counts = data["counts"] # 4 x n encoder counts
        encoder_stamps = data["time_stamps"] # encoder time stamps
        encoder_data_dict = encoder2vel(data)

    with np.load("../data/Hokuyo%d.npz"%dataset) as data:
        lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
        lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
        lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
        lidar_range_min = data["range_min"] # minimum range value [m]
        lidar_range_max = data["range_max"] # maximum range value [m]
        lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
        lidar_stamsp = data["time_stamps"]  # acquisition times of the lidar scans
        hokuyo_data_dict = {
            "angle_min" : data["angle_min"],
            "angle_max" : data["angle_max"],
            "angle_increment" : data["angle_increment"],
            "range_min" : data["range_min"],
            "range_max" : data["range_max"],
            "ranges" : data["ranges"].T,
            "timestamps" : data["time_stamps"]
        }

    with np.load("../data/Imu%d.npz"%dataset) as data:
        imu_angular_velocity = data["angular_velocity"].T[:,2] # angular velocity in rad/sec
        imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
        imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
        imu_data_dict = {
            "angular_velocity" : data["angular_velocity"].T[:,2],
            "timestamps" : data["time_stamps"]
        }

    image_data = {}
    with np.load("../data/Kinect%d.npz"%dataset) as data:
        disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
        image_data["disparity_timestamps"] = data["disparity_time_stamps"]
        rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images
        image_data["rgb_timestamps"] = data["rgb_time_stamps"]
    
    disparity_images_path = "../data/Disparity%d"%dataset
    rgb_images_path = "../data/RGB%d"%dataset

    disparity_images, rgb_images = read_disparity_rgb(disparity_images_path, rgb_images_path)

    combined_data = imu_encoder_time_sync(imu_data_dict, encoder_data_dict)

    hokuyo_data_dict = process_hokuyo(hokuyo_data_dict)
    
    hokuyo_data_dict["timestamps"] -= combined_data["min_time"]
    # Uncomment
    image_data["disparity_timestamps"] -= combined_data["min_time"]
    image_data["rgb_timestamps"] -= combined_data["min_time"]
    image_data["rgb_sync"] = time_sync_dispartity(image_data["disparity_timestamps"], image_data["rgb_timestamps"])
    image_data["encoder_sync"] = time_sync_dispartity(image_data["disparity_timestamps"], combined_data["encoder_stamps"])
    image_data["hokuyo_sync"] = time_sync_dispartity(image_data["disparity_timestamps"], hokuyo_data_dict["timestamps"])
    image_data["hokuyo_dis_sync"] = time_sync_dispartity(hokuyo_data_dict["timestamps"], image_data["disparity_timestamps"])

    # rgbu, rgbv, x, y, z, valid = get_rgbijd(disparity_images)
    preproc_image_data = None
    with open("rgdb%d.pkl"%dataset, "rb") as f:
        preproc_image_data = pickle.load(f)
    
    rgbu, rgbv, x, y, z, valid = preproc_image_data["rgbu"], preproc_image_data["rgbv"], preproc_image_data["x"], preproc_image_data["y"], preproc_image_data["z"], preproc_image_data["valid"]

    time_sync_hokuyo = encoder_hokuyo_sync(combined_data, hokuyo_data_dict)

    resolution = 0.07

    DR_particle_filter = ParticleFilter(n = 10, variance=[1,1,1], resolution = resolution, map_size=(1600,1600))

    dr_states = []
    states = []

    previous_encoder = 0

    # Get initial map size estimate via dead reckoning
    print("")
    for i in tqdm(range(len(combined_data['velocity']))):
        DR_particle_filter.predict(combined_data, noise=False)
        dr_states.append(DR_particle_filter.particles[0])

    dr_states = np.array(dr_states)

    rows_min = dr_states[:,1].min()
    rows_max = dr_states[:,1].max()

    cols_min = dr_states[:,0].min()
    cols_max = dr_states[:,0].max()

    padding = 500
    

    row_range = np.int32((rows_max - rows_min) / resolution + padding)
    cols_range = np.int32((cols_max - cols_min) / resolution + padding)

    initial_location_x = np.array((rows_max) / resolution + padding/2)
    initial_location_y = np.array((-cols_min) / resolution + padding/2)

    print("initial", np.array([initial_location_x, initial_location_y]))

    #0.001, 0.001, 0.001

    particle_filter = ParticleFilter(n = args.particles, map_size = (row_range, cols_range), resolution = resolution, variance = [0.001,0.001,0.001], initial_location = np.array([initial_location_x, initial_location_y]))
    skip = args.skip

    # Dead Reckoning map
    # particle_filter = ParticleFilter(n = 5, map_size = (row_range, cols_range), resolution = resolution, variance = [0,0,0], initial_location = np.array([initial_location_x, initial_location_y]))
    # skip = 1
    # for i in tqdm(range(len(hokuyo_data_dict['points']))):
        
    #     # if i == 400:
    #     #     break
    #     if i%skip==0:
    #         nearest_encoder = get_closest_encoder(combined_data, hokuyo_data_dict, time_sync_hokuyo,  i)
    #         for ind in range(nearest_encoder - previous_encoder):
    #             particle_filter.predict(combined_data, noise=False)
    #             # states.append(particle_filter.particles[0])
    #         previous_encoder = nearest_encoder
    #         particle_filter.predict_wo_index_update(combined_data, hokuyo_data_dict["timestamps"][i], sensor = "Encoder")
    #         # particle_filter.update(hokuyo_data_dict, i)
    #         states.append(particle_filter.get_current_location())
    #         particle_filter.update_map(hokuyo_data_dict, i)
    #         # particle_filter.resample()
    # with open("./final_runs/drstates_"+str(particle_filter.n)+"_"+str(dataset)+"_"+str(skip)+".npy", 'wb') as f:
    #     np.save(f, states)
    # with open("./final_runs/drlog_odds_"+str(particle_filter.n)+"_"+str(dataset)+"_"+str(skip)+".npy", 'wb') as f:
    #     np.save(f, particle_filter.log_odds)
    # with open("./final_runs/drmap_"+str(particle_filter.n)+"_"+str(dataset)+"_"+str(skip)+".npy", 'wb') as f:
    #     np.save(f, particle_filter.map)

    # PF map
    for i in tqdm(range(len(hokuyo_data_dict['points']))):
        
        # if i == 400:
        #     break
        if i%skip==0:
            nearest_encoder = get_closest_encoder(combined_data, hokuyo_data_dict, time_sync_hokuyo,  i)
            for ind in range(nearest_encoder - previous_encoder):
                particle_filter.predict(combined_data)
                # states.append(particle_filter.particles[0])
            previous_encoder = nearest_encoder
            particle_filter.predict_wo_index_update(combined_data, hokuyo_data_dict["timestamps"][i], sensor = "Encoder")
            particle_filter.update(hokuyo_data_dict, i)
            states.append(particle_filter.get_current_location())
            particle_filter.update_map(hokuyo_data_dict, i)
            particle_filter.resample()
            # if i%500==0:
            #     with open("./videos_20/test_log_odds_"+str(particle_filter.n)+"_"+str(dataset)+"_"+str(i)+".npy", 'wb') as f:
            #         np.save(f, particle_filter.log_odds)
            #     with open("./videos_20/test_states_"+str(particle_filter.n)+"_"+str(dataset)+"_"+str(i)+".npy", 'wb') as f:
            #         np.save(f, np.array(states))
        
    states = np.array(states)

    # with open("./final_runs/test_states_"+str(particle_filter.n)+"_"+str(dataset)+"_"+str(skip)+".npy", 'wb') as f:
    #     np.save(f, states)
    # with open("./final_runs/test_log_odds_"+str(particle_filter.n)+"_"+str(dataset)+"_"+str(skip)+".npy", 'wb') as f:
    #     np.save(f, particle_filter.log_odds)
    # with open("./final_runs/test_map_"+str(particle_filter.n)+"_"+str(dataset)+"_"+str(skip)+".npy", 'wb') as f:
    #     np.save(f, particle_filter.map)
    

    texture_map = np.zeros((particle_filter.map.shape[0], particle_filter.map.shape[1], 3))

    # states = np.load("./final_runs/states_"+str(particle_filter.n)+"_"+str(dataset)+"_"+str(skip)+".npy")

    print("Creating texture map")
    for i, state in enumerate(tqdm(states)):
        disparity_index = image_data["hokuyo_dis_sync"][i*skip]
        if disparity_index == len(x):
            break
        rgb_index = image_data["rgb_sync"][disparity_index]
        if rgb_index == len(rgb_images):
            break
            rgb_index = rgb_index - 1

        x_temp, y_temp, z_temp = x[disparity_index], y[disparity_index], z[disparity_index]

        final_coords = convert_to_world_coordinates(state, x_temp, y_temp, z_temp)
        final_points, final_colours = convert_grid_to_pc(final_coords, rgb_images[rgb_index], valid[disparity_index])
        height = 0.38001 + 0.254/2
        grid_indices, final_colors = particle_filter.xyz2grid(final_points, final_colours, [height - 0.5, height + 0.5])
        texture_map[grid_indices[:,0], grid_indices[:,1]] =  final_colors
        # if i%100==0:
        #     with open("./videos_tex_20/texture_"+str(particle_filter.n)+"_"+str(dataset)+"_"+str(i)+".npy", 'wb') as f:
        #         np.save(f, texture_map)

    
    # with open("./final_runs/texture_map_"+str(particle_filter.n)+"_"+str(dataset)+"_"+str(skip)+".npy", 'wb') as f:
    #     np.save(f, texture_map)


    
    plt.figure(figsize = (20,20))
    
    # particle_filter.map[particle_filter.log_odds>0] = 1
    # particle_filter.log_odds[particle_filter.log_odds>1000] = 1000
    # particle_filter.log_odds[particle_filter.log_odds<-1000] = -1000
    
    # #plt.imshow(particle_filter.map, cmap = 'gray')
    particle_filter.log_odds = np.clip(particle_filter.log_odds, -709.78, 709.78)

    # Save map image
    plt.imshow(1.0 - 1./(1.+np.exp(particle_filter.log_odds)), 'Greys')
    # plt.savefig(str(particle_filter.n) + "-map1.png")
    plt.ioff()
    plt.show()
    
    # #Save Trajectory image
    plt.plot(states[:,0], states[:,1], label = "Dead Reckoning")
    # plt.savefig(str(particle_filter.n) + "-traj1.png")
    plt.ioff()
    plt.show()

    # #Save Map with trajectory
    plt.imshow(1.0 - 1./(1.+np.exp(particle_filter.log_odds)), 'Greys')
    plt.plot((states[:,0].reshape(-1)/particle_filter.resolution) + particle_filter.initial_grid_location[1], (-states[:,1].reshape(-1) / particle_filter.resolution) + particle_filter.initial_grid_location[0], label = "Dead Reckoning")
    # plt.savefig(str(particle_filter.n) + "-maptraj1.png")
    plt.ioff()
    plt.show()

    # Texture Map
    plt.imshow(texture_map/255.0)
    plt.ioff()
    plt.show()
    # print("Plotted trajectory")

    

