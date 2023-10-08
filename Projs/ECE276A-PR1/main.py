import pandas as pd
import os
import numpy as np
import transforms3d
import jax
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

g = 9.8
accelerometer_sensitivity = 300 #mV/g
accelerometer_ref = 3.3 #V
gyroscope_sensititivity_roll_pitch = (3.33 * (180)) / (np.pi) #mV/degree/s
gyroscope_v_ref_roll_pitch = 3.3 #V
gyroscope_sensititivity_yaw = (3.33 * (180)) / (np.pi) #mV/degree/s
gyroscope_v_ref_yaw = 3.3 #V
imu_data_path = "trainset/imu"
vicon_data_path = "trainset/vicon"
cam_data_path = "trainset/cam"

imu_test_path = "testset/imu"
cam_test_path = "testset/cam"

def read_pickle(file):
  return pd.read_pickle(file)

def read_folder_pickle(folder):
  data = []
  files = os.listdir(folder)
  files.sort()
  for file in files:
    data.append(read_pickle(os.path.join(folder, file)))
  return data

def get_imu_value(raw, bias, vref, sensitivity):
  scale_factor = ((vref / 1023) / sensitivity)
  return ((raw - bias) * scale_factor).T

def calculate_bias(values, time, time_threshold, default_value = np.array([[0,0,0]])):
  values_new = values[time < time_threshold]
  bias = (values_new - default_value).mean(axis = 0)
  return bias.reshape(1,-1)

def get_unbiased_imu_value(imu_data, vicon_data, index):
  accelerometer_values = imu_data[index]['vals'][:3].astype(np.float16)
  gyroscope_values = imu_data[index]['vals'][3:].astype(np.float16)
  vicon_values = vicon_data[index]['rots']
  time_vicon = vicon_data[index]['ts']# - vicon_data[0]['ts'][0][0]
  timestamps = imu_data[index]['ts'].T
  time = (timestamps - timestamps[0]).reshape(-1)

  accelerometer_readings = get_imu_value(accelerometer_values, 0, accelerometer_ref*1000, accelerometer_sensitivity)
  accelerometer_readings[:,0:2] = - accelerometer_readings[:,0:2]
  gyroscope_readings = get_imu_value(gyroscope_values, 0, gyroscope_v_ref_roll_pitch*1000, gyroscope_sensititivity_yaw)

  accelerometer_readings = accelerometer_readings - calculate_bias(accelerometer_readings, time, 4, np.array([[0, 0, 1]]))
  gyroscope_readings = gyroscope_readings - calculate_bias(gyroscope_readings, time, 4)
  gyroscope_readings = gyroscope_readings[:,[1,2,0]]


  return accelerometer_readings, gyroscope_readings, timestamps.reshape(-1), vicon_values, time_vicon


def getrpy(quat):
  return transforms3d.euler.quat2euler(quat)
@jax.jit
def exponential_map(delta_t, omega_t):
  rotation_vector = (delta_t * omega_t)/2
  norm_r = jax.numpy.linalg.norm(rotation_vector)
  return jax.numpy.hstack([jax.numpy.cos(norm_r).reshape(1,), (rotation_vector / norm_r) * jax.numpy.sin(norm_r)])

@jax.jit
def log_map(quat_t):
  quat_t_norm = jax.numpy.linalg.norm(quat_t)
  return jax.numpy.hstack([jax.numpy.log(quat_t_norm), (quat_t[1:] / jax.numpy.linalg.norm(quat_t[1:])) * jax.numpy.arccos(quat_t[0] / quat_t_norm)])

@jax.jit
def quaternion_multiply(q, p):
  return jax.numpy.hstack([(q[0]*p[0] - (q[1:]*p[1:]).sum()).reshape(1,), q[0]*p[1:] + p[0]*q[1:] + jax.numpy.cross(q[1:], p[1:])])

@jax.jit
def motion_model(quat_t, delta_t, omega_t):
  return quaternion_multiply(quat_t, exponential_map(delta_t, omega_t))

@jax.jit
def quaternion_inverse(quat_t):
  return jax.numpy.hstack([quat_t[0].reshape(1,), -quat_t[1:]]) / (jax.numpy.linalg.norm(quat_t)**2)

@jax.jit
def observation_model(quat_t):
  return quaternion_multiply(quaternion_multiply(quaternion_inverse(quat_t), jax.numpy.array([0,0,0,-1])), quat_t)

@jax.jit
def cost_one_timestep(quat_t, quat_t_1, delta_t, omega_t, a_t):
  return 0.5*(jax.numpy.linalg.norm(2*log_map(quaternion_multiply(quaternion_inverse(quat_t_1), motion_model(quat_t, delta_t, omega_t))))**2 + jax.numpy.linalg.norm(a_t - observation_model(quat_t))**2)


@jax.jit
def vectorized_exponential_map(delta_t, omega_t):
  rotation_vector = (delta_t * omega_t)/2
  norm_r = jax.numpy.linalg.norm(rotation_vector, axis =1).reshape(-1,1)
  return jax.numpy.hstack([jax.numpy.cos(norm_r).reshape(-1, 1), (rotation_vector / norm_r) * jax.numpy.sin(norm_r)])

@jax.jit
def vectorized_log_map(quat_t):
  quat_t_norm = jax.numpy.linalg.norm(quat_t, axis = 1).reshape(-1,1)
  return jax.numpy.hstack([jax.numpy.log(quat_t_norm), (quat_t[:, 1:] / jax.numpy.linalg.norm(quat_t[:, 1:])) * jax.numpy.arccos(quat_t[:, 0].reshape(-1,1) / quat_t_norm)])  

@jax.jit
def vectorized_quaternion_multiply(q, p):
  return jax.numpy.hstack([((q[:,0]*p[:,0]) - (q[:, 1:]*p[:, 1:]).sum(axis = 1)).reshape(-1,1), ((q[:, 0]).reshape(-1,1))*p[:, 1:] + ((p[:, 0]).reshape(-1,1))*q[:, 1:] + jax.numpy.cross(q[:, 1:], p[:, 1:])])

@jax.jit
def vectorized_motion_model(quat_t, delta_t, omega_t):
  return vectorized_quaternion_multiply(quat_t, vectorized_exponential_map(delta_t, omega_t))

@jax.jit
def vectorized_observation_model(quat_t):
  return vectorized_quaternion_multiply(vectorized_quaternion_multiply(vectorized_quaternion_inverse(quat_t), jax.numpy.array([[0,0,0,1]]*quat_t.shape[0])), quat_t)

@jax.jit
def vectorized_quaternion_inverse(quats):
  return jax.numpy.hstack([quats[:, 0].reshape(-1, 1), -quats[:, 1:]]) / (jax.numpy.linalg.norm(quats, axis = 1)**2).reshape(-1,1)

def cost(q_array, delta_t, omega_t, a_t):
  new_q_array = jax.numpy.vstack([[[1, 0, 0, 0]], q_array])
  return 0.5*(jax.numpy.linalg.norm(\
                                2*vectorized_log_map(\
                                                     vectorized_quaternion_multiply(\
                                                                                    vectorized_quaternion_inverse(new_q_array[1:]),\
                                                                                     vectorized_motion_model(new_q_array[:-1], delta_t, omega_t)\
                                                                                    )\
                                                      + 1e-6), axis = 1)**2\
          ).sum() \
          + 0.5*(\
                jax.numpy.linalg.norm(\
                                      a_t - vectorized_observation_model(new_q_array[1:])[:, 1:], axis = 1\
                                      )**2\
             ).sum()

def find_initial_quaternion(delta_t, gyroscope_readings):
  quats = []
  q_0 = jax.numpy.array([1,0,0,0])
  for i in range(len(delta_t)):
    q_0 = vectorized_motion_model(q_0.reshape(-1,4), delta_t[i].reshape(-1, 1), gyroscope_readings[i].reshape(-1, 3)).reshape(4,)
    quats.append(q_0)
  return jax.numpy.array(quats)

def projected_gradient_descent(q_array, imu_data, alpha = 0.1, steps = 500, index = 0):
  accelerometer_readings, gyroscope_readings, time, vicon_readings, time_vicon = get_unbiased_imu_value(imu_data, vicon_data, index)
  delta_t = (time[1:] - time[:-1]).reshape(-1,1)
  gyroscope_readings = gyroscope_readings[:-1]
  accelerometer_readings = accelerometer_readings[1:]
  grad_fn = jax.grad(cost)
  for i in range(steps):
    if i==0:
      print("Iteration", i, "Cost:", cost(q_array, delta_t,  gyroscope_readings, accelerometer_readings))
    if(i == steps-1):
      print("Iteration", i, "Cost:", cost(q_array, delta_t,  gyroscope_readings, accelerometer_readings))
    gradient = grad_fn(q_array, delta_t,  gyroscope_readings, accelerometer_readings)
    q_array = q_array - alpha*gradient
    q_array = q_array / jax.numpy.linalg.norm(q_array, axis = 1).reshape(-1,1)
  return q_array


def plot_quat_vicon(quats, vicon, time, time_vicon):
  q_0 = [1, 0, 0, 0]
  rpy_predicted = [[getrpy(q_0)[0], getrpy(q_0)[1], getrpy(q_0)[2]]]
  rpy_vicon = []
  t_old = 0
  for i, t in enumerate(time):
    if i==0:
      continue
    current_rpy = getrpy(quats[i-1])
    rpy_predicted.append([current_rpy[0], current_rpy[1], current_rpy[2]])
    t_old = t
  
  for i in range(vicon_readings.shape[2]):
    rotation_matrix = vicon_readings[:,:,i]
    rpy = transforms3d.euler.mat2euler(rotation_matrix, 'sxyz')
    rpy_vicon.append(rpy)
  rpy_predicted = np.array(rpy_predicted)
  rpy_vicon = np.array(rpy_vicon)
  plt.figure(figsize = (15,7))
  plt.plot(time.reshape(-1), np.array(rpy_predicted[:,0]).reshape(-1))
  plt.plot(time_vicon.reshape(-1), np.array(rpy_vicon[:, 0]).reshape(-1))

def plot_quat_vicon_precomputed(init, quats, time, time_vicon, rpy_vicon, acc, figname):
  q_0 = [1, 0, 0, 0]
  rpy_initial = [[getrpy(q_0)[0], getrpy(q_0)[1], getrpy(q_0)[2]]]
  t_old = 0
  for i, t in enumerate(time):
    if i==0:
      continue
    current_rpy = getrpy(init[i-1])
    rpy_initial.append([current_rpy[0], current_rpy[1], current_rpy[2]])
    t_old = t
  
  plot_acc_initial = vectorized_observation_model(init)
  rpy_initial = np.array(rpy_initial)
  
  q_0 = [1, 0, 0, 0]
  rpy_predicted = [[getrpy(q_0)[0], getrpy(q_0)[1], getrpy(q_0)[2]]]
  t_old = 0
  for i, t in enumerate(time):
    if i==0:
      continue
    current_rpy = getrpy(quats[i-1])
    rpy_predicted.append([current_rpy[0], current_rpy[1], current_rpy[2]])
    t_old = t
  


  plot_acc = vectorized_observation_model(quats)
  rpy_predicted = np.array(rpy_predicted)
  rpy_vicon = np.array(rpy_vicon)
  plt.figure(figsize = (30,30))
  plt.subplot(6,1,1)
  plt.plot(time.reshape(-1), np.array(rpy_initial[:,0]).reshape(-1), label = "Initial Predicted Roll")
  plt.plot(time.reshape(-1), np.array(rpy_predicted[:,0]).reshape(-1), label = "Optimized Predicted Roll")
  plt.plot(time_vicon.reshape(-1), np.array(rpy_vicon[:, 0]).reshape(-1), label = "GT Roll")

  plt.legend(loc="upper right")

  plt.subplot(6,1,2)
  plt.plot(time.reshape(-1), np.array(rpy_initial[:,1]).reshape(-1), label = "Initial Predicted Pitch")
  plt.plot(time.reshape(-1), np.array(rpy_predicted[:,1]).reshape(-1), label = "Optimized Predicted Pitch")
  plt.plot(time_vicon.reshape(-1), np.array(rpy_vicon[:, 1]).reshape(-1), label = "GT Pitch")

  plt.legend(loc="upper right")

  plt.subplot(6,1,3)
  plt.plot(time.reshape(-1), np.array(rpy_initial[:,2]).reshape(-1), label = "Initial Predicted Yaw")
  plt.plot(time.reshape(-1), np.array(rpy_predicted[:,2]).reshape(-1), label = "Optimized Predicted Yaw")
  plt.plot(time_vicon.reshape(-1), np.array(rpy_vicon[:, 2]).reshape(-1), label = "GT Yaw")

  plt.legend(loc="upper right")

  plt.subplot(6,1,4)
  plt.plot(time[1:].reshape(-1), np.array(plot_acc_initial[:,1]).reshape(-1), label = "Initial Predicted Acceleration X")
  plt.plot(time[1:].reshape(-1), np.array(plot_acc[:,1]).reshape(-1), label = "Optimized Predicted Acceleration X")
  plt.plot(time.reshape(-1), np.array(acc[:, 0]).reshape(-1), label = "GT Acceleration X")

  plt.legend(loc="upper right")

  plt.subplot(6,1,5)
  plt.plot(time[1:].reshape(-1), np.array(plot_acc_initial[:,2]).reshape(-1), label = "Initial Predicted Acceleration Y")
  plt.plot(time[1:].reshape(-1), np.array(plot_acc[:,2]).reshape(-1), label = "Optimized Predicted Acceleration Y")
  plt.plot(time.reshape(-1), np.array(acc[:, 1]).reshape(-1),  label = "GT Acceleration Y")

  plt.legend(loc="upper right")

  plt.subplot(6,1,6)
  plt.plot(time[1:].reshape(-1), np.array(plot_acc_initial[:,3]).reshape(-1), label = "Initial Predicted Acceleration X")
  plt.plot(time[1:].reshape(-1), np.array(plot_acc[:,3]).reshape(-1), label = "Optimized Predicted Acceleration Z")
  plt.plot(time.reshape(-1), np.array(acc[:, 2]).reshape(-1), label = "GT Acceleration X")

  plt.legend(loc="upper right")

  plt.savefig(figname)

def plot_quat_no_vicon_precomputed(init, quats, time, acc, figname):
  q_0 = [1, 0, 0, 0]
  rpy_initial = [[getrpy(q_0)[0], getrpy(q_0)[1], getrpy(q_0)[2]]]
  t_old = 0
  for i, t in enumerate(time):
    if i==0:
      continue
    current_rpy = getrpy(init[i-1])
    rpy_initial.append([current_rpy[0], current_rpy[1], current_rpy[2]])
    t_old = t
  
  plot_acc_initial = vectorized_observation_model(init)
  rpy_initial = np.array(rpy_initial)
  
  q_0 = [1, 0, 0, 0]
  rpy_predicted = [[getrpy(q_0)[0], getrpy(q_0)[1], getrpy(q_0)[2]]]
  t_old = 0
  for i, t in enumerate(time):
    if i==0:
      continue
    current_rpy = getrpy(quats[i-1])
    rpy_predicted.append([current_rpy[0], current_rpy[1], current_rpy[2]])
    t_old = t
  


  plot_acc = vectorized_observation_model(quats)
  rpy_predicted = np.array(rpy_predicted)
  plt.figure(figsize = (30,30))
  plt.subplot(6,1,1)
  plt.plot(time.reshape(-1), np.array(rpy_initial[:,0]).reshape(-1), label = "Initial Predicted Roll")
  plt.plot(time.reshape(-1), np.array(rpy_predicted[:,0]).reshape(-1), label = "Optimized Predicted Roll")

  plt.legend(loc="upper right")

  plt.subplot(6,1,2)
  plt.plot(time.reshape(-1), np.array(rpy_initial[:,1]).reshape(-1), label = "Initial Predicted Pitch")
  plt.plot(time.reshape(-1), np.array(rpy_predicted[:,1]).reshape(-1), label = "Optimized Predicted Pitch")

  plt.legend(loc="upper right")

  plt.subplot(6,1,3)
  plt.plot(time.reshape(-1), np.array(rpy_initial[:,2]).reshape(-1), label = "Initial Predicted Yaw")
  plt.plot(time.reshape(-1), np.array(rpy_predicted[:,2]).reshape(-1), label = "Optimized Predicted Yaw")

  plt.legend(loc="upper right")

  plt.subplot(6,1,4)
  plt.plot(time[1:].reshape(-1), np.array(plot_acc_initial[:,1]).reshape(-1), label = "Initial Predicted Acceleration X")
  plt.plot(time[1:].reshape(-1), np.array(plot_acc[:,1]).reshape(-1), label = "Optimized Predicted Acceleration X")

  plt.legend(loc="upper right")

  plt.subplot(6,1,5)
  plt.plot(time[1:].reshape(-1), np.array(plot_acc_initial[:,2]).reshape(-1), label = "Initial Predicted Acceleration Y")
  plt.plot(time[1:].reshape(-1), np.array(plot_acc[:,2]).reshape(-1), label = "Optimized Predicted Acceleration Y")

  plt.legend(loc="upper right")

  plt.subplot(6,1,6)
  plt.plot(time[1:].reshape(-1), np.array(plot_acc_initial[:,3]).reshape(-1), label = "Initial Predicted Acceleration X")
  plt.plot(time[1:].reshape(-1), np.array(plot_acc[:,3]).reshape(-1), label = "Optimized Predicted Acceleration Z")

  plt.legend(loc="upper right")

  plt.savefig(figname)

def get_start_0(grid):
  return grid - grid[0,0]

def image_to_spherical_local(image_width, image_height, horizontal_fov, vertical_fov):
  angle_per_pixel_horizontal = vertical_fov / image_width
  angle_per_pixel_vertical   = horizontal_fov / image_height
  min_horizontal_angle = - (horizontal_fov / 2)
  max_horizontal_angle = (horizontal_fov / 2)
  min_vertical_angle   = - (vertical_fov / 2)
  max_vertical_angle   = (vertical_fov / 2)

  horizontal_spherical =  jax.numpy.linspace(min_horizontal_angle, max_horizontal_angle, image_width)
  vertical_spherical =  jax.numpy.linspace(min_vertical_angle, max_vertical_angle, image_height)

  spherical_image = jax.numpy.meshgrid(horizontal_spherical, vertical_spherical)
  spherical_image = jax.numpy.stack([spherical_image[0], spherical_image[1]], axis = 2)

  horizontal_spherical_0 =  jax.numpy.linspace(0, max_horizontal_angle - min_horizontal_angle, image_width)
  vertical_spherical_0 =  jax.numpy.linspace(0, max_vertical_angle - min_vertical_angle, image_height)
  start_0 = jax.numpy.meshgrid(horizontal_spherical_0, vertical_spherical_0)
  start_0 = jax.numpy.stack([start_0[0], start_0[1]], axis = 2)

  return spherical_image

@jax.jit
def spherical_to_cartesian(spherical_image_grid, r):
  z = - jax.numpy.sin(spherical_image_grid[:,:, 1]) * r
  project_xy = jax.numpy.cos(spherical_image_grid[:,:, 1]) * r
  x = project_xy * jax.numpy.cos(spherical_image_grid[:,:, 0])
  y = - project_xy * jax.numpy.sin(spherical_image_grid[:,:, 0])

  return(jax.numpy.stack([x, y, z], axis = 2))


@jax.jit
def rotate_image_grid(quat, cartesian_image_grid):
  q = quat
  quat_left = jax.numpy.repeat(quat.reshape(1,4), cartesian_image_grid.shape[0] * cartesian_image_grid.shape[1], axis = 0)
  q_inv = quaternion_inverse(quat)
  quat_right = jax.numpy.repeat(q_inv.reshape(1,4), cartesian_image_grid.shape[0] * cartesian_image_grid.shape[1], axis = 0)
  zeros = jax.numpy.zeros((cartesian_image_grid.shape[0], cartesian_image_grid.shape[1]))
  print(quat_left.shape)
  rotated_image_grid = vectorized_quaternion_multiply(vectorized_quaternion_multiply(quat_left, 
              jax.numpy.stack([zeros, cartesian_image_grid[:,:,0], cartesian_image_grid[:,:,1], cartesian_image_grid[:,:,2]], axis = 2).reshape(-1,4)), quat_right)


  rotated_image_grid = rotated_image_grid.reshape(cartesian_image_grid.shape[0], cartesian_image_grid.shape[1], 4)[:, :, 1:]

  return rotated_image_grid

@jax.jit
def rotate_image_grid_using_top_left(quat, cartesian_image_grid):
  top_left_coords = cartesian_image_grid[0,0,:]
  return quaternion_multiply(quaternion_multiply(quat.reshape(4,), jax.numpy.hstack([jax.numpy.array([0]).reshape(1,), top_left_coords.reshape(3,)])), quaternion_inverse(quat))[1:]

@jax.jit
def cartesian_to_spherical(cartesian_image_grid):
  lamb = jax.numpy.arctan2( - cartesian_image_grid[:,:,1], cartesian_image_grid[:,:,0])
  phi = jax.numpy.arctan2( - cartesian_image_grid[:,:,2], jax.numpy.sqrt(cartesian_image_grid[:,:,0]**2 + cartesian_image_grid[:,:,1]**2))
  return jax.numpy.stack([lamb, phi], axis = 2)

@jax.jit
def cartesian_to_spherical_using_top_left(top_left, start_0):
  lamb = jax.numpy.arctan2(top_left[1], top_left[0])
  phi = jax.numpy.arctan2(top_left[2], jax.numpy.sqrt(top_left[0]**2 + top_left[1]**2))
  return start_0 + jax.numpy.array([lamb, phi]).reshape(2,)

def find_nearest_quat(image_time, time):
  new_array = np.array(time - image_time)
  new_array[new_array>0] = np.inf
  return np.argmin(np.absolute(new_array))

def add_image_to_panaroma(final_image, image, spherical_rotated_coords):
  final_coords_xy = (((spherical_rotated_coords[:, :, 0] + (np.pi)) / (2*np.pi)) * final_image.shape[1]).astype(np.int32)
  final_coords_xy[final_coords_xy == final_image.shape[1]] = 1079
  final_coords_z  = np.array((((spherical_rotated_coords[:, :, 1] + (np.pi/2)) / (np.pi)) * final_image.shape[0]).astype(np.int32))
  final_pixel_coords = np.stack([final_coords_z.reshape(-1), final_coords_xy.reshape(-1)], axis = 1)
  final_image[final_pixel_coords[:,0], final_pixel_coords[:,1], :] = image.reshape(-1,3)
  return final_image

def create_panaroma(images, image_time, quats, time, final_image_size):
  spherical_image_grid = image_to_spherical_local(image_width, image_height, horizontal_fov, vertical_fov)
  cartesian_image_grid = spherical_to_cartesian(spherical_image_grid, 1)
  final_image = np.zeros((final_image_size[0], final_image_size[1], 3))
  for i in tqdm(range(images.shape[3])):
    index = find_nearest_quat(image_time[0, i], time)
    current_quat = quats[index]
    rotated_cartesian_image = rotate_image_grid(current_quat, cartesian_image_grid)
    spherical_rotated_coords = cartesian_to_spherical(rotated_cartesian_image)
    
    final_image = add_image_to_panaroma(final_image, images[:,:,:,i], np.array(spherical_rotated_coords))
  return final_image
 
def compute_rre(R_est: np.ndarray, R_gt: np.ndarray):
  """Compute the relative rotation error (geodesic distance of rotation)."""
  assert R_est.shape == (3, 3), 'R_est: expected shape (3, 3), received shape {}.'.format(R_est.shape)
  assert R_gt.shape == (3, 3), 'R_gt: expected shape (3, 3), received shape {}.'.format(R_gt.shape)
  # relative rotation error (RRE)
  rre = np.arccos(np.clip(0.5 * (np.trace(R_est.T @ R_gt) - 1), -1.0, 1.0))
  return rre

imu_data = read_folder_pickle(imu_data_path)
vicon_data = read_folder_pickle(vicon_data_path)
cam_data = read_folder_pickle(cam_data_path)

imu_test_data = read_folder_pickle(imu_test_path)
cam_test_data = read_folder_pickle(cam_test_path)

# Training Set RPY, Acc graph generation
print("Generating Graphs Training Set")
for i in range(len(imu_data)):
  print("File No:", (i+1))
  accelerometer_readings, gyroscope_readings, time, vicon_readings, time_vicon = get_unbiased_imu_value(imu_data, vicon_data, i)
  delta_t = (time[1:] - time[:-1]).reshape(-1,1)
  init = find_initial_quaternion(delta_t.reshape(-1,1), gyroscope_readings[:-1])
  q_array_final = projected_gradient_descent(init, imu_data, index = i, alpha = 0.01)
  rpy_vicon = []
  quats_vicon = []
  for j in range(vicon_readings.shape[2]):
    rotation_matrix = vicon_readings[:,:,j]
    try:
      quat = transforms3d.quaternions.mat2quat(rotation_matrix)
    except:
      quats_vicon.append(quats_vicon[-1])
      #rpy = transforms3d.euler.mat2euler(rotation_matrix, 'sxyz')
      rpy_vicon.append(rpy_vicon[-1])
      continue
    quats_vicon.append(quat)
    rpy = transforms3d.euler.mat2euler(rotation_matrix, 'sxyz')
    rpy_vicon.append(rpy)
  plot_quat_vicon_precomputed(init, q_array_final, time, time_vicon, rpy_vicon, accelerometer_readings, str(i+1) + "-figure.png")

#Generating Panaroma for Training Set using estimate and VICON - This will run optimization again
print("Generating Panaroma Training Set")
panaroma_indices = [0,1,7,8]

# Now we start panaroma and spherical coordinates
horizontal_fov = 60 * np.pi / 180
vertical_fov   = 45 * np.pi / 180

image_height = 240
image_width = 320

min_horizontal_angle = - (horizontal_fov / 2)
max_horizontal_angle = (horizontal_fov / 2)
min_vertical_angle   = - (vertical_fov / 2)
max_vertical_angle   = (vertical_fov / 2)

for i in range(len(panaroma_indices)):
  print("File No:", (panaroma_indices[i]+1))
  accelerometer_readings, gyroscope_readings, time, vicon_readings, time_vicon = get_unbiased_imu_value(imu_data, vicon_data, panaroma_indices[i])
  delta_t = (time[1:] - time[:-1]).reshape(-1,1)
  init = find_initial_quaternion(delta_t.reshape(-1,1), gyroscope_readings[:-1])
  q_array_final = projected_gradient_descent(init, imu_data, index = panaroma_indices[i] ,alpha = 0.01)
  rpy_vicon = []
  quats_vicon = []
  for j in range(vicon_readings.shape[2]):
    rotation_matrix = vicon_readings[:,:,j]
    try:
      quat = transforms3d.quaternions.mat2quat(rotation_matrix)
    except:
      quats_vicon.append(quats_vicon[-1])
      #rpy = transforms3d.euler.mat2euler(rotation_matrix, 'sxyz')
      rpy_vicon.append(rpy_vicon[-1])
      continue
    quats_vicon.append(quat)
    rpy = transforms3d.euler.mat2euler(rotation_matrix, 'sxyz')
    rpy_vicon.append(rpy)
  final_image = create_panaroma(cam_data[i]['cam'], cam_data[i]['ts'], q_array_final, time, (720, 1080))
  cv2.imwrite(str(i+1)+"-panaroma-predicted.png", np.stack([final_image[:,:,2], final_image[:,:,1], final_image[:,:,0]], axis = 2))
  final_image = create_panaroma(cam_data[i]['cam'], cam_data[i]['ts'], quats_vicon, time_vicon, (720, 1080))
  cv2.imwrite(str(i+1)+"-panaroma-vicon.png", np.stack([final_image[:,:,2], final_image[:,:,1], final_image[:,:,0]], axis = 2))


print("Generating Graphs Test Set")
for i in range(len(imu_test_data)):
  print("File No:", (panaroma_indices[i]+1))
  accelerometer_readings, gyroscope_readings, time, vicon_readings, time_vicon = get_unbiased_imu_value(imu_test_data, vicon_data, i)
  delta_t = (time[1:] - time[:-1]).reshape(-1,1)
  init = find_initial_quaternion(delta_t.reshape(-1,1), gyroscope_readings[:-1])
  q_array_final = projected_gradient_descent(init, imu_test_data, index = i, alpha = 0.01)
  plot_quat_no_vicon_precomputed(init, q_array_final, time, accelerometer_readings, str(i+1) + "-test-figure.png")

print("Generating Panaroma Test Set")
for i in range(len(imu_test_data)):
  print(i)
  accelerometer_readings, gyroscope_readings, time, vicon_readings, time_vicon = get_unbiased_imu_value(imu_test_data, vicon_data, i)
  delta_t = (time[1:] - time[:-1]).reshape(-1,1)
  init = find_initial_quaternion(delta_t.reshape(-1,1), gyroscope_readings[:-1])
  q_array_final = projected_gradient_descent(init, imu_test_data, index = i ,alpha = 0.01)
  final_image = create_panaroma(cam_test_data[i]['cam'], cam_test_data[i]['ts'], q_array_final, time, (720, 1080))
  cv2.imwrite(str(i+1)+"-test-panaroma-predicted.png", np.stack([final_image[:,:,2], final_image[:,:,1], final_image[:,:,0]], axis = 2))