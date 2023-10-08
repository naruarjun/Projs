import numpy as np
from pr3_utils import *
np.set_printoptions(suppress=True, precision=3)

class ekf:
    def __init__(self, covariance, W):
        self.mean_pose = np.eye(4)
        self.covariance_pose = covariance
        self.W = W
        self.clock = 0
    
    def predict(self, u, timestamp):
        tau = timestamp - self.clock
        self.mean_pose = self.mean_pose @ (twist2pose(tau * axangle2twist(u.reshape(1,6)))[0])
        F_J = pose2adpose(twist2pose(- tau * axangle2twist(u.reshape(1,6))))[0]
        eigvals = np.linalg.eigvals(F_J @  self.covariance_pose @ F_J.T)
        if np.all(eigvals>=0) != True:
            print(eigvals)
        self.covariance_pose = F_J @  self.covariance_pose @ F_J.T + self.W
        self.clock = timestamp

class ekf_landmark:
    def __init__(self, K_s, o_T_i, V):
        self.mean_landmarks = []
        self.mean_landmarks_homogenised = []
        self.covariance_landmarks = []
        self.V = V
        self.num_landmarks = 0
        self.K_s = K_s
        self.o_T_i = o_T_i
        self.P = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0]])
    
    def update(self, T, visibility_matrix, features):
        # Extracting valid features
        valid_features = features[visibility_matrix]

        
        
        # Jacobian Computation
        self.H, camera_observations, num_visible_landmarks = self.compute_jacobian_observation(T, visibility_matrix)

        
        
        # Kalman Gain
        # w_1, _ = np.linalg.eig(self.H @ self.covariance_landmarks @ (self.H.T))
        # print("W1", w_1[w_1<0])
        # w_2, _ = np.linalg.eig(np.kron(self.V, np.eye(num_visible_landmarks)))
        # print("W2", w_2[w_2<0])
        kalman_gain = self.covariance_landmarks @ (self.H.T) @ np.linalg.inv((self.H @ self.covariance_landmarks @ (self.H.T) + np.kron(self.V, np.eye(num_visible_landmarks))))
        
        # Mean Update
        self.mean_landmarks = (self.mean_landmarks.reshape(-1) + kalman_gain @ (valid_features.reshape(-1) - camera_observations.reshape(-1))).reshape(-1,3)
        
        
        self.mean_landmarks_homogenised[:,:3] = self.mean_landmarks
        
        #Covariance update
        self.covariance_landmarks = (np.eye(self.covariance_landmarks.shape[0]) - (kalman_gain @ self.H)) @ self.covariance_landmarks

    def conv_pose_to_imu(self, T):
        return self.o_T_i @ np.linalg.inv(T) #(inversePose(T.reshape(1,4,4))[0])
    
    def compute_jacobian_observation(self, T, visibility_matrix):
        common_term = self.conv_pose_to_imu(T) # iTo T^-1
        another_common_term = (common_term @ self.mean_landmarks_homogenised.T).T # iTo T^-1 m_
        camera_observations =  (self.K_s @ (projection(another_common_term).T)).T # K_s iTo T^-1 m_



        num_valid_indices = visibility_matrix.sum()
        self.H = np.zeros((4*num_valid_indices, 3*self.num_landmarks))
        j=0
        for i in range(self.num_landmarks):
            if visibility_matrix[i]:
                self.H[4*j:4*j+4, 3*i:3*i+3] = self.K_s @ (projectionJacobian(another_common_term[i].reshape(1,4))[0]) @ common_term @ (self.P.T)
                j+=1
        camera_observations = camera_observations[visibility_matrix[:self.num_landmarks]]
        return self.H, camera_observations, j
    
    def initialize(self, position):
        first = False
        if self.num_landmarks==0:
            first = True

        if first==True:
            self.mean_landmarks = position
            if len(position.shape) == 1:
                self.num_landmarks += 1
                self.mean_landmarks_homogenised = np.ones(position.shape[0]+1)
                self.mean_landmarks_homogenised[:3] = position
                self.covariance_landmarks = np.eye(position.shape[0])*100
            else:
                self.num_landmarks += position.shape[0]
                self.mean_landmarks_homogenised = np.ones((position.shape[0], position.shape[1]+1))
                self.mean_landmarks_homogenised[:,:3] = position
                self.covariance_landmarks = np.eye(position.shape[0]*position.shape[1])*100
        else:
            self.mean_landmarks = np.vstack((self.mean_landmarks, position))
            if len(position.shape) == 1:
                self.num_landmarks += 1
                new_landmark_homogenised = np.ones(position.shape[0]+1)
                new_landmark_homogenised[:3] = position
                self.mean_landmarks_homogenised = np.vstack((self.mean_landmarks_homogenised, new_landmark_homogenised))
                old_covar = self.covariance_landmarks.copy()
                self.covariance_landmarks = np.eye(self.covariance_landmarks.shape[0] + position.shape[0])*100
                self.covariance_landmarks[0:old_covar.shape[0], 0:old_covar.shape[1]] = old_covar

            else:
                self.num_landmarks += position.shape[0]
                new_landmark_homogenised = np.ones((position.shape[0], position.shape[1]+1))
                new_landmark_homogenised[:,:3] = position
                self.mean_landmarks_homogenised = np.vstack((self.mean_landmarks_homogenised, new_landmark_homogenised))
                old_covar = self.covariance_landmarks.copy()
                self.covariance_landmarks = np.eye(self.covariance_landmarks.shape[0] + position.shape[1]*position.shape[0])*100
                self.covariance_landmarks[0:old_covar.shape[0], 0:old_covar.shape[1]] = old_covar

    def compute_timestep(self, imu_poses, features, coordinates, visibility_matrix, index):
        current_imu_pose = imu_poses[index]
        current_features = features[index]
        current_visibility = visibility_matrix[index]
        current_coordinates_temp = coordinates[index]
        current_coordinates = np.ones((current_coordinates_temp.shape[0], current_coordinates_temp.shape[1]+1))
        current_coordinates[:,:3] = current_coordinates_temp
        world_coordinates = (current_imu_pose @ current_coordinates.T).T[:,:3]
        old_landmark_numbers = self.num_landmarks
        if self.num_landmarks < len(current_visibility):
            if current_visibility[self.num_landmarks:].sum()!=0:
                self.initialize(world_coordinates[self.num_landmarks:][current_visibility[self.num_landmarks:]])
        current_visibility[old_landmark_numbers:] = False
        self.update(current_imu_pose, current_visibility, current_features)
        w = np.linalg.eigvals(self.covariance_landmarks)
        if np.all(w>0) != True:
            print("Invalid Covar")

class combined_ekf:
    def __init__(self, covariance_pose, covariance_landmark, K_s, o_T_i, W, V):
        self.mean_pose = np.eye(4)
        self.W = W
        self.clock = 0
        self.means_homogenised = []
        self.covariance = covariance_pose
        self.covariance_landmark = covariance_landmark
        self.V = V
        self.num_landmarks = 0
        self.K_s = K_s
        self.o_T_i = o_T_i
        self.P = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0]])
        self.mean_landmarks = []
        self.mean_landmarks_homogenised = []
    
    def predict(self, u, timestamp):
        tau = timestamp - self.clock
        self.mean_pose = self.mean_pose @ (twist2pose(tau * axangle2twist(u.reshape(1,6)))[0])
        F_J = pose2adpose(twist2pose(-tau * axangle2twist(u.reshape(1,6))))[0]
        self.covariance[:6,:6] = (F_J @  self.covariance[:6,:6] @ (F_J.T)) + self.W
        self.clock = timestamp
    
    def update(self, visibility_matrix, features, num_landmarks_to_update):
        # Extracting valid features
        valid_features = features[visibility_matrix]

        # print("Valid", valid_features)
        

        # Jacobian Computation
        self.H, camera_observations, num_visible_landmarks = self.compute_jacobian_observation(self.mean_pose, visibility_matrix)

        # print("Camera", camera_observations)

        # Kalman Gain
        # eigvals = np.linalg.eigvals(self.H @ self.covariance @ (self.H.T))
        # if np.all(eigvals>=0)!= True:
        #     print(self.H.shape, eigvals, (self.H @ self.covariance @ (self.H.T)).shape)
        kalman_gain = self.covariance @ (self.H.T) @ np.linalg.inv((self.H @ self.covariance @ (self.H.T) + np.kron(np.eye(num_visible_landmarks), self.V)))
        
        # Kalman Gain x innovation
        update_term = (kalman_gain @ (valid_features.reshape(-1) - camera_observations.reshape(-1))).reshape(-1)

        # Mean Update
        self.mean_pose = self.mean_pose @ twist2pose(axangle2twist(update_term[:6].reshape(1,6))[0])
        self.mean_landmarks = (self.mean_landmarks.reshape(-1) + update_term[6:].reshape(-1)).reshape(-1,3)
        
        
        self.mean_landmarks_homogenised[:,:3] = self.mean_landmarks
        
        #Covariance update
        self.covariance = (np.eye(self.covariance.shape[0]) - (kalman_gain @ self.H)) @ self.covariance

    def conv_pose_to_imu(self, T):
        return self.o_T_i @ np.linalg.inv(T) #(inversePose(T.reshape(1,4,4))[0])
    
    def compute_jacobian_observation(self, T, visibility_matrix):
        common_term = self.conv_pose_to_imu(T) # iTo T^-1
        another_common_term = (common_term @ self.mean_landmarks_homogenised.T).T # iTo T^-1 m_
        pose_jacobian_term = (np.linalg.inv(T) @ self.mean_landmarks_homogenised.T).T
        # pose_jacobian_term = (np.linalg.inv(self.o_T_i) @ (another_common_term.T)).T
        camera_observations =  (self.K_s @ (projection(another_common_term).T)).T # K_s iTo T^-1 m_
        num_valid_indices = visibility_matrix.sum()
        self.H = np.zeros((4*num_valid_indices, 3*self.num_landmarks + 6))
        j=0
        for i in range(self.num_landmarks):
            if visibility_matrix[i]:
                self.H[4*j:4*j+4, :6] = - self.K_s @ (projectionJacobian(another_common_term[i].reshape(1,4))[0]) @ self.o_T_i @ (circle_dot(pose_jacobian_term[i].reshape(1,4))[0])
                self.H[4*j:4*j+4, 3*i + 6 : 3*i + 6 + 3] = self.K_s @ (projectionJacobian(another_common_term[i].reshape(1,4))[0]) @ common_term @ (self.P.T) # common_term_jacobian @ common_term @ (self.P.T)
                j+=1
        camera_observations = camera_observations[visibility_matrix[:self.num_landmarks]]
        return self.H, camera_observations, j
    
    def initialize(self, position):
        first = False
        if self.num_landmarks==0:
            first = True

        if first==True:
            self.mean_landmarks = position
            if len(position.shape) == 1:
                self.num_landmarks += 1
                self.mean_landmarks_homogenised = np.ones(position.shape[0]+1)
                self.mean_landmarks_homogenised[:3] = position
                old_covar = self.covariance.copy()
                self.covariance = np.eye(position.shape[0] + 6)*0
                self.covariance[:6,:6] = old_covar[:6,:6]
                self.covariance[6:,6:] = self.covariance_landmark
            else:
                self.num_landmarks += position.shape[0]
                self.mean_landmarks_homogenised = np.ones((position.shape[0], position.shape[1]+1))
                self.mean_landmarks_homogenised[:,:3] = position
                old_covar = self.covariance.copy()
                self.covariance = np.eye(position.shape[0]*position.shape[1] + old_covar.shape[0])*0
                self.covariance[:old_covar.shape[0],:old_covar.shape[1]] = old_covar.copy()
                covariance_new_landmarks = np.kron(np.eye(position.shape[0]), self.covariance_landmark) # np.kron(self.covariance_landmark, np.eye(position.shape[0]))
                self.covariance[old_covar.shape[0]:, old_covar.shape[1]:] = covariance_new_landmarks
        else:
            self.mean_landmarks = np.vstack((self.mean_landmarks, position))
            if len(position.shape) == 1:
                self.num_landmarks += 1
                new_landmark_homogenised = np.ones(position.shape[0]+1)
                new_landmark_homogenised[:3] = position
                self.mean_landmarks_homogenised = np.vstack((self.mean_landmarks_homogenised, new_landmark_homogenised))
                old_covar = self.covariance.copy()
                self.covariance = np.eye(old_covar.shape[0] + position.shape[0])*0
                self.covariance[:old_covar.shape[0],:old_covar.shape[1]] = old_covar
                self.covariance[old_covar.shape[0]:, old_covar.shape[1]:] = self.covariance_landmark # self.covariance_landmark
            else:
                self.num_landmarks += position.shape[0]
                new_landmark_homogenised = np.ones((position.shape[0], position.shape[1]+1))
                new_landmark_homogenised[:,:3] = position
                self.mean_landmarks_homogenised = np.vstack((self.mean_landmarks_homogenised, new_landmark_homogenised))
                old_covar = self.covariance.copy()
                self.covariance = np.eye(position.shape[0]*position.shape[1] + old_covar.shape[0])*0
                self.covariance[:old_covar.shape[0],:old_covar.shape[1]] = old_covar
                covariance_new_landmarks = np.kron(np.eye(position.shape[0]), self.covariance_landmark) # np.kron(self.covariance_landmark, np.eye(position.shape[0]))
                self.covariance[old_covar.shape[0]:, old_covar.shape[1]:] = covariance_new_landmarks

    def compute_timestep(self, features, coordinates, visibility_matrix, linear_velocity, angular_velocity, t, index):
        old_covar = self.covariance.copy()
        if index > 0:
            self.predict(linangtou((linear_velocity[index]).reshape(1,-1), (angular_velocity[index]).reshape(1,-1)), t[index])
        # print("Pose", self.mean_pose)
        current_features = features[index]
        current_visibility = visibility_matrix[index]
        current_coordinates_temp = coordinates[index]
        current_coordinates = np.ones((current_coordinates_temp.shape[0], current_coordinates_temp.shape[1]+1))
        current_coordinates[:,:3] = current_coordinates_temp
        world_coordinates = (self.mean_pose @ current_coordinates.T).T[:,:3]
        old_landmark_numbers = self.num_landmarks
        if self.num_landmarks < len(current_visibility):
            if current_visibility[self.num_landmarks:].sum()!=0:
                self.initialize(world_coordinates[self.num_landmarks:][current_visibility[self.num_landmarks:]])
        current_visibility[old_landmark_numbers:] = False
        # diags = np.diagonal(self.covariance[:6,:6]) - np.diagonal(old_covar[:6,:6])
        # print("Robot Pose", np.all(diags>=0))
        # if np.all(diags>=0) != True:
        #     print("Covar Decrease")
        #     exit()
        self.update(current_visibility, current_features, old_landmark_numbers)
        # w = np.linalg.eigvals(self.covariance)
        # if np.all(w>=0) != True:
        #     print("Invalid Covar")
        #     print(w)
        # if check_symmetric(self.covariance) != True:
        #     print("Non Symmetric")
        #     print(self.covariance)
        #     exit()

def check_symmetric(a, rtol=1e-02, atol=1e-02):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)