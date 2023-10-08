import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler

def load_data(file_name):
    '''
    function to read visual features, IMU measurements, and calibration parameters
    Input:
        file_name: the input data file. Should look like "XX.npz"
    Output:
        t: time stamp
            with shape 1*t
        features: visual feature point coordinates in stereo images, 
            with shape 4*n*t, where n is number of features
        linear_velocity: velocity measurements in IMU frame
            with shape 3*t
        angular_velocity: angular velocity measurements in IMU frame
            with shape 3*t
        K: (left)camera intrinsic matrix
            with shape 3*3
        b: stereo camera baseline
            with shape 1
        imu_T_cam: extrinsic transformation from (left) camera to imu frame, in SE(3).
            with shape 4*4
    '''
    with np.load(file_name) as data:
    
        t = data["time_stamps"] # time_stamps
        features = data["features"] # 4 x num_features : pixel coordinates of the visual features
        linear_velocity = data["linear_velocity"] # linear velocity in body-frame coordinates
        angular_velocity = data["angular_velocity"] # angular velocity in body-frame coordinates
        K = data["K"] # intrinsic calibration matrix
        b = data["b"] # baseline
        imu_T_cam = data["imu_T_cam"] # transformation from left camera frame to imu frame 
        #(1, 3026) (4, 13289, 3026) (3, 3026) (3, 3026) (3, 3) 0.6 (4, 4)
    
    return t.T,np.transpose(features, (2,1,0)),linear_velocity.T,angular_velocity.T,K,b,imu_T_cam


def visualize_trajectory_2d(pose,path_name="Unknown",show_ori=False):
    '''
    function to visualize the trajectory in 2D
    Input:
        pose:   4*4*N matrix representing the camera pose, 
                where N is the number of poses, and each
                4*4 matrix is in SE(3)
    '''
    fig,ax = plt.subplots(figsize=(5,5))
    n_pose = pose.shape[2]
    ax.plot(pose[0,3,:],pose[1,3,:],'r-',label=path_name)
    ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
    ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
  
    if show_ori:
        select_ori_index = list(range(0,n_pose,max(int(n_pose/50), 1)))
        yaw_list = []
        
        for i in select_ori_index:
            _,_,yaw = mat2euler(pose[:3,:3,i])
            yaw_list.append(yaw)
    
        dx = np.cos(yaw_list)
        dy = np.sin(yaw_list)
        dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
        ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
            color="b",units="xy",width=1)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.grid(False)
    ax.legend()
    plt.show(block=True)

    return fig, ax

def visualize_trajectory_2d_feats(pose, features, path_name="Unknown",show_ori=False, fig_name = "temp.png"):
    '''
    function to visualize the trajectory in 2D
    Input:
        pose:   4*4*N matrix representing the camera pose, 
                where N is the number of poses, and each
                4*4 matrix is in SE(3)
    '''
    fig,ax = plt.subplots(figsize=(10,10))
    n_pose = pose.shape[2]
    ax.plot(pose[0,3,:],pose[1,3,:],'r-',label=path_name, color = "blue")
    if features is not None:
        ax.plot(features[:,0], features[:,1], 'ro', markersize = 0.5, label = "Landmark", color = "red")
    ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
    ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
  
    if show_ori:
        select_ori_index = list(range(0,n_pose,max(int(n_pose/50), 1)))
        yaw_list = []
        
        for i in select_ori_index:
            _,_,yaw = mat2euler(pose[:3,:3,i])
            yaw_list.append(yaw)
    
        dx = np.cos(yaw_list)
        dy = np.sin(yaw_list)
        dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
        ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
            color="b",units="xy",width=1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.grid(False)
    ax.legend()
    plt.savefig(fig_name)
    # plt.clf()
    plt.show(block=True)

    return fig, ax



# q function from the slides
def projection(ph):
    '''
    ph = n x 4 = homogeneous point coordinates
    r = n x 4 = ph/ph[...,2] = normalized z axis coordinates
    '''  
    return ph/ph[...,2,None]

# Jacobian of q
def projectionJacobian(ph):
    '''
    ph = n x 4 = homogeneous point coordinates
    J = n x 4 x 4 = Jacobian of ph/ph[...,2]
    '''  
    J = np.zeros(ph.shape+(4,))
    iph2 = 1.0/ph[...,2]
    ph2ph2 = ph[...,2]**2
    J[...,0,0], J[...,1,1],J[...,3,3] = iph2,iph2,iph2
    J[...,0,2] = -ph[...,0]/ph2ph2
    J[...,1,2] = -ph[...,1]/ph2ph2
    J[...,3,2] = -ph[...,3]/ph2ph2
    return J

def inversePose(T):
    '''
    @Input:
    T = n x 4 x 4 = n elements of SE(3)
    @Output:
    iT = n x 4 x 4 = inverse of T
    '''
    iT = np.empty_like(T)
    iT[...,0,0], iT[...,0,1], iT[...,0,2] = T[...,0,0], T[...,1,0], T[...,2,0] 
    iT[...,1,0], iT[...,1,1], iT[...,1,2] = T[...,0,1], T[...,1,1], T[...,2,1] 
    iT[...,2,0], iT[...,2,1], iT[...,2,2] = T[...,0,2], T[...,1,2], T[...,2,2]
    iT[...,:3,3] = -np.squeeze(iT[...,:3,:3] @ T[...,:3,3,None])
    iT[...,3,:] = T[...,3,:]
    return iT

def axangle2skew(a):
    '''
    converts an n x 3 axis-angle to an n x 3 x 3 skew symmetric matrix 
    '''
    S = np.empty(a.shape[:-1]+(3,3))
    S[...,0,0].fill(0)
    S[...,0,1] =-a[...,2]
    S[...,0,2] = a[...,1]
    S[...,1,0] = a[...,2]
    S[...,1,1].fill(0)
    S[...,1,2] =-a[...,0]
    S[...,2,0] =-a[...,1]
    S[...,2,1] = a[...,0]
    S[...,2,2].fill(0)
    return S

def axangle2twist(x):
    '''
    @Input:
    x = n x 6 = n elements of position and axis-angle
    @Output:
    T = n x 4 x 4 = n elements of se(3)
    '''
    T = np.zeros(x.shape[:-1]+(4,4))
    T[...,0,1] =-x[...,5]
    T[...,0,2] = x[...,4]
    T[...,0,3] = x[...,0]
    T[...,1,0] = x[...,5]
    T[...,1,2] =-x[...,3]
    T[...,1,3] = x[...,1]
    T[...,2,0] =-x[...,4]
    T[...,2,1] = x[...,3]
    T[...,2,3] = x[...,2]
    return T

def twist2axangle(T):
    '''
    converts an n x 4 x 4 twist (se3) matrix to an n x 6 axis-angle 
    '''
    return T[...,[0,1,2,2,0,1],[3,3,3,1,2,0]]

# Curly Hat
def axangle2adtwist(x):
    '''
    @Input:
    x = n x 6 = n elements of position and axis-angle
    @Output:
    A = n x 6 x 6 = n elements of ad(se(3))
    '''
    A = np.zeros(x.shape+(6,))
    A[...,0,1] =-x[...,5]
    A[...,0,2] = x[...,4]
    A[...,0,4] =-x[...,2]
    A[...,0,5] = x[...,1]

    A[...,1,0] = x[...,5]
    A[...,1,2] =-x[...,3]
    A[...,1,3] = x[...,2]
    A[...,1,5] =-x[...,0]

    A[...,2,0] =-x[...,4]
    A[...,2,1] = x[...,3]
    A[...,2,3] =-x[...,1]
    A[...,2,4] = x[...,0]

    A[...,3,4] =-x[...,5] 
    A[...,3,5] = x[...,4] 
    A[...,4,3] = x[...,5]
    A[...,4,5] =-x[...,3]   
    A[...,5,3] =-x[...,4]
    A[...,5,4] = x[...,3]
    return A

def twist2pose(T):
    '''
    converts an n x 4 x 4 twist (se3) matrix to an n x 4 x 4 pose (SE3) matrix 
    '''
    rotang = np.sqrt(np.sum(T[...,[2,0,1],[1,2,0]]**2,axis=-1)[...,None,None]) # n x 1
    Tn = np.nan_to_num(T / rotang)
    Tn2 = Tn@Tn
    Tn3 = Tn@Tn2
    eye = np.zeros_like(T)
    eye[...,[0,1,2,3],[0,1,2,3]] = 1.0
    return eye + T + (1.0 - np.cos(rotang))*Tn2 + (rotang - np.sin(rotang))*Tn3
  
def axangle2pose(x):
    '''
    @Input:
    x = n x 6 = n elements of position and axis-angle
    @Output:
    T = n x 4 x 4 = n elements of SE(3)
    '''
    return twist2pose(axangle2twist(x))

def pose2adpose(T):
    '''
    converts an n x 4 x 4 pose (SE3) matrix to an n x 6 x 6 adjoint pose (ad(SE3)) matrix 
    '''
    calT = np.empty(T.shape[:-2]+(6,6))
    calT[...,:3,:3] = T[...,:3,:3]
    calT[...,:3,3:] = axangle2skew(T[...,:3,3]) @ T[...,:3,:3]
    calT[...,3:,:3] = np.zeros(T.shape[:-2]+(3,3))
    calT[...,3:,3:] = T[...,:3,:3]
    return calT

def hat_3d(v):
    zeros = np.zeros_like(v[:, 2])
    arr = np.array([[zeros, - v[:, 2], v[:, 1]],
                    [v[:, 2], zeros, -v[:, 0]],
                    [-v[:, 1], v[:, 0], zeros]])
    arr = np.transpose(arr, (2,0,1))
    return arr

def circle_dot(v):
    """
    v : nx4
    """
    iden = np.zeros((3,3,v.shape[0]))
    iden[[0,1,2], [0,1,2], :] = 1
    arr = np.vstack((np.hstack((iden, - np.transpose(hat_3d(v[:,:3]), (1,2,0)))), np.zeros((1,6,v.shape[0]))))
    return np.transpose(arr, (2,0,1))

# def adtwist2adpose(T, rotang):
#     '''
#     converts an n x 6 x 6 twist (ad(SE3)) to an n x 6 x 6 adjoint pose (ad(SE3)) matrix 
#     '''
#     Tn = np.nan_to_num(T / rotang)
#     Tn2 = Tn@Tn
#     Tn3 = Tn@Tn2
#     Tn4 = Tn@Tn3
#     eye = np.zeros_like(T)
#     eye[...,[0,1,2,3,4,5],[0,1,2,3,4,5]] = 1.0
#     return eye + ((3*np.sin(rotang) - rotang * np.cos(rotang))/(2))*Tn + ((4 - rotang * np.sin(rotang) - 4 * np.cos(rotang))/(2))*Tn2 + ((np.sin(rotang) - rotang * np.cos(rotang))/(2))*Tn3 + ((2 - rotang*np.sin(rotang) - 2 * np.cos(rotang))/(2))*Tn4

# def exp_pose(pose):
#     '''
#     @Input:
#     pose = n x 4 x 4 = n elements of SE(3)
#     @Output:
#     exp_pose_ = n x 4 x 4 = n elements of SE(3) of exponential map
#     '''
#     angles_norm = np.linalg.norm(twist2axangle(pose), axis = 1).reshape(-1,1,1)
#     print("Angles_norm_mine", angles_norm)
#     n_identity = np.zeros_like(pose)
#     for i in range(4):
#         n_identity[:,i,i] = 1
#     pose_sq = pose @ pose
#     pose_cube = pose_sq @ pose
#     exp_pose_ = n_identity + pose + ((1-np.cos(angles_norm))/(angles_norm**2))*(pose_sq) + ((angles_norm - np.sin(angles_norm))/(angles_norm**3)) * pose_cube
#     return exp_pose_

# def exp_adjoint(adjoint):
#     '''
#     @Input:
#     adjoint = n x 6 x 6 = n elements of ad(SE(3))
#     @Output:
#     exp_adjoint_ = n x 6 x 6 = n elements of ad(SE(3)) of exponential map
#     '''
#     angles_norm = np.linalg.norm(twist2axangle(adjoint), axis = 1).reshape(-1,1,1)
#     n_identity = np.zeros_like(adjoint)
#     for i in range(6):
        # n_identity[:,i,i] = 1
    

def linangtou(linear_velocity, angular_velocity):
    '''
    @Input:
    linear_velocity = n x 3 = Linear Velocity
    angular_velocity = n x 3 = Angular Velocity
    @Output:
    u = concatenated vector for twist calculation
    '''
    return np.hstack((linear_velocity, angular_velocity))
