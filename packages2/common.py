import math
import numpy as np
from numpy.linalg import inv, svd, norm, pinv
from scipy.spatial.transform import Rotation as Rot
import cv2

def skew(x):
    x = x.ravel()
    return np.array([[0, -x[2], x[1]],
                    [x[2], 0, -x[0]],
                    [-x[1], x[0], 0]])

def skew2(x):
    y = cv2.Rodrigues(x)
    return y[0]

def solveSVD(A):
    U, S, VT = svd(A)
    x = VT.T[:,-1]
    return x

def get_RX_tX(X):
    _Rx = X[:9].reshape(3,3)
    _tX = X[9:]
    w = np.sign(np.linalg.det(_Rx)) / (np.abs(np.linalg.det(_Rx)) ** (1/3))
    Rx = w * _Rx
    tX = w * _tX
    return Rx.T, tX

def solveLS(A,B):
    u,s,v = svd(A)
    _s = inv(np.diag(s))
    _ss = np.zeros((3,u.shape[0]))
    _ss[:3,:3] = _s
    x = np.dot(np.dot(v.T,_ss),np.dot(u.T,B))
    return x

def R_2_angle_axis(R):
    rotvec = Rot.from_matrix(R).as_rotvec()
    theta = norm(rotvec)
    u = rotvec/theta
    return u.reshape(3,1), theta

def get_Translation(R,RA_I,TA,TB):
    RxTB = np.dot(R,TB[:3,0]).reshape(3,1)
    for i in range(1,int((TB.shape[0])/3)):
        RxTB = np.append(RxTB,np.dot(R,TB[i*3:(i+1)*3,0].reshape(3,1)),axis=0)
    T = RxTB - TA
    tX = np.dot(inv(np.dot(RA_I.T,RA_I)),np.dot(RA_I.T,T))
    tX = np.dot(pinv(RA_I),T)
    return tX

def generate_input_data(n):
    matrices = []
    for _ in range(n):
        matrix = np.random.rand(4, 4)
        matrix[3, :] = 0
        matrix[3, 3] = 1
        matrices.append(matrix)
    return matrices

def getRotation(Rx):
    '''Get the rotation matrix that satisfy othorgonality'''
    u,s,v = svd(Rx)
    return np.dot(u,v)

def RPY_to_rmtx(roll, pitch, yaw):
    alpha = yaw
    beta = pitch
    gamma = roll
    ca = math.cos(alpha)
    cb = math.cos(beta)
    cg = math.cos(gamma)
    sa = math.sin(alpha)
    sb = math.sin(beta)
    sg = math.sin(gamma)
    mtx = [[0 for _ in range(3)] for _ in range(3)]
    mtx[0][0] = ca*cb           # r11
    mtx[0][1] = ca*sb*sg-sa*cg  # r12
    mtx[0][2] = ca*sb*cg+sa*sg  # r13
    mtx[1][0] = sa*cb           # r21
    mtx[1][1] = sa*sb*sg+ca*cg  # r22
    mtx[1][2] = sa*sb*cg-ca*sg  # r23
    mtx[2][0] = -sb             # r31
    mtx[2][1] = cb*sg           # r32
    mtx[2][2] = cb*cg           # r33
    return mtx

def to_mtx(Rx, tX):
    result = np.zeros((4, 4))
    result[:3, :3] = Rx
    for i in range(3):
        result[i, 3] = tX[i]
    result[3, 3] = 1
    return result