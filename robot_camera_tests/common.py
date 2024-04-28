import numpy as np
from numpy.linalg import inv, svd, norm, pinv
from scipy.spatial.transform import Rotation as Rot


def skew(x):
    x = x.ravel()
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

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