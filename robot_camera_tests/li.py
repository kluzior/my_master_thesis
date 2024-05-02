import numpy as np
from common import skew, solveSVD, get_RX_tX, getRotation


def calibrate(A_list, B_list):
    N = len(A_list)
    I = np.eye(3)
    I9 = np.eye(9)

    S = None
    T = None

    for i in range(N):
        An = A_list[i]
        Bn = B_list[i]
        
        RA = An[:3,:3]
        tA = An[:3,3].reshape(3,1)
        tA_ = skew(tA)
        RB = Bn[:3,:3]
        tB = Bn[:3,3].reshape(3,1)

        S1 = np.append(I9 - np.kron(RB, RA), np.zeros((9,3)), axis=1)
        S2 = np.append(np.kron(tB.T, tA_), np.dot(tA_, (I - RA)), axis=1)

        _S = np.append(S1, S2, axis=0)
        _T = np.append(np.zeros((9,1)), tA.reshape(3,1), axis=0)
        
        S = np.append(S, _S, axis=0) if S is not None else _S
        T = np.append(T, _T, axis=0) if T is not None else _T

    Rx_tX = solveSVD(S)
    Rx, tX = get_RX_tX(Rx_tX)

    Rx = getRotation(Rx)

    return Rx, tX.reshape(3,1)
