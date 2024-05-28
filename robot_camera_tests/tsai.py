import numpy as np
from numpy.linalg import norm
from common import skew, solveLS, R_2_angle_axis, get_Translation, skew2

def calibrate(A_list, B_list):
    N = len(A_list)
    
    S = None
    RA_I = None
    T = None
    TA = None
    TB = None

    for i in range(N):
        An = A_list[i]
        Bn = B_list[i]

        RA = An[:3,:3]
        tA = An[:3,3].reshape(3,1)
        tA_ = skew2(tA)
        RB = Bn[:3,:3]
        tB = Bn[:3,3].reshape(3,1)

        # Rotation matrix to angle(w) - axis(u) convertion
        uA, wA = R_2_angle_axis(RA)
        uB, wB = R_2_angle_axis(RB)

        _S = skew2(uA + uB)
        _T = uB - uA 

        _RA_I = RA - np.eye(3)
        _TA = tA
        _TB = tB
        
        S = np.append(S, _S, axis=0) if S is not None else _S
        T = np.append(T, _T, axis=0) if T is not None else _T
        
        RA_I = np.append(RA_I,_RA_I,axis=0) if RA_I is not None else _RA_I
        TA = np.append(TA,_TA,axis=0) if TA is not None else _TA
        TB = np.append(TB,_TB,axis=0) if TB is not None else _TB


    ux = solveLS(S,T)    
    uX = 2*ux/(np.sqrt(1+norm(ux)**2))
    
    Rx = (1-norm(uX)**2/2)*np.eye(3) + 0.5*(uX*uX.T + np.sqrt(4-norm(uX)**2)*skew2(uX))
    
    tX = get_Translation(Rx,RA_I,TA,TB)

    return Rx, tX.reshape(3,1)
