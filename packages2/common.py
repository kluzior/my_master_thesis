import math

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

