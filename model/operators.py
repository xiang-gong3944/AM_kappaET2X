import scipy.linalg
import numpy as np


# ホッピングパラメータ
# 各ホッピングの果たしている役割を見るために t = 0 としてみたい
ta = -0.207     # eV
tb = -0.067     # eV
tp = -0.102     # eV
tq = 0.043      # eV

# 軌道の数
n_orbit = 4

def Hamiltonian(kx, ky, Delta, U=0.0):
    """ある波数のでのハミルトニアン
    Args:
        (float) kx: 波数のx成分
        (float) ky: 波数のy成分
        (float) U: オンサイト相互作用の強さ
        (float) delta: 反強磁性分子場の強さ

    Returns:
        ハミルトニアンの固有値[0]と固有ベクトルの行列[1]
    """

    # ホッピング項
    H = np.zeros((8,8), dtype=np.complex128)
    H[0,1] = ta + tb*np.exp(-1j*kx)                          # A1up   from A2up
    H[0,2] = tq * (1 + np.exp(1j*ky))                        # A1up   from B1up
    H[0,3] = tp * np.exp(1j*ky) * (1 + np.exp(1j*kx))        # A1up   from B2up

    H[1,2] = tp * (1 + np.exp(1j*kx))                      # A2up   from B1up
    H[1,3] = tq * np.exp(1j*kx) * (1 + np.exp(1j*ky))      # A2up   from B2up

    H[2,3] = ta + tb*np.exp(1j*kx)                         # B1up   from B2up

    H[4,5] = H[0,1]                                         # A1down from A2down
    H[4,6] = H[0,2]                                         # A1down from B1down
    H[4,7] = H[0,3]                                         # A1down from B2down

    H[5,6] = H[1,2]                                         # A2down from B1down
    H[5,7] = H[1,3]                                         # A2down from B2down

    H[6,7] = H[2,3]                                         # B1down from B2down

    #エルミート化
    for i in range(1,8):
        for j in range(0, i):
            H[i][j] = H[j][i].conjugate()
    del i, j

    # 反強磁性分子内磁場を表すハートリー項
    H[0,0] = - U * Delta / 4    # A1 up
    H[1,1] = - U * Delta / 4    # A2 up
    H[2,2] = + U * Delta / 4    # B1 up
    H[3,3] = + U * Delta / 4    # B2 up
    H[4,4] = + U * Delta / 4    # A1 down
    H[5,5] = + U * Delta / 4    # A2 down
    H[6,6] = - U * Delta / 4    # B1 down
    H[7,7] = - U * Delta / 4    # B2 down

    return scipy.linalg.eigh(H)


def Current(kx, ky, mu):
    """ある波数での電流演算子行列

    Args:
        kx (float): 波数のx成分
        ky (float): 波数のy成分
        mu (str): 電流の方向. "x", "y", "z" のみ受け付ける

    Return:
        J (ndarray): 8x8の電流演算子行列
    """

    if (mu == "x"):
        J = np.zeros((8,8), dtype=np.complex128)

        J[0,1] =-1j * tb * np.exp(-1j*kx)                           # A1up   from A2up
        J[0,3] = 1j * tp * np.exp(1j*(kx+ky))                       # A1up   from B2up

        J[1,2] = 1j * tp * np.exp(1j*kx)                            # A2up   from B1up
        J[1,3] = 1j * tq * np.exp(1j*kx) * (1 + np.exp(1j*ky))      # A2up   from B2up

        J[2,3] = 1j * tb*np.exp(1j*kx)                              # B1up   from B2up

        J[4,5] = J[0,1]                                         # A1down from A2down
        J[4,7] = J[0,3]                                         # A1down from B2down

        J[5,6] = J[1,2]                                         # A2down from B1down
        J[5,7] = J[1,3]                                         # A2down from B2down

        J[6,7] = J[2,3]                                         # B1down from B2down

    elif (mu == "y"):
        J = np.zeros((8,8), dtype=np.complex128)

        J[0,2] = 1j * tq * np.exp(1j*ky)                             # A1up   from B1up
        J[0,3] = 1j * tp * np.exp(1j*ky) * (1 + np.exp(1j*kx))      # A1up   from B2up

        J[1,3] = 1j * tq * np.exp(1j*(kx + ky))                      # A2up   from B2up

        J[4,6] = J[0,2]                                         # A1down from B1down
        J[4,7] = J[0,3]                                         # A1down from B2down

        J[5,7] = J[1,3]                                         # A2down from B2down


    elif (mu == "z"):
        J = np.zeros((8,8), dtype=np.complex128)

    else :
        print("The current direction is incorrect.")
        return

    #エルミート化
    for i in range(1,8):
        for j in range(0, i):
            J[i][j] = J[j][i].conjugate()
    del i, j

    return -J

def SpinCurrent(kx, ky, mu):
    """ある波数での電流演算子行列

    Args:
        kx (float): 波数のx成分
        ky (float): 波数のy成分
        mu (str): 電流の方向. "x", "y", "z" のみ受け付ける

    Return:
        J (ndarray): 8x8の電流演算子行列
    """

    if (mu == "x"):
        J = np.zeros((8,8), dtype=np.complex128)

        J[0,1] =-1j * tb * np.exp(-1j*kx)                           # A1up   from A2up
        J[0,3] = 1j * tp * np.exp(1j*(kx+ky))                       # A1up   from B2up

        J[1,2] = 1j * tp * np.exp(1j*kx)                            # A2up   from B1up
        J[1,3] = 1j * tq * np.exp(1j*kx) * (1 + np.exp(1j*ky))      # A2up   from B2up

        J[2,3] = 1j * tb*np.exp(1j*kx)                              # B1up   from B2up

        J[4,5] =-J[0,1]                                         # A1down from A2down
        J[4,7] =-J[0,3]                                         # A1down from B2down

        J[5,6] =-J[1,2]                                         # A2down from B1down
        J[5,7] =-J[1,3]                                         # A2down from B2down

        J[6,7] =-J[2,3]                                         # B1down from B2down

    elif (mu == "y"):
        J = np.zeros((8,8), dtype=np.complex128)

        J[0,2] = 1j * tq * np.exp(1j*ky)                             # A1up   from B1up
        J[0,3] = 1j * tp * np.exp(1j*ky) * (1 + np.exp(1j*kx))      # A1up   from B2up

        J[1,3] = 1j * tq * np.exp(1j*(kx + ky))                      # A2up   from B2up

        J[4,6] =-J[0,2]                                         # A1down from B1down
        J[4,7] =-J[0,3]                                         # A1down from B2down

        J[5,7] =-J[1,3]                                         # A2down from B2down


    elif (mu == "z"):
        J = np.zeros((8,8), dtype=np.complex128)

    else :
        print("The current direction is incorrect.")
        return

    #エルミート化
    for i in range(1,8):
        for j in range(0, i):
            J[i][j] = J[j][i].conjugate()
    del i, j

    return J/2
