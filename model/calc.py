import numpy as np
from . import operators as op


def dos(model,E_fineness=1000,sigma2 = 0.0001):

    print("dos calculation start")
    model.E = np.linspace(np.min(model.enes)-0.1,np.max(model.enes)+0.1,E_fineness)
    model.dos = np.array([])

    for e in model.E:
        model.dos = np.append(
            model.dos,np.sum(np.exp(-(e-model.enes)**2 / 2 / sigma2 ) / np.sqrt(2 * np.pi * sigma2)))
    del e

    model.dos /= np.sum(model.dos)*(model.E[1]-model.E[0])

    print("dos calculation finished")

    return


def kF_index(model):
    """
    フェルミ面のインデックスを計算する
    """
    if(model.kF_index.size != 3):
        return

    print("kF index calculation start")

    for i in range(model.k_mesh):
        for j in range(model.k_mesh):
            for m in range(model.n_orbit*2):
                candidate_kF_index  = np.array([i,j,m])
                ene_ij = model.enes[i,j,m]
                # 八方で確かめる
                if(i < model.k_mesh-1): # 南方向
                    ene_delta = model.enes[i+1,j,m]
                    if((ene_ij-model.ef)*(ene_delta-model.ef) < 0
                        and np.abs(ene_ij-model.ef) < np.abs(ene_delta-model.ef)):
                        model.kF_index = np.vstack((model.kF_index,candidate_kF_index))
                        continue

                if(j < model.k_mesh-1): # 東方向
                    ene_delta = model.enes[i,j+1,m]
                    if((ene_ij-model.ef)*(ene_delta-model.ef) < 0
                        and np.abs(ene_ij-model.ef) < np.abs(ene_delta-model.ef)):
                        model.kF_index = np.vstack((model.kF_index,candidate_kF_index ))
                        continue

                if(i > 0): # 北方向
                    ene_delta = model.enes[i-1,j,m]
                    if((ene_ij-model.ef)*(ene_delta-model.ef) < 0
                        and np.abs(ene_ij-model.ef) < np.abs(ene_delta-model.ef)):
                        model.kF_index = np.vstack((model.kF_index,candidate_kF_index ))
                        continue

                if(j > 0): # 西方向
                    ene_delta = model.enes[i,j-1,m]
                    if((ene_ij-model.ef)*(ene_delta-model.ef) < 0
                        and np.abs(ene_ij-model.ef) < np.abs(ene_delta-model.ef)):
                        model.kF_index = np.vstack((model.kF_index,candidate_kF_index ))
                        continue

                if(i < model.k_mesh-1 and j < model.k_mesh-1): # 南東方向
                    ene_delta = model.enes[i+1,j+1,m]
                    if((ene_ij-model.ef)*(ene_delta-model.ef) < 0
                        and np.abs(ene_ij-model.ef) < np.abs(ene_delta-model.ef)):
                        model.kF_index = np.vstack((model.kF_index,candidate_kF_index ))
                        continue

                if(i > 0 and j < model.k_mesh-1): # 北東方向
                    ene_delta = model.enes[i-1,j+1,m]
                    if((ene_ij-model.ef)*(ene_delta-model.ef) < 0
                        and np.abs(ene_ij-model.ef) < np.abs(ene_delta-model.ef)):
                        model.kF_index = np.vstack((model.kF_index,candidate_kF_index))
                        continue

                if(i > 0 and j > 0): # 北西方向
                    ene_delta = model.enes[i-1,j-1,m]
                    if((ene_ij-model.ef)*(ene_delta-model.ef) < 0
                        and np.abs(ene_ij-model.ef) < np.abs(ene_delta-model.ef)):
                        model.kF_index = np.vstack((model.kF_index,candidate_kF_index))
                        continue

                if(i < model.k_mesh-1 and j > 0): # 南西方向
                    ene_delta = model.enes[i+1,j-1,m]
                    if((ene_ij-model.ef)*(ene_delta-model.ef) < 0
                        and np.abs(ene_ij-model.ef) < np.abs(ene_delta-model.ef)):
                        model.kF_index = np.vstack((model.kF_index,candidate_kF_index))
                        continue
    del i,j,m

    model.kF_index = np.delete(model.kF_index,0,0)
    print("kF index calculation finished\n")
    return


def spin_conductivity(model,mu,nu,omega=0,gamma=0.0001):
    """スピン伝導度の計算

    Args:
        mu (str, optional): スピン流れの流れる方向. Defaults to "x".
        nu (str, optional): 電場を加える方向. Defaults to "y".
        gamma (float, optional): ダンピングファクター. Defaults to 0.0001.

    Returns:
        complex: 複素伝導度が帰ってくる

    Remark:
        カレントに自己無撞着に決めるパラメータがあるときには気を付けること
    """
    print("SpinConductivity calculation start.")

    # スピン伝導度 複素数として初期化
    chi = 0.0 + 0.0*1j
    chis = np.zeros((model.k_mesh, model.k_mesh), np.complex128)

    # ブリュアンゾーンのメッシュの生成
    kx,ky = model._gen_kmesh()

    for i in range(model.k_mesh):
        for j in range(model.k_mesh):

            chi_ij = 0.0 + 0.0*1j
            Jmu_matrix = np.conjugate(model.eigenStates[i,j].T) @ op.SpinCurrent(kx[i,j],ky[i,j],mu) @ model.eigenStates[i,j]
            Jnu_matrix = np.conjugate(model.eigenStates[i,j].T) @ op.Current(kx[i,j],ky[i,j],nu) @ model.eigenStates[i,j]

            for m in range(model.n_orbit*2):
                for n in range(model.n_orbit*2):

                    Jmu = Jmu_matrix[m,n]
                    Jnu = Jnu_matrix[n,m]

                    # バンド間遷移 (van Vleck 項)
                    if(np.abs(model.enes[i,j][m]-model.enes[i,j][n])>1e-6):

                        efm = fermi_dist(model.enes[i,j][m],model.ef, 1000)
                        efn = fermi_dist(model.enes[i,j][n],model.ef, 1000)

                        add_chi = Jmu * Jnu * (efm - efn) / (
                            (model.enes[i,j][m]-model.enes[i,j][n])*(model.enes[i,j][m]-model.enes[i,j][n] + omega + 1j*gamma))
                        chi_ij += add_chi
                        chi += add_chi

                    # バンド内遷移
                    else:
                        # フェルミ分布の微分
                        f_diff = (fermi_dist_diff(model.enes[i,j][m],model.ef)
                                  +fermi_dist_diff(model.enes[i,j][n],model.ef))/2

                        add_chi = Jmu * Jnu * f_diff / (omega + 1j*gamma)
                        chi_ij += add_chi
                        chi += add_chi

            chis[i,j] = chi_ij

    chi /= (model.k_mesh*model.k_mesh*1j)
    chis /= (model.k_mesh*model.k_mesh*1j)

    munu = mu + nu
    if (munu == "xx"):
        model.chi_xx = chis
    elif (munu == "yy"):
        model.chi_yy = chis
    elif (munu == "xy"):
        model.chi_xy = chis
    elif (munu == "yx"):
        model.chi_yx = chis

    print("Spin Conductivity calculation finished")
    print("ReChi = {:1.2e}, ImChi = {:1.2e}\n".format(np.real(chi),np.imag(chi)))

    return chi

def fermi_dist(ene,ef: float,beta: float=1000):
    # ene が数字でも配列でも numpy 配列に変換
    a = np.array(beta*(ene-ef))
    # オーバーフローを防ぐ
    a = np.clip(a, -700, 700)

    return 1/(np.exp(a)+1)


def fermi_dist_diff(ene,ef: float,beta: float=1000):
    # ene が数字でも配列でも numpy 配列に変換
    a = np.array(beta*(ene-ef))
    # オーバーフローを防ぐ
    a = np.clip(a, -700, 700)

    return -beta/(2*np.cosh(a/2))**2
