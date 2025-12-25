import numpy as np
from SCA_DEM_ani import dem_ani
# ===========================================================
# 基础工具：ku2c, Kronecker delta, Voigt <-> 四阶张量转换
# ===========================================================

def ku2c(K, mu):
    """
    等效各向同性刚度矩阵 6x6
    输入: K, mu (GPa)
    输出: C (6,6)
    """
    C = np.zeros((6, 6), dtype=float)
    lam = K - 2.0 * mu / 3.0

    C[0, 0] = lam + 2 * mu
    C[1, 1] = C[0, 0]
    C[2, 2] = C[0, 0]

    C[0, 1] = lam
    C[0, 2] = lam
    C[1, 2] = lam
    C[1, 0] = lam
    C[2, 0] = lam
    C[2, 1] = lam

    C[3, 3] = mu
    C[4, 4] = mu
    C[5, 5] = mu

    return C


_voigt_pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

def ku2c_iso(K, G):
    """
    K, G 是标量（单位：Pa），返回 6x6 刚度矩阵 C（各向同性各向同性）
    Voigt 顺序: 11,22,33,23,13,12
    """
    lam = K - 2.0 * G / 3.0
    C = np.zeros((6, 6), dtype=float)

    # 正应力块
    C[0, 0] = lam + 2 * G
    C[1, 1] = lam + 2 * G
    C[2, 2] = lam + 2 * G

    C[0, 1] = lam
    C[0, 2] = lam
    C[1, 0] = lam
    C[1, 2] = lam
    C[2, 0] = lam
    C[2, 1] = lam

    # 剪切
    C[3, 3] = G
    C[4, 4] = G
    C[5, 5] = G

    return C

def c22c4(C6):
    """
    6x6 Voigt 刚度矩阵 -> 3x3x3x3 四阶张量
    尽量保持与 MATLAB 版 c22c4 一致的简单映射。
    """
    C6 = np.asarray(C6)
    C4 = np.zeros((3, 3, 3, 3), dtype=float)
    for a, (i, j) in enumerate(_voigt_pairs):
        for b, (k, l) in enumerate(_voigt_pairs):
            val = C6[a, b]
            C4[i, j, k, l] = val
            C4[j, i, k, l] = val
            C4[i, j, l, k] = val
            C4[j, i, l, k] = val
    return C4


def c42c2(C4):
    """
    3x3x3x3 四阶张量 -> 6x6 Voigt 刚度矩阵
    """
    C4 = np.asarray(C4)
    C6 = np.zeros((6, 6), dtype=float)
    for a, (i, j) in enumerate(_voigt_pairs):
        for b, (k, l) in enumerate(_voigt_pairs):
            C6[a, b] = C4[i, j, k, l]
    return C6


def mydelta(i, j):
    """Kronecker delta"""
    return 1.0 if i == j else 0.0


# ===========================================================
# 1. VRH: Voigt-Reuss-Hill 平均
# ===========================================================

def VRH(x, f):
    """
    MATLAB: [m_vrh] = VRH(M)
        x = M.vrh.m (N1,N2,N3)
        f = M.vrh.f (N1,N3)

    这里直接用数组版本:
        x: (N1, N2, N3)
        f: (N1, N3)
    返回:
        m_vrh: (N1, N2)
    """
    x = np.asarray(x)
    f = np.asarray(f)
    N1, N2, N3 = x.shape
    m_vrh = np.zeros((N1, N2), dtype=float)

    for i in range(N2):          # 模量种类: 1=K, 2=mu
        for j in range(N1):      # 深度点
            m_v = 0.0
            m_r = 0.0
            for k in range(N3):  # 组分
                m_v += f[j, k] * x[j, i, k]
                m_r += f[j, k] / x[j, i, k]
            m_r = 1.0 / m_r
            m_vrh[j, i] = 0.5 * (m_v + m_r)
    return m_vrh


# ===========================================================
# 2. Wood 混合流体
# ===========================================================

def Wood(m):
    """
    MATLAB: [kr,den] = Wood(M)
        m = M.wood (h_n,2,3)
            m(:,1,1) = k_w
            m(:,1,2) = k_g
            m(:,2,1) = rho_w
            m(:,2,2) = rho_g
            m(:,1,3) = f_w
            m(:,2,3) = f_g

    Python:
        m: np.ndarray (h_n, 2, 3)
    返回:
        kr: (h_n,) 混合流体体积模量
        den: (h_n,) 混合流体密度
    """
    m = np.asarray(m)
    h_n = m.shape[0]

    k_w = m[:, 0, 0]
    k_g = m[:, 0, 1]
    rho_w = m[:, 1, 0]
    rho_g = m[:, 1, 1]
    f_w = m[:, 0, 2]
    f_g = m[:, 1, 2]

    kr = np.zeros(h_n, dtype=float)
    den = np.zeros(h_n, dtype=float)

    for i in range(h_n):
        kr_m = f_w[i] / k_w[i] + f_g[i] / k_g[i]
        kr[i] = 1.0 / kr_m
        den[i] = f_w[i] * rho_w[i] + f_g[i] * rho_g[i]
    return kr, den


# ===========================================================
# 3. 各向异性 SCA_DEM_ani_0901 接口 (内部还缺 SCA_ani_0113 & DEM_ani)
# ===========================================================

def SCA_DEM_ani_0901(M):
    """
    这是 MATLAB SCA_DEM_ani_0901 的 Python 框架。
    注意: 内部调用的 SCA_ani_0113(M) 和 DEM_ani(M) 你还没给，我先保留接口。

    期望的 M 结构:
        M.m1: (6,6,r_n) 背景刚度
        M.m2: (6,6,r_n) 包含物刚度
        M.phi2: (r_n,)   目标孔隙度 (总孔隙度或软孔比例)
    返回:
        Cmx3: (6,6,r_n) 等效刚度矩阵
    """
    # 这里先抄 MATLAB 的逻辑框架，但 SCA_ani_0113 和 DEM_ani 需要你自己实现
    r_n = M["m1"].shape[2]
    Cmx3 = np.zeros((6, 6, r_n), dtype=float)

    # 设置默认参数（与 MATLAB 同）
    M_local = dict(M)  # 浅拷贝一个局部字典
    M_local["NMIN"] = 2
    M_local["idem"] = 2
    M_local["errmax"] = 5e2
    M_local["al"] = 90
    M_local["xl"] = 0
    M_local["af"] = 0
    M_local["xf"] = 0

    M_local["ioptel"] = np.zeros((2, 1), dtype=int)
    M_local["ioptel"][0, 0] = 1
    M_local["ioptel"][1, 0] = 1

    M_local["axis"] = np.zeros((2, 3), dtype=float)
    M_local["axis"][0, :] = [1, 1, 1]
    M_local["axis"][1, :] = [5, 5, 1]

    M_local["ioptoe"] = np.zeros((2, 1), dtype=int)
    M_local["ioptoe"][0, 0] = 1
    M_local["ioptoe"][1, 0] = 1

    M_local["err0"] = 0.7
    M_local["dv"] = 0.01  # or 0.0005, 取决于你的主函数设定

    for i in range(r_n):
        # 各向异性 SCA: 先计算 phi=0.5 时的等效模量
        M_local["phi"] = np.array([0.5])
        M_local["N"] = 1

        M_local["myEC"] = np.zeros((6, 6, 2))
        M_local["myEC"][:, :, 0] = M["m1"][:, :, i]
        M_local["myEC"][:, :, 1] = M["m2"][:, :, i]

        M_local["vliq"] = np.zeros((2, 1))
        M_local["vliq"][0, 0] = 1.0 - 0.5
        M_local["vliq"][1, 0] = 0.5

        # TODO: 这里需要你提供 SCA_ani_0113(M_local) 的 Python 实现
        raise NotImplementedError("SCA_ani_0113 / DEM_ani 还没有移植到 Python。")

    return Cmx3


# ===========================================================
# 4. Backus_mult241016: 多点 Backus 平均
# ===========================================================

def Backus_mult241016(ix1, phit, vrh_f, B_m):
    """
    MATLAB: [CM] = Backus_mult241016(M,B_m)
        ix1 = M.backus.m  (N1,2)  [K, mu]
        M.phit            (N1,)
        M.vrh.f           (N1,N3)

    Python:
        ix1:  (N1,2)   each row: [K, mu]
        phit: (N1,)
        vrh_f: (N1,N3)
        B_m:  int, odd window length

    返回:
        CM: (N1, 6, 6)  VTI 背景刚度矩阵
    """
    ix1 = np.asarray(ix1)
    phit = np.asarray(phit)
    vrh_f = np.asarray(vrh_f)

    N1 = ix1.shape[0]
    lamda_m = ix1[:, 0] - (2.0 / 3.0) * ix1[:, 1]
    mu_m = ix1[:, 1]
    m = np.column_stack([lamda_m, mu_m, phit])
    nm = B_m // 2

    # 数据头尾镜像对称延长
    m_x = np.zeros((N1 + 2 * nm, 3), dtype=float)
    m_x[nm:nm + N1, :] = m
    m_x[:nm, :] = m[:nm, :][::-1, :]
    m_x[nm + N1:, :] = m[-nm:, :][::-1, :]

    CM = np.zeros((6, 6, N1), dtype=float)

    for i in range(N1):
        # 原代码中对 M.vrh.f(:,1)==1 的特殊处理
        if i < N1 - 1:
            if vrh_f[i + 1, 0] == 1:
                CM[:, :, i] = ku2c(ix1[i, 0], ix1[i, 1])
                continue
        if vrh_f[i, 0] == 1:
            CM[:, :, i] = ku2c(ix1[i, 0], ix1[i, 1])
            continue
        if i > 0:
            if vrh_f[i - 1, 0] == 1:
                CM[:, :, i] = ku2c(ix1[i, 0], ix1[i, 1])
                continue

        # 计算窗口内非零孔隙度的层
        p1 = m_x[i:i + B_m, 2]  # phit
        zero_mask = (p1 == 0)
        B_m1 = B_m - int(np.sum(zero_mask))  # 有效层数

        lamda_x = m_x[i:i + B_m, 0][~zero_mask]
        mu_x = m_x[i:i + B_m, 1][~zero_mask]

        A1 = A2 = A3 = 0.0
        B1 = B2 = B3 = 0.0
        C0 = 0.0
        F1 = F2 = 0.0
        D0 = 0.0
        Mx0 = 0.0

        for j in range(B_m1):
            lamda = lamda_x[j]
            mu = mu_x[j]
            ci = 1.0 / B_m1
            A1 += ci * (4 * mu * (lamda + mu) / (lamda + 2 * mu))
            A2 += ci * (1.0 / (lamda + 2 * mu))
            A3 += ci * (lamda / (lamda + 2 * mu))
            B1 += ci * (2 * mu * lamda / (lamda + 2 * mu))
            B2 += ci * (1.0 / (lamda + 2 * mu))
            B3 += ci * (lamda / (lamda + 2 * mu))
            C0 += ci * (1.0 / (lamda + 2 * mu))
            F1 += ci * (1.0 / (lamda + 2 * mu))
            F2 += ci * (lamda / (lamda + 2 * mu))
            D0 += ci * (1.0 / mu)
            Mx0 += ci * mu

        A = A1 + (1.0 / A2) * A3 ** 2
        B = B1 + (1.0 / B2) * B3 ** 2
        C = 1.0 / C0
        F = (1.0 / F1) * F2
        D = 1.0 / D0
        Mx = Mx0

        Cmat = np.array([
            [A, B, F, 0, 0, 0],
            [B, A, F, 0, 0, 0],
            [F, F, C, 0, 0, 0],
            [0, 0, 0, D, 0, 0],
            [0, 0, 0, 0, D, 0],
            [0, 0, 0, 0, 0, Mx],
        ], dtype=float)

        CM[:, :, i] = Cmat

    return CM


# ===========================================================
# 5. Schoenberg: 裂缝线性滑移模型
# ===========================================================

def Schoenberg(Cb, hudm, mode="HTI"):
    """
    MATLAB: [Csbg] = Schoenberg(M, type)
        M.Cb   -> 背景介质刚度 (6,6,N)
        M.hudm -> HTI: (N,2) [phi_f, X]
                 OA : (N,4) [phi_f_h, X_h, phi_f_v, X_v]

    Python:
        Cb:   (6,6,N)
        hudm: see above
        mode: 'HTI' or 'OA'
    返回:
        Csbg: (6,6,N)
    """
    Cb = np.asarray(Cb)
    N = Cb.shape[2]
    Csbg = np.zeros_like(Cb)

    if mode == "OA":
        for i in range(N):
            c0 = Cb[:, :, i]
            mu = c0[3, 3]
            lamda = c0[2, 2] - 2 * mu
            U1 = 16 * (lamda + 2 * mu) / (3 * (3 * lamda + 4 * mu))
            U3 = 4 * (lamda + 2 * mu) / (3 * (lamda + mu))

            # 水平裂缝
            phi_f = hudm[i, 0]
            X = hudm[i, 1]
            e = 3 * phi_f / (4 * np.pi * X)
            dN = (lamda + 2 * mu) * U3 * e / mu
            dT = U1 * e

            c11 = -(lamda + 2 * mu) * dN
            c12 = -lamda * dN
            c22 = -lamda ** 2 * dN / (lamda + 2 * mu)
            c44 = 0.0
            c55 = -mu * dT

            cf = np.array([
                [c11, c12, c12, 0, 0, 0],
                [c12, c22, c22, 0, 0, 0],
                [c12, c22, c22, 0, 0, 0],
                [0, 0, 0, c44, 0, 0],
                [0, 0, 0, 0, c55, 0],
                [0, 0, 0, 0, 0, c55],
            ], dtype=float)

            MM = np.array([
                [0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, -1],
                [0, 0, 0, 0, -1, 0],
                [0, 0, 0, 1, 0, 0],
            ], dtype=float)
            cm1 = MM @ cf @ MM.T

            # 垂直裂缝
            phi_f_v = hudm[i, 2]
            X_v = hudm[i, 3]
            e = 3 * phi_f_v / (4 * np.pi * X_v)
            dN = (lamda + 2 * mu) * U3 * e / mu
            dT = U1 * e

            c11 = -(lamda + 2 * mu) * dN
            c12 = -lamda * dN
            c22 = -lamda ** 2 * dN / (lamda + 2 * mu)
            c44 = 0.0
            c55 = -mu * dT

            cf2 = np.array([
                [c11, c12, c12, 0, 0, 0],
                [c12, c22, c22, 0, 0, 0],
                [c12, c22, c22, 0, 0, 0],
                [0, 0, 0, c44, 0, 0],
                [0, 0, 0, 0, c55, 0],
                [0, 0, 0, 0, 0, c55],
            ], dtype=float)

            cm2 = cf2
            Csbg[:, :, i] = c0 + cm1 + cm2

    elif mode == "HTI":
        for i in range(N):
            c0 = Cb[:, :, i]
            mu = c0[3, 3]
            lamda = c0[2, 2] - 2 * mu
            U1 = 16 * (lamda + 2 * mu) / (3 * (3 * lamda + 4 * mu))
            U3 = 4 * (lamda + 2 * mu) / (3 * (lamda + mu))

            # 垂直裂缝
            phi_f = hudm[i, 0]
            X = hudm[i, 1]
            e = 3 * phi_f / (4 * np.pi * X)
            dN = (lamda + 2 * mu) * U3 * e / mu
            dT = U1 * e

            c11 = -(lamda + 2 * mu) * dN
            c12 = -lamda * dN
            c22 = -lamda ** 2 * dN / (lamda + 2 * mu)
            c44 = 0.0
            c55 = -mu * dT

            cf2 = np.array([
                [c11, c12, c12, 0, 0, 0],
                [c12, c22, c22, 0, 0, 0],
                [c12, c22, c22, 0, 0, 0],
                [0, 0, 0, c44, 0, 0],
                [0, 0, 0, 0, c55, 0],
                [0, 0, 0, 0, 0, c55],
            ], dtype=float)

            Csbg[:, :, i] = c0 + cf2

    else:
        raise ValueError("mode must be 'HTI' or 'OA'.")

    return Csbg


# ===========================================================
# 6. BKG: Brown & Korringa 各向异性流体替换
# ===========================================================

def BKG(C0_all, m_bkg):
    """
    MATLAB: [Cbkg] = BKG(M)
        M.bkgm.C0 -> (6,6,N) 干岩石刚度
        M.bkgm.m  -> (N,3) [km, kfl, phi]

    Python:
        C0_all: (6,6,N)
        m_bkg:  (N,3) [km, kfl, phi]
    返回:
        Cbkg: (6,6,N) 饱和刚度
    """
    C0_all = np.asarray(C0_all)
    m_bkg = np.asarray(m_bkg)
    N = C0_all.shape[2]
    Cbkg = np.zeros_like(C0_all)

    for ik in range(N):
        km = m_bkg[ik, 0]
        kfl = m_bkg[ik, 1]
        phi = m_bkg[ik, 2]

        C0_6 = C0_all[:, :, ik]
        C0 = c22c4(C0_6)  # -> (3,3,3,3)

        C_sat = np.zeros_like(C0)
        # i,j,k,l 对应 0..2
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        c_a = 0.0
                        c_b = 0.0
                        c_cd = 0.0
                        for ii in range(3):
                            c_a += C0[i, j, ii, ii]
                            c_b += C0[ii, ii, k, l]
                            for jj in range(3):
                                c_cd += C0[ii, ii, jj, jj]

                        num = (km * mydelta(i, j) - c_a / 3.0) * \
                              (km * mydelta(k, l) - c_b / 3.0)
                        den = (km / kfl) * phi * (km - kfl) + (km - c_cd / 9.0)

                        C_sat[i, j, k, l] = C0[i, j, k, l] + num / den

        C_sat_6 = c42c2(C_sat)
        Cbkg[:, :, ik] = C_sat_6

    return Cbkg


# ===========================================================
# 7. C2T: 正交各向异性刚度 -> 弹性/各向异性参数
# ===========================================================

def C2T(C_all, den):
    """
    MATLAB: [m0,m1,m2] = C2T(M)
        M.c2t.c   -> (6,6,N)
        M.c2t.den -> (N,)

    Python:
        C_all: (6,6,N)
        den:   (N,)
    返回:
        m0: (N,8) [M0,mu0,den,epsilon0,delta0,ZN,ZV,ZH]
        m1: (N,9) [Vp0,Vs0,epsilon1,epsilon2,gamma1,gamma2,delta1,delta2,delta3]
        m2: (N,8) [Vp0,Vs0,den,epsilon0,delta0,ZN,ZV,ZH]
    """
    C_all = np.asarray(C_all)
    den = np.asarray(den)
    N = C_all.shape[2]

    m0 = np.zeros((N, 8), dtype=float)
    m1 = np.zeros((N, 9), dtype=float)
    m2 = np.zeros((N, 8), dtype=float)

    for i in range(N):
        c = C_all[:, :, i]
        rho = den[i]

        Vp0 = np.sqrt(c[2, 2] / rho) * 1e3
        Vs0 = np.sqrt(c[4, 4] / rho) * 1e3   
        # 注释里提到曾从 c(5,5) 改为 c(4,4)
        # 如果想用 $C_{44}$，Python 代码应改为 c[3, 3]

        epsilon1 = (c[1, 1] - c[2, 2]) / (2 * c[2, 2])
        epsilon2 = (c[0, 0] - c[2, 2]) / (2 * c[2, 2])
        gamma1 = (c[5, 5] - c[4, 4]) / (2 * c[4, 4])
        gamma2 = (c[5, 5] - c[3, 3]) / (2 * c[3, 3])

        delta1 = ((c[1, 2] + c[3, 3]) ** 2 - (c[2, 2] - c[3, 3]) ** 2) / \
                 (2 * c[2, 2] * (c[2, 2] - c[3, 3]))
        delta2 = ((c[0, 2] + c[4, 4]) ** 2 - (c[2, 2] - c[4, 4]) ** 2) / \
                 (2 * c[2, 2] * (c[2, 2] - c[4, 4]))
        delta3 = ((c[0, 1] + c[5, 5]) ** 2 - (c[0, 0] - c[5, 5]) ** 2) / \
                 (2 * c[0, 0] * (c[0, 0] - c[5, 5]))

        M0 = rho * Vp0 ** 2
        mu0 = rho * Vs0 ** 2
        epsilon0 = epsilon1
        delta0 = delta1
        g = Vs0 ** 2 / Vp0 ** 2
        ZN = (epsilon1 - epsilon2) / (2 * g * (1 - g))
        ZH = ZN - delta3 / (2 * g)
        ZV = (delta1 - delta2) / (2 * g) - (1 - 2 * g) * ZN

        m0[i, :] = [M0, mu0, rho, epsilon0, delta0, ZN, ZV, ZH]
        m1[i, :] = [Vp0, Vs0, epsilon1, epsilon2, gamma1, gamma2, delta1, delta2, delta3]
        m2[i, :] = [Vp0, Vs0, rho, epsilon0, delta0, ZN, ZV, ZH]

    # 250417 改：过滤异常值（裂缝弱度过大或为负）
    mean_zn = np.mean(m0[:, 5])
    mean_zv = np.mean(m0[:, 6])
    mean_zh = np.mean(m0[:, 7])

    for i in range(N):
        if i <= 2:
            continue
        if (m0[i, 5] > 20 * mean_zn or
            m0[i, 6] > 20 * mean_zv or
            m0[i, 7] > 20 * mean_zh):
            m0[i, :] = np.mean(m0[i - 3:i, :], axis=0)
            m1[i, :] = np.mean(m1[i - 3:i, :], axis=0)
            m2[i, :] = np.mean(m2[i - 3:i, :], axis=0)
        if (m0[i, 5] < 0 or m0[i, 6] < 0 or m0[i, 7] < 0):
            m0[i, :] = np.mean(m0[i - 3:i, :], axis=0)
            m1[i, :] = np.mean(m1[i - 3:i, :], axis=0)
            m2[i, :] = np.mean(m2[i - 3:i, :], axis=0)

    return m0, m1, m2



def compute_vsh_from_gr(df_std, gr_clean=30.0, gr_shale=120.0):
    """
    用 GR 计算 Vsh（泥质体积），简单线性模型：
    Vsh = (GR - GR_clean) / (GR_shale - GR_clean)，并裁剪到 [0,1]
    """
    gr = df_std["GR"].values
    vsh = (gr - gr_clean) / (gr_shale - gr_clean)
    vsh = np.clip(vsh, 0.0, 1.0)
    return vsh