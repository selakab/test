# rock_physics_utils.py
import numpy as np
from numba import jit
# ============================================================
# 基础工具：Voigt <-> Kelvin
# ============================================================

def voigt_to_kelvin(Voigt: np.ndarray) -> np.ndarray:
    """6x6 Voigt -> Kelvin 矩阵."""
    M = np.array([
        [1.0, 0, 0, 0, 0, 0],
        [0, 1.0, 0, 0, 0, 0],
        [0, 0, 1.0, 0, 0, 0],
        [0, 0, 0, np.sqrt(2.0), 0, 0],
        [0, 0, 0, 0, np.sqrt(2.0), 0],
        [0, 0, 0, 0, 0, np.sqrt(2.0)],
    ])
    return M @ Voigt @ M


def kelvin_to_voigt(Kelvin: np.ndarray) -> np.ndarray:
    """6x6 Kelvin -> Voigt."""
    M = np.array([
        [1.0, 0, 0, 0, 0, 0],
        [0, 1.0, 0, 0, 0, 0],
        [0, 0, 1.0, 0, 0, 0],
        [0, 0, 0, 1.0 / np.sqrt(2.0), 0, 0],
        [0, 0, 0, 0, 1.0 / np.sqrt(2.0), 0],
        [0, 0, 0, 0, 0, 1.0 / np.sqrt(2.0)],
    ])
    return M @ Kelvin @ M


# ============================================================
# 一堆几何/角度工具：orientation / azr / azi / xyz ...
# ============================================================

def dircos(x, y, z):
    small = 1e-4
    xm = np.sqrt(x * x + y * y + z * z)
    if abs(x) > small:
        x = x / xm
    else:
        x = 0.0
    if abs(y) > small:
        y = y / xm
    else:
        y = 0.0
    if abs(z) > small:
        z = z / xm
    else:
        z = 0.0
    return x, y, z


def xyz(az, xinc):
    """az, xinc -> x,y,z 方向余弦."""
    rad = np.pi / 180.0
    raz = az * rad
    rinc = xinc * rad
    x = np.cos(raz) * np.cos(rinc)
    y = -np.sin(raz) * np.cos(rinc)
    z = np.sin(rinc)
    return x, y, z


def cross_vec(x1, y1, z1, x2, y2, z2):
    """向量叉乘."""
    x3 = y1 * z2 - z1 * y2
    y3 = z1 * x2 - x1 * z2
    z3 = x1 * y2 - y1 * x2
    return x3, y3, z3


def arctan_deg(x, y):
    """和 MATLAB 版本一样，把 atan2 控制到 0~360°."""
    small = 1e-5
    deg = 180.0 / np.pi
    ax = abs(x)
    ay = abs(y)

    if ay < small and ax < small:
        return 0.0
    if ay <= small and x > 0:
        return 0.0
    if ax <= small and y > 0:
        return 90.0
    if ay <= small and x < 0:
        return 180.0
    if ax <= small and y < 0:
        return 270.0

    ratio = y / x
    if ratio > 350:
        ratio = 350.0
    elif ratio < -350:
        ratio = -350.0

    angle = deg * np.arctan(ratio)

    if x > 0 and y > 0:
        return angle
    if x < 0 and y > 0:
        return 180.0 + angle
    if x < 0 and y < 0:
        return 180.0 + angle
    if x > 0 and y < 0:
        return 360.0 + angle
    return angle


def arccos_deg(x):
    if x > 1.0:
        x = 1.0
    elif x < -1.0:
        x = -1.0
    angle = np.arccos(x) * 180.0 / np.pi
    return x, angle


def azi(x, y, z, ihemi):
    """笛卡尔 -> 方位角/倾角（地理系）."""
    if z < 0:
        xt, yt, zt = -x, -y, -z
    else:
        xt, yt, zt = x, y, z

    xt, yt, zt = dircos(xt, yt, zt)
    az = arctan_deg(xt, yt)
    az = 360.0 - az  # 转为左手系

    r = np.sqrt(xt * xt + yt * yt + zt * zt)
    if zt == 0:
        polar = 0.0
    else:
        polar = zt / r
    _, xinc = arccos_deg(polar)
    xinc = 90.0 - xinc

    if ihemi == -1 and z < 0:
        az = az + 180.0
        xinc = -xinc
    if az > 360.0:
        az -= 360.0
    return az, xinc


def azr(al, xl, af, xf):
    """根据 A1, A3 的方位/倾角构造 3x3 旋转矩阵."""
    deg = 180.0 / np.pi
    r = np.zeros((3, 3))

    x1, y1, z1 = xyz(al, xl)
    x3, y3, z3 = xyz(af, xf)

    x1, y1, z1 = dircos(x1, y1, z1)
    x3, y3, z3 = dircos(x3, y3, z3)

    dot = x1 * x3 + y1 * y3 + z1 * z3
    dot = max(min(dot, 1.0), -1.0)
    _ = np.arccos(dot) * deg  # value1，仅检查不用

    x2, y2, z2 = cross_vec(x3, y3, z3, x1, y1, z1)
    x2, y2, z2 = dircos(x2, y2, z2)
    x3, y3, z3 = cross_vec(x1, y1, z1, x2, y2, z2)
    x1, y1, z1 = cross_vec(x2, y2, z2, x3, y3, z3)
    x2, y2, z2 = cross_vec(x3, y3, z3, x1, y1, z1)

    dot = x1 * x3 + y1 * y3 + z1 * z3
    dot = max(min(dot, 1.0), -1.0)
    _ = np.arccos(dot) * deg  # value2

    r[0, :] = [x1, y1, z1]
    r[1, :] = [x2, y2, z2]
    r[2, :] = [x3, y3, z3]
    # det 一般应为 +1
    return r


def orientation(al, xl, af, xf, ihemi):
    """MATLAB orientation 的 Python 版."""
    re = azr(al, xl, af, xf)
    az1, xinc1 = azi(re[0, 0], re[0, 1], re[0, 2], ihemi)
    az2, xinc2 = azi(re[1, 0], re[1, 1], re[1, 2], ihemi)
    az3, xinc3 = azi(re[2, 0], re[2, 1], re[2, 2], ihemi)
    return re, az1, xinc1, az2, xinc2, az3, xinc3


# ============================================================
# Green 张量相关：gauleg / tcheb / error4G / green3 / rot
# ============================================================

def gauleg(x1, x2, n):
    """Gauss-Legendre 积分点与权重."""
    m = n // 2
    x = np.zeros(n)
    w = np.zeros(n)
    xm = 0.5 * (x1 + x2)
    xl = 0.5 * (x2 - x1)
    xn = float(n)

    for i in range(1, m + 1):
        xi = float(i)
        z = np.cos(np.pi * (xi - 0.25) / (xn + 0.5))
        p1 = 1.0
        p2 = 0.0
        for j in range(1, n + 1):
            xj = float(j)
            p3 = p2
            p2 = p1
            p1 = ((2 * j - 1) * z * p2 - (xj - 1) * p3) / xj
        pp = n * (z * p1 - p2) / (z * z - 1.0)
        z1 = z
        z = z1 - p1 / pp
        while abs(z - z1) > np.finfo(float).eps:
            p1 = 1.0
            p2 = 0.0
            for j in range(1, n + 1):
                xj = float(j)
                p3 = p2
                p2 = p1
                p1 = ((2 * j - 1) * z * p2 - (xj - 1) * p3) / xj
            pp = n * (z * p1 - p2) / (z * z - 1.0)
            z1 = z
            z = z1 - p1 / pp
        x[i - 1] = xm - xl * z
        x[n - i] = xm + xl * z
        w[i - 1] = 2.0 * xl / ((1.0 - z * z) * pp * pp)
        w[n - i] = w[i - 1]
    return x, w


def tcheb(nb):
    x = np.zeros(nb // 2)
    w = np.zeros(nb // 2)
    ntcheb = nb // 2
    x1 = 0.0
    x2 = 1.0
    xp, wp = gauleg(x1, x2, nb)
    for i in range(nb // 2):
        x[i] = 0.5 - xp[i]
        w[i] = wp[i]
    return ntcheb, x, w

'''
def green3(C2, axis, x, w, ntcheb):
    """
    Green 张量，基本保持 MATLAB 形式。
    C2: 6x6 刚度矩阵
    axis: (3,) 椭球半轴
    x, w: 积分点/权重 (长度 ntcheb)
    """
    ijkl = np.array([[1, 6, 5],
                     [6, 2, 4],
                     [5, 4, 3]], dtype=int) - 1
    
    # 对应 Voigt 索引: 1->11, 2->22, 3->33, 4->23, 5->31, 6->12
    # 注意这里使用 0-based 索引 (1->0, 2->1, 3->2)
    ijv = np.array([
        [0, 0],  # Voigt 1 (C11)
        [1, 1],  # Voigt 2 (C22)
        [2, 2],  # Voigt 3 (C33)
        [1, 2],  # Voigt 4 (C23)
        [2, 0],  # Voigt 5 (C13/C31)
        [0, 1]   # Voigt 6 (C12)
    ], dtype=int)

    G = np.zeros((6, 6))
    g2 = np.zeros((6, 6))
    ck = np.zeros((3, 3))
    cp = np.zeros((3, 3))
    xk = np.zeros(3)

    nint = 2 * ntcheb
    xp = np.zeros(nint)
    wp = np.zeros(nint)
    for i in range(ntcheb):
        xp[i] = 0.5 + x[i]
        wp[i] = w[i]
        xp[i + ntcheb] = 0.5 - x[i]
        wp[i + ntcheb] = w[i]

    pas = np.pi / 2.0
    stheta = np.zeros((nint, 2))
    ctheta = np.zeros((nint, 2))
    sphi = np.zeros((nint, 2))
    cphi = np.zeros((nint, 2))

    for ine in range(2):
        xx1 = pas * ine
        for ig in range(nint):
            xx = xx1 + pas * xp[ig]
            stheta[ig, ine] = np.sin(xx)
            ctheta[ig, ine] = np.cos(xx)
            sphi[ig, ine] = np.sin(xx)
            cphi[ig, ine] = np.cos(xx)

    for it in range(2):
        for ip in range(2):
            for jt in range(nint):
                for jp in range(nint):
                    sth = stheta[jt, it]
                    cth = ctheta[jt, it]
                    sph = sphi[jp, ip]
                    cph = cphi[jp, ip]

                    xk[0] = sth * cph
                    xk[1] = sth * sph
                    xk[2] = cth

                    # Christoffel matrix
                    for i in range(3):
                        for k in range(i + 1):
                            ckik = 0.0
                            for j in range(3):
                                for l in range(3):
                                    mm = ijkl[j, i]
                                    nn = ijkl[l, k]
                                    ckik += C2[nn, mm] * xk[j] * xk[l]
                            ck[k, i] = ckik
                            ck[i, k] = ckik
                    cp[:, :] = ck
                    ck = np.linalg.inv(cp)

                    for i in range(6):
                        i1 = ijv[i, 0]
                        i2 = ijv[i, 1]
                        for j in range(6):
                            j1 = ijv[j, 0]
                            j2 = ijv[j, 1]
                            val = (ck[i1, j1] * xk[i2] * xk[j2] +
                                   ck[i2, j1] * xk[i1] * xk[j2] +
                                   ck[i1, j2] * xk[i2] * xk[j1] +
                                   ck[i2, j2] * xk[i1] * xk[j1])
                            g2[j, i] = 0.5 * val

                    ro = np.sqrt((axis[0] * xk[0]) ** 2 +
                                 (axis[1] * xk[1]) ** 2 +
                                 (axis[2] * xk[2]) ** 2)
                    alfa = 2.0 * ro
                    sgalfa = np.sign(alfa) if alfa != 0 else 1.0
                    t = (-alfa ** 3 + 6 * ro * alfa * alfa - 6 * alfa * ro * ro) * sgalfa
                    f1 = 2.0 * t / (ro ** 6)

                    for i in range(6):
                        for j in range(6):
                            G[j, i] += g2[j, i] * sth * f1 * wp[jt] * wp[jp]

    fact = axis[0] * axis[1] * axis[2] / (32.0 * np.pi)
    G *= fact * pas * pas
    return G
'''
@jit(nopython=True, cache=True)
def green3(C2, axis, x, w, ntcheb):
    # 1. 预定义的常量数组 (Moved constant arrays here for clarity, Numba handles them well)
    ijkl = np.array([[0, 5, 4],
                     [5, 1, 3],
                     [4, 3, 2]], dtype=np.int32) # 也就是原来的 [[1,6,5],[6,2,4],[5,4,3]]-1
    
    # ijv 对应: 11, 22, 33, 23, 31, 12 -> Indices: 0, 1, 2, [1,2], [2,0], [0,1]
    ijv = np.array([
        [0, 0],  # Voigt 0 (C11)
        [1, 1],  # Voigt 1 (C22)
        [2, 2],  # Voigt 2 (C33)
        [1, 2],  # Voigt 3 (C23)
        [2, 0],  # Voigt 4 (C13)
        [0, 1]   # Voigt 5 (C12)
    ], dtype=np.int32)

    # 2. 初始化变量
    G = np.zeros((6, 6), dtype=np.float64)
    g2 = np.zeros((6, 6), dtype=np.float64)
    ck = np.zeros((3, 3), dtype=np.float64)
    cp = np.zeros((3, 3), dtype=np.float64)
    xk = np.zeros(3, dtype=np.float64)

    # 3. 积分点处理
    nint = 2 * ntcheb
    xp = np.zeros(nint, dtype=np.float64)
    wp = np.zeros(nint, dtype=np.float64)
    for i in range(ntcheb):
        xp[i] = 0.5 + x[i]
        wp[i] = w[i]
        xp[i + ntcheb] = 0.5 - x[i]
        wp[i + ntcheb] = w[i]

    pas = np.pi / 2.0
    stheta = np.zeros((nint, 2), dtype=np.float64)
    ctheta = np.zeros((nint, 2), dtype=np.float64)
    sphi = np.zeros((nint, 2), dtype=np.float64)
    cphi = np.zeros((nint, 2), dtype=np.float64)

    # 预计算三角函数
    for ine in range(2):
        xx1 = pas * ine
        for ig in range(nint):
            xx = xx1 + pas * xp[ig]
            stheta[ig, ine] = np.sin(xx)
            ctheta[ig, ine] = np.cos(xx)
            sphi[ig, ine]   = np.sin(xx)
            cphi[ig, ine]   = np.cos(xx)

    # 4. 主循环 (The Big Loop)
    for it in range(2):
        for ip in range(2):
            for jt in range(nint):
                sth = stheta[jt, it]
                cth = ctheta[jt, it]
                # 提前计算循环不变项
                xk2 = cth # xk[2]
                
                for jp in range(nint):
                    sph = sphi[jp, ip]
                    cph = cphi[jp, ip]
                    
                    # 向量 k (xk)
                    xk[0] = sth * cph
                    xk[1] = sth * sph
                    xk[2] = xk2

                    # Christoffel matrix (ck = C_ijkl * x_j * x_l)
                    # 这一步是计算热点，Numba 会自动展开优化
                    ck[:] = 0.0 # 重置
                    for i in range(3):
                        for k in range(i + 1): # 利用对称性只算下三角
                            val = 0.0
                            for j in range(3):
                                for l in range(3):
                                    # ijkl[j, i] 获取 C 的索引
                                    mm = ijkl[j, i]
                                    nn = ijkl[l, k]
                                    val += C2[nn, mm] * xk[j] * xk[l]
                            ck[k, i] = val
                            ck[i, k] = val
                    
                    # 矩阵求逆 (3x3)
                    # Numba 支持 np.linalg.inv，对于 3x3 非常快
                    cp[:] = ck # 复制一份去求逆
                    ck_inv = np.linalg.inv(cp) # 结果存回 ck_inv

                    # 计算 g2
                    for i in range(6):
                        i1 = ijv[i, 0]
                        i2 = ijv[i, 1]
                        for j in range(6):
                            j1 = ijv[j, 0]
                            j2 = ijv[j, 1]
                            
                            term1 = ck_inv[i1, j1] * xk[i2] * xk[j2]
                            term2 = ck_inv[i2, j1] * xk[i1] * xk[j2]
                            term3 = ck_inv[i1, j2] * xk[i2] * xk[j1]
                            term4 = ck_inv[i2, j2] * xk[i1] * xk[j1]
                            
                            g2[j, i] = 0.5 * (term1 + term2 + term3 + term4)

                    # 逻辑优化：数学化简 f1
                    # 原逻辑：
                    # ro = sqrt(...)
                    # alfa = 2*ro
                    # t = (-alfa^3 + 6*ro*alfa^2 - 6*alfa*ro^2) * sgalfa
                    # f1 = 2*t / ro^6
                    # --------------------------------
                    # 化简后：f1 = 8.0 / ro^3
                    
                    # 注意：axis 必须是 float
                    r_sq = (axis[0]*xk[0])**2 + (axis[1]*xk[1])**2 + (axis[2]*xk[2])**2
                    ro = np.sqrt(r_sq)
                    
                    if ro > 1e-12: # 防止除以零
                        f1 = 8.0 / (ro**3)
                    else:
                        f1 = 0.0

                    # 累加到 Green Tensor
                    weight = sth * f1 * wp[jt] * wp[jp]
                    for i in range(6):
                        for j in range(6):
                            G[j, i] += g2[j, i] * weight

    fact = axis[0] * axis[1] * axis[2] / (32.0 * np.pi)
    G *= fact * pas * pas
    return G

def rot(l1, l2, ijkl, r, ec):
    """旋转刚度张量."""
    c = np.zeros((6, 6))
    d = ec.copy()
    # m,n 对应 Voigt 索引
    for m in range(6):
        i = l1[m]
        j = l2[m]
        for n in range(m + 1):
            k = l1[n]
            l = l2[n]
            x_val = 0.0
            for lp in range(3):
                y_val = 0.0
                for lq in range(3):
                    lt = ijkl[lp, lq]
                    y_val += r[j, lq] * (
                        r[k, 0] * (r[l, 0] * d[lt, 0] +
                                   r[l, 1] * d[lt, 5] +
                                   r[l, 2] * d[lt, 4]) +
                        r[k, 1] * (r[l, 0] * d[lt, 5] +
                                   r[l, 1] * d[lt, 1] +
                                   r[l, 2] * d[lt, 3]) +
                        r[k, 2] * (r[l, 0] * d[lt, 4] +
                                   r[l, 1] * d[lt, 3] +
                                   r[l, 2] * d[lt, 2])
                    )
                x_val += r[i, lp] * y_val
            c[m, n] = x_val
            c[n, m] = x_val
    return c


def error4G(cm, errmax, eaxis):
    """Green 张量误差分析，决定积分分段数 nb."""
    gtold = np.zeros((6, 6))
    for inb in range(1, 101):
        nb = inb * 4
        ntcheb, x, w = tcheb(nb)
        gt = green3(cm, eaxis, x, w, ntcheb)
        diag = 0.0
        for i in range(6):
            # 只累加对角线的误差
            diff = abs(gt[i, i] - gtold[i, i]) / abs(gt[i, i]) if gt[i, i] != 0 else 0.0
            diag += diff
            gtold[i, i] = gt[i, i]
        diagav = 100.0 * diag / 6.0
        if diagav < errmax:
            ntcheb, x, w = tcheb(nb)
            return nb, x, w, ntcheb
    # 若 100 次仍不收敛，返回最后一次
    ntcheb, x, w = tcheb(nb)
    return nb, x, w, ntcheb


# ============================================================
# SCA 里的 mysca / DEM 里的 difdem2
# ============================================================

def mysca(csca, vb, cinc, gt):
    """SCA 中用的等效刚度更新."""
    csca = csca.T
    cinc = cinc.T
    gt = gt.T
    iden = np.eye(6)

    cscam = voigt_to_kelvin(csca)
    cincm = voigt_to_kelvin(cinc)
    gm = voigt_to_kelvin(gt)

    am = iden + gm @ (cincm - cscam)
    c_um1 = cincm @ np.linalg.inv(am)
    c_dm1 = np.linalg.inv(am)

    c_um2 = kelvin_to_voigt(c_um1)
    c_dm2 = kelvin_to_voigt(c_dm1)

    c_u = vb * c_um2
    c_d = vb * c_dm2
    return c_u.T, c_d.T


def difdem2(cdem, vb, cinc, gt, dvb):
    """DEM 微分更新."""
    cdem = cdem.T
    cinc = cinc.T
    gt = gt.T

    iden = np.eye(6)

    cdemm = voigt_to_kelvin(cdem)
    cincm = voigt_to_kelvin(cinc)
    gm = voigt_to_kelvin(gt)

    am = iden + gm @ (cincm - cdemm)
    cdevm = (cincm - cdemm) @ np.linalg.inv(am)
    cdev = kelvin_to_voigt(cdevm)

    f = 1.0 / (1.0 - vb)
    cdem = cdem + dvb * f * cdev
    return cdem.T


# ============================================================
# 1. Python 版 SCA_ani_0113
# ============================================================

def sca_ani_0113(M):
    """
    Python 版 SCA_ani_0113(M).

    需要的字段（和 MATLAB 一致）：
        M['errmax'], M['ioptel'], M['ioptoe'], M['axis'],
        M['al'], M['xl'], M['af'], M['xf'],
        M['idem'], M['vliq'], M['NMIN'], M['err0'],
        M['myEC'], M['N']
    返回：
        Cmx  : (4, N)
        Cmx1 : (6, 6, N)
    """
    ijkl = np.array([[1, 6, 5],
                     [6, 2, 4],
                     [5, 4, 3]], dtype=int) - 1
    l1 = np.array([1, 2, 3, 2, 3, 1], dtype=int) - 1
    l2 = np.array([1, 2, 3, 3, 1, 2], dtype=int) - 1

    errmax = M['errmax']
    ioptel = np.array(M['ioptel'], dtype=int).reshape(-1)
    ioptoe = np.array(M['ioptoe'], dtype=int).reshape(-1)
    axis = np.array(M['axis'], dtype=float).copy()
    al = float(M['al'])
    xl = float(M['xl'])
    af = float(M['af'])
    xf = float(M['xf'])
    idem = int(M['idem']) - 1  # MATLAB 1/2 -> Python 0/1
    vliq = np.array(M['vliq'], dtype=float)  # 形状 NMIN x N
    M_ihemi = 1
    NMIN = int(M['NMIN'])
    err0 = float(M['err0'])

    ECM = np.zeros((NMIN, 6, 6), dtype=float)
    REMIN = np.zeros((NMIN, 3, 3), dtype=float)

    for MIN in range(NMIN):
        ECM[MIN] = np.array(M['myEC'][:, :, MIN], dtype=float)
        if ioptel[MIN] == 1:
            axis[MIN, :] = 1.0
            ioptoe[MIN] = 1
            REMIN[MIN] = np.eye(3)
        elif ioptel[MIN] == 2:
            if ioptoe[MIN] == 1:
                REMIN[MIN] = np.eye(3)
            elif ioptoe[MIN] == 2:
                re, *_ = orientation(al, xl, af, xf, M_ihemi)
                REMIN[MIN] = re

    # inclusion 相（流体）
    MIN = idem
    eaxis = axis[MIN].copy()
    EC = ECM[MIN].copy()
    RBACK = REMIN[MIN].copy()
    RBACKI = RBACK.T

    N = int(M['N'])
    Cmx = np.zeros((4, N), dtype=float)
    Cmx1 = np.zeros((6, 6, N), dtype=float)

    for i in range(N):
        # 计算 Voigt 上限作为初值
        csca0 = np.zeros((6, 6), dtype=float)
        for j in range(NMIN):
            csca0 += vliq[j, i] * ECM[j]

        err = 1e15
        while err > err0:
            c_u = np.zeros((6, 6), dtype=float)
            c_d = np.zeros((6, 6), dtype=float)
            for j in range(NMIN):
                _, x, w, ntcheb = error4G(csca0, errmax, eaxis)
                Cdemr1 = rot(l1, l2, ijkl, RBACK, csca0)
                GtR1 = green3(Cdemr1, eaxis, x, w, ntcheb)
                Gt1 = rot(l1, l2, ijkl, RBACKI, GtR1)
                c_um, c_dm = mysca(csca0, vliq[j, i], ECM[j], Gt1)
                c_u += c_um
                c_d += c_dm

            c_u2 = voigt_to_kelvin(c_u.T)
            c_d2 = voigt_to_kelvin(c_d.T)
            cscam1 = c_u2 @ np.linalg.inv(c_d2)
            cscam2 = kelvin_to_voigt(cscam1)
            csca = cscam2.T
            err = np.sum(np.abs(csca - csca0))
            csca0 = csca

        Cmx1[:, :, i] = csca
        Cmx[0, i] = csca[0, 0]
        Cmx[1, i] = csca[2, 2]
        Cmx[2, i] = csca[3, 3]
        Cmx[3, i] = csca[5, 5]

    return Cmx, Cmx1


# ============================================================
# 2. Python 版 DEM_ani（只保留干岩/高频刚度部分）
# ============================================================

def dem_ani(M):
    """
    Python 版 DEM_ani(M).

    需要字段：
        M['errmax'], M['ioptel'], M['ioptoe'], M['axis'],
        M['al'], M['xl'], M['af'], M['xf'],
        M['idem'], M['NMIN'], M['myEC'],
        M['v1'], M['v2'], M['dv']
    返回：
        vliq : 一维数组，积分过程中所有 Vb
        Cmx  : 6x6 刚度矩阵（最后一次，即给定孔隙度的刚度）
    """
    ijkl = np.array([[1, 6, 5],
                     [6, 2, 4],
                     [5, 4, 3]], dtype=int) - 1
    l1 = np.array([1, 2, 3, 2, 3, 1], dtype=int) - 1
    l2 = np.array([1, 2, 3, 3, 1, 2], dtype=int) - 1

    errmax = M['errmax']
    ioptel = np.array(M['ioptel'], dtype=int).reshape(-1)
    ioptoe = np.array(M['ioptoe'], dtype=int).reshape(-1)
    axis = np.array(M['axis'], dtype=float).copy()
    al = float(M['al'])
    xl = float(M['xl'])
    af = float(M['af'])
    xf = float(M['xf'])
    idem = int(M['idem']) - 1
    NMIN = int(M['NMIN'])
    ihemi = 1

    ECM = np.zeros((NMIN, 6, 6), dtype=float)
    REMIN = np.zeros((NMIN, 3, 3), dtype=float)

    for MIN in range(NMIN):
        ECM[MIN] = np.array(M['myEC'][:, :, MIN], dtype=float)
        if ioptel[MIN] == 1:
            axis[MIN, :] = 1.0
            ioptoe[MIN] = 1
            REMIN[MIN] = np.eye(3)
        elif ioptel[MIN] == 2:
            if ioptoe[MIN] == 1:
                REMIN[MIN] = np.eye(3)
            elif ioptoe[MIN] == 2:
                re, *_ = orientation(al, xl, af, xf, ihemi)
                REMIN[MIN] = re

    if idem == 0:
        idem2 = 1
    else:
        idem2 = 0

    Cbg = ECM[idem2].copy()

    Cdem1 = Cbg.copy()
    Cdem2 = Cbg.copy()
    Cpore = np.zeros((6, 6), dtype=float)

    v1 = float(M['v1'])
    v2 = float(M['v2'])
    dv = float(M['dv'])
    vliq = np.arange(v1, v2 + 0.5 * dv, dv)
    Vb = v1

    MIN = idem
    eaxis = axis[MIN].copy()
    EC = ECM[MIN].copy()
    RBACK = REMIN[MIN].copy()
    RBACKI = RBACK.T

    for k in range(len(vliq)):
        if k > 0:
            Vb = Vb + dv
        Va = 1.0 - Vb

        _, x, w, ntcheb = error4G(Cdem2, errmax, eaxis)
        Cdemr1 = rot(l1, l2, ijkl, RBACK, Cdem1)
        Cdemr2 = rot(l1, l2, ijkl, RBACK, Cdem2)
        GtR1 = green3(Cdemr1, eaxis, x, w, ntcheb)
        GtR2 = green3(Cdemr2, eaxis, x, w, ntcheb)
        Gt1 = rot(l1, l2, ijkl, RBACKI, GtR1)
        Gt2 = rot(l1, l2, ijkl, RBACKI, GtR2)

        Cdem1 = difdem2(Cdem1, Vb, EC, Gt1, dv)
        Cdem2 = difdem2(Cdem2, Vb, Cpore, Gt2, dv)

    Cmx = Cdem1.copy()  # 最后一次高频刚度
    return vliq, Cmx
