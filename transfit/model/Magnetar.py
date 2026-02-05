# sn_uniform_fast_final_opt.py
# -*- coding: utf-8 -*-

import numpy as np
import numba  # 导入Numba
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u

# unified constants (CGS, Numba-friendly)
from transfit.constants import (
    PI, C_LIGHT, DAY,
    M_SUN, R_SUN,
    SIGMA_SB, H_PLANCK, K_BOLTZ,
)

# -----------------------------------------------------------------------------
# 核心计算函数，使用 Numba JIT 编译
# -----------------------------------------------------------------------------

@numba.njit(fastmath=True, cache=True)
def thomas_algorithm(a, b, c_up, d):
    """
    Numba-jitted Thomas algorithm for solving a tridiagonal system Ax=d.
    a: lower diagonal (a[0] is ignored)
    b: main diagonal
    c_up: upper diagonal (c_up[-1] is ignored)
    d: right-hand side vector
    Returns the solution vector x.
    """
    n = len(d)
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    x = np.zeros(n)

    # Forward elimination
    c_prime[0] = c_up[0] / b[0]
    d_prime[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * c_prime[i - 1]
        c_prime[i] = c_up[i] / denom
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom

    # Backward substitution
    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


@numba.njit(fastmath=True, cache=True)
def _fast_time_loop_numba(
    # Grid parameters
    Ny, Nx, dx, dy,
    # Pre-calculated physics arrays
    fR_vals, f_ob_vals, heat_vals, xi_vals,
    # Pre-calculated matrix components
    mu_const, up_const, lo_const, diag_const,
    # Initial condition & constants
    e_initial, Lfac
):
    """
    The entire time-evolution loop, JIT-compiled with Numba.
    This function contains the performance-critical part of the calculation.
    """
    # 内存管理：仅需两个向量
    e_now = e_initial.copy()
    e_next = np.empty_like(e_now)

    # 输出数组
    L_bol_out = np.zeros(Ny)

    # 预分配三对角矩阵的对角线和RHS向量
    # Thomas algorithm inputs: a=lower, b=main, c_up=upper, d=rhs
    a = np.zeros(Nx + 1)
    b_diag = np.zeros(Nx + 1)
    c_up = np.zeros(Nx + 1)
    rhs = np.zeros(Nx + 1)

    # 空间切片（在Numba中定义常量比传入更高效）
    i_mid = slice(1, Nx)
    im1 = slice(0, Nx - 1)
    ip1 = slice(2, Nx + 1)
    xi_inner = xi_vals[1:-1]

    # --- 主时间循环 (在编译代码中运行) ---
    for n in range(Ny):
        fR_now, fR_next = fR_vals[n], fR_vals[n + 1]

        # --- 组装 A 矩阵 (Ax=d) ---
        # 内点 (i=1..Nx-1)
        mu_fr_next = mu_const * fR_next
        b_diag[i_mid] = 1.0 + mu_fr_next * diag_const
        c_up[i_mid] = -mu_fr_next * up_const    # 上对角线
        a[i_mid] = -mu_fr_next * lo_const       # 下对角线

        # 左边界: -e_0 + e_1 = 0
        b_diag[0] = -1.0
        c_up[0] = 1.0
        a[0] = 0.0  # 在Thomas算法中被忽略

        # 右边界: (dx - f_ob)*e_N + f_ob*e_{N-1} = 0
        b_diag[Nx] = dx - f_ob_vals[n + 1]
        a[Nx] = f_ob_vals[n + 1]
        c_up[Nx] = 0.0  # 在Thomas算法中被忽略

        # --- 组装 RHS 向量 d ---
        S_now_inner = xi_inner * (fR_now * heat_vals[n])
        S_next_inner = xi_inner * (fR_next * heat_vals[n + 1])

        rhs[i_mid] = (
            e_now[i_mid]
            + 0.5 * dy * (S_now_inner + S_next_inner)
            + (mu_const * fR_now) * (
                up_const * (e_now[ip1] - e_now[i_mid])
                - lo_const * (e_now[i_mid] - e_now[im1])
            )
        )
        rhs[0] = 0.0
        rhs[Nx] = 0.0

        # --- 调用Numba优化的求解器 ---
        e_next = thomas_algorithm(a, b_diag, c_up, rhs)

        # --- 计算光度并存储 ---
        L_bol_out[n] = Lfac * (e_next[Nx - 1] - e_next[Nx])

        # --- 准备下一步 (无内存拷贝的引用交换) ---
        e_now, e_next = e_next, e_now

    return L_bol_out


class MagnetarModel:
    """
    JIT-Accelerated Fast PDE light-curve model for supernovae.
    - All optimizations from previous version are retained.
    - The entire time-evolution loop is JIT-compiled with Numba.
    - Replaces scipy.linalg.solve_banded with a custom Numba-jitted Thomas algorithm solver.
    - This version is ideal for heavy-duty MCMC sampling.
    """

    def __init__(self):
        # JIT 预热：初始化时跑一次小计算触发编译，避免第一次正式调用卡顿
        print("Initializing and JIT-compiling the model...")
        dummy_theta = (10.0, 1.0, 1.0, 0.1, 100.0, 0.5, 0.2, 0.03, 4000)
        self.calculate_light_curve(dummy_theta, Nx=10, Ny=20)
        print("Model is ready for fast execution.")

    def calculate_light_curve(self, theta, Nx=100, Ny=1000, t_max_days=150.0):
        pi, c, day = PI, C_LIGHT, DAY

        # --- unpack theta: magnetar version ---
        (M_ej, v_ej, E_Th_in, P_ms, B14, R_max_in, kappa0, kappa_gamma, T_floor) = theta

        M_ej = float(M_ej) * M_SUN
        E_Th_in = float(E_Th_in) * 1.0e49
        R_max_in = float(R_max_in) * R_SUN
        x_s = 0.05
        kappa0 = float(kappa0)
        kappa_g = float(kappa_gamma)
        v_ej = float(v_ej) * 1e9

        # --- x grid (same) ---
        x_min, x_max = 1.0, 1.0e4
        x_heat = np.clip(x_s * x_max, x_min, x_max)

        # --- dynamics / density profile (same) ---
        E_K = 0.5 * M_ej * v_ej * v_ej
        I_M = (x_max**3 - x_min**3) / 3.0
        I_K = (x_max**5 - x_min**5) / 5.0

        R_min_in = R_max_in / x_max
        rho_in = M_ej / (4.0 * pi * I_M * R_min_in**3)
        v_min = np.sqrt(2.0 * I_M * E_K / (I_K * M_ej))
        t_ex = R_min_in / v_min
        t_diff = 3.0 * kappa0 * rho_in * R_min_in**2 / c
        tau_in = kappa0 * rho_in * R_min_in

        # gamma leakage timescale (keep your form)
        t_gamma = np.sqrt((3.0 * kappa_g * M_ej) / (4.0 * pi * v_ej * v_ej))

        # ------------------------------------------------------------------
        # Magnetar injection: E_p, t_p, and dimensionless heat(t)
        # ------------------------------------------------------------------
        # canonical NS parameters
        I_ns = 1.0e45      # g cm^2
        R_ns = 1.0e6       # cm

        P = float(P_ms) * 1.0e-3          # s
        B = float(B14) * 1.0e14           # G
        Omega = 2.0 * pi / P

        # rotational energy
        E_p = 0.5 * I_ns * Omega * Omega  # erg

        # dipole spin-down timescale
        t_p = (6.0 * I_ns * c**3) / (B * B * R_ns**6 * Omega * Omega)  # s

        # characteristic specific power scale (erg/g/s)
        eps0 = (E_p / t_p) / M_ej

        # NEW u0 uses magnetar power scale
        u0 = rho_in * eps0 * t_diff

        # L0 and e0_coeff follow the same nondimensionalization logic
        L0 = (4.0 * pi * R_min_in * c * u0) / (3.0 * kappa0 * rho_in)
        e0_coeff = E_Th_in / (2.0 * pi * u0 * x_max**2 * R_min_in**3)

        # ------------------------------------------------------------------
        # xi profile (uniform inside [x_min, x_heat])
        # ------------------------------------------------------------------
        if x_heat <= x_min + 1e-14:
            xi0 = 0.0
        else:
            denom_heat = (x_heat**3 - x_min**3) / 3.0
            xi0 = I_M / denom_heat
        xi0 = max(xi0, 0.0)

        # --- grids (same) ---
        Nx, Ny = int(Nx), int(Ny)
        x_vals = np.linspace(x_min, x_max, Nx + 1)
        dx = (x_max - x_min) / Nx
        x2 = x_vals * x_vals

        t_max = float(t_max_days) * day
        y_max = t_max / t_diff
        t_min = 1e-6 * day
        y_min = t_min / t_diff
        y_vals = np.linspace(y_min, y_max, Ny + 1)
        dy = y_vals[1] - y_vals[0]

        fR_vals = 1.0 + (y_vals * t_diff / t_ex)
        f_ob_vals = -(4.0 / (3.0 * tau_in)) * (fR_vals * fR_vals)

        # ------------------------------------------------------------------
        # build magnetar heat(t): dimensionless relative to (E_p/t_p)
        # heat(t) = f_dep(t) / (1 + t/t_p)^2
        # ------------------------------------------------------------------
        t_phys = y_vals * t_diff

        dep = np.zeros_like(t_phys)
        mask = t_phys > 0.0
        dep[mask] = 1.0 - np.exp(- (t_gamma / t_phys[mask])**2)

        heat = dep / (1.0 + t_phys / t_p)**2

        # --- xi(x) ---
        xi_vals = np.zeros_like(x_vals)
        if x_heat > x_min:
            mask_heat = (x_vals >= x_min) & (x_vals <= x_heat)
            xi_vals[mask_heat] = xi0

        # --- matrix precompute ---
        i_mid = slice(1, Nx)
        im1 = slice(0, Nx - 1)
        ip1 = slice(2, Nx + 1)
        x_inner = x_vals[1:-1]
        inv_xdx2 = 1.0 / (x_inner * dx)**2
        mu_const = 0.25 * dy * inv_xdx2
        up_const = (x2[ip1] + x2[i_mid])
        lo_const = (x2[i_mid] + x2[im1])
        diag_const = (x2[ip1] + 2.0 * x2[i_mid] + x2[im1])

        e_initial = e0_coeff / x_vals
        Lfac = L0 * (x_max * x_max) / dx

        L_out = _fast_time_loop_numba(
            Ny, Nx, dx, dy,
            fR_vals, f_ob_vals, heat, xi_vals,
            mu_const, up_const, lo_const, diag_const,
            e_initial, Lfac
        )

        t_s = (y_vals * t_diff)[1:]

        # --- photosphere + T floor ---
        R_nom = R_max_in * fR_vals[1:]
        Teff_try = (L_out / (4.0 * pi * R_nom * R_nom * SIGMA_SB))**0.25
        R_floor = np.sqrt(L_out / (4.0 * pi * SIGMA_SB * (T_floor**4)))

        T_eff_values = np.where(Teff_try > T_floor, Teff_try, T_floor)
        R_outer_values = np.where(Teff_try > T_floor, R_nom, R_floor)

        return t_s, L_out, T_eff_values, R_outer_values
    
    def L_bol(self, t_obs, theta, z=0.0, **kwargs):
        t_s, L_series, T_eff_values, R_outer_values = self.calculate_light_curve(theta, **kwargs)
        t_obs_grid = t_s * (1.0 + z)
        return np.interp(t_obs, t_obs_grid, L_series, left=L_series[0], right=L_series[-1])

    def M_ab(self, t_obs, theta, nu_obs, z):
        nu_obs = nu_obs * (1.0 + z)
        lum_dist = cosmo.luminosity_distance(z)
        DL_z = lum_dist.to(u.cm).value

        t_s, L_bol_values, T_for_calc, R_outer = self.calculate_light_curve(theta)
        t_obs_grid = t_s * (1.0 + z)

        x_obs = H_PLANCK * nu_obs / (K_BOLTZ * T_for_calc)
        B_nu = 2.0 * H_PLANCK * nu_obs**3 / (C_LIGHT**2) / (np.exp(x_obs) - 1.0)
        L_nu = 4.0 * PI * PI * R_outer**2 * B_nu
        F_nu = L_nu / (4.0 * PI * (DL_z**2))

        M_ab_values = -2.5 * np.log10(F_nu) - 48.6 - 2.5 * np.log10(1.0 + z)
        M_ab_obs = np.interp(t_obs, t_obs_grid, M_ab_values, left=M_ab_values[0], right=M_ab_values[-1])
        return M_ab_obs

