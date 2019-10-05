from __future__ import division

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse, io
from scipy.linalg import norm
from scipy.sparse.linalg import gmres
from sklearn.metrics import mean_absolute_error


class prop_rock(object):
    """
    This is a class that captures rock physical properties, including permeability, porosity, and
    compressibility.
    """
    def __init__(self, kx=0, ky=0, por=0, cr=0, kro=0, dkro=0, krg=0, dkrg=0):
        self.kx = kx
        self.ky = ky
        self.por = por
        self.cr = cr
        self.kro = kro
        self.krg = krg
        self.dkro = dkro
        self.dkrg = dkrg

    def calc_kro(self, sg):
        self.kro = (1 - sg) ** 1.5
        return self.kro

    def calc_dkro(self, sg):
        self.dkro = -1.5 * (1 - sg) ** 0.5
        return self.dkro

    def calc_krg(self, sg):
        self.krg = (sg) ** 2
        return self.krg

    def calc_dkrg(self, sg):
        self.dkrg = 2 * sg
        return self.dkrg

    def plot_kro(self):
        sgx = np.linspace(0, 1, 500)
        kro_try = []
        dkro_try = []
        for i in sgx:
            kro_try.append(prop_rock.calc_kro(self, i))
            dkro_try.append(prop_rock.calc_dkro(self, i))
        f, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(sgx, kro_try, 'r-', label=r'$kr_o$')
        ax2.plot(sgx, dkro_try, 'r--', label=r'$\frac{\partial kro_o}{\partial sg}$')
        ax1.set_xlabel('Gas Saturation (fraction)')
        ax1.set_ylabel('kro')
        ax2.set_ylabel('kro derivative')
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        plt.grid()
        plt.show()
        return f

    def plot_krg(self):
        sgx = np.linspace(0, 1, 500)
        krg_try = []
        dkrg_try = []
        for i in sgx:
            krg_try.append(prop_rock.calc_krg(self, i))
            dkrg_try.append(prop_rock.calc_dkrg(self, i))
        f, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(sgx, krg_try, 'r-', label=r'$kr_g$')
        ax2.plot(sgx, dkrg_try, 'r--', label=r'$\frac{\partial kr_g}{\partial sg}$')
        ax1.set_xlabel('Gas Saturation (fraction)')
        ax1.set_ylabel('krg')
        ax2.set_ylabel('krg derivative')
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        plt.grid()
        plt.show()
        return f

    def plot_all(self):
        f1 = prop_rock.plot_kro(self)
        f2 = prop_rock.plot_krg(self)
        return f1, f2


class prop_fluid(object):
    """
    This object contains fluid properties. Phase: oil and gas. Isothermal; only a function of pressure.
    """
    def __init__(self, c_o=0, mu_o=0, rho_o=0, mu_g=0, dmu_g=0, rmu_g=0, p_bub=0, p_atm=14.7, b_o=0, b_g=0,
                 dp=0, rs=0, db_o=0, db_g=0, drs=0):
        self.c_o = c_o
        self.mu_o = mu_o
        self.rho_o = rho_o
        self.mu_g = mu_g
        self.rmu_g = rmu_g
        self.dmu_g = dmu_g
        self.p_bub = p_bub
        self.p_atm = p_atm
        self.b_o = b_o
        self.db_o = db_o
        self.b_g = b_g
        self.db_g = db_g
        self.dp = dp
        self.rs = rs
        self.drs = drs

    def calc_mu_g(self, p):
        self.mu_g = 3e-10 * p ** 2 + 1e-6 * p + 0.0133
        return self.mu_g

    def calc_dmu_g(self, p):
        self.dmu_g = 3e-10 * 2 * p + 1e-6
        return self.dmu_g

    def calc_rmu_g(self, p):
        self.rmu_g = 20000000000 * (3 * p + 5000) / (3 * p ** 2 + 10000 * p + 133000000) ** 2
        return self.rmu_g

    def calc_dp(self, p):
        if p < self.p_bub:
            self.dp = self.p_atm - p
        else:
            self.dp = self.p_atm - self.p_bub
        return self.dp

    def calc_bo(self, p):
        if p < self.p_bub:
            self.b_o = 1 / np.exp(-8e-5 * (self.p_atm - p))
        else:
            self.b_o = 1 / (np.exp(-8e-5 * (self.p_atm - self.p_bub)) * np.exp(-self.c_o * (p - self.p_bub)))
        return self.b_o

    def calc_dbo(self, p):
        if p < self.p_bub:
            self.db_o = -8e-5 * np.exp(8e-5 * (self.p_atm - p))
        else:
            self.db_o = self.c_o * np.exp(8e-5 * (self.p_atm - self.p_bub)) * np.exp(self.c_o * (p - self.p_bub))
        return self.db_o

    def calc_bg(self, p):
        self.b_g = 1 / (np.exp(1.7e-3 * prop_fluid.calc_dp(self, p)))
        return self.b_g

    def calc_dbg(self, p):
        if p < self.p_bub:
            self.db_g = 1.7e-3 * np.exp(-1.7e-3 * prop_fluid.calc_dp(self, p))
        else:
            self.db_g = 0
        return self.db_g

    def calc_rs(self, p):
        if p < self.p_bub:
            rs_factor = 1
        else:
            rs_factor = 0
        self.rs = 178.11 ** 2 / 5.615 * ((p / self.p_bub) ** 1.3 * rs_factor + (1 - rs_factor))
        return self.rs

    def calc_drs(self, p):
        if p < self.p_bub:
            rs_factor = 1
        else:
            rs_factor = 0
        self.drs = 178.11 ** 2 / 5.615 * (1.3 * p ** 0.3 / self.p_bub ** 1.3 * rs_factor + 0 * (1 - rs_factor))
        return self.drs

    def plot_bo(self):
        px = np.linspace(1, 5000, 1000)
        bo_try = []
        dbo_try = []
        for i in px:
            bo_try.append(prop_fluid.calc_bo(self, i))
            dbo_try.append(prop_fluid.calc_dbo(self, i))
        f, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(px, bo_try, 'r-', label=r'$b_o$')
        ax2.plot(px, dbo_try, 'r--', label=r'$\frac{\partial b_o}{\partial p}$')
        ax1.set_xlabel('Pressure (psi)')
        ax1.set_ylabel('Oil Shrinkage (RB/STB)')
        ax2.set_ylabel('Oil Shrinkage Derivative')
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        plt.grid()
        plt.show()
        return f

    def plot_bg(self):
        px = np.linspace(1, 5000, 1000)
        bg_try = []
        dbg_try = []
        for i in px:
            bg_try.append(prop_fluid.calc_bg(self, i))
            dbg_try.append(prop_fluid.calc_dbg(self, i))
        f, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(px, bg_try, 'r-', label=r'$b_g$')
        ax2.plot(px, dbg_try, 'r--', label=r'$\frac{\partial b_g}{\partial p}$')
        ax1.set_xlabel('Pressure (psi)')
        ax1.set_ylabel('Gas Shrinkage (RB/STB)')
        ax2.set_ylabel('Gas Shrinkage Derivative')
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        plt.grid()
        plt.show()
        return f

    def plot_rs(self):
        px = np.linspace(1, 5000, 1000)
        rs_try = []
        drs_try = []
        for i in px:
            rs_try.append(prop_fluid.calc_rs(self, i))
            drs_try.append(prop_fluid.calc_drs(self, i))
        f, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(px, rs_try, 'r-', label=r'$R_s$')
        ax2.plot(px, drs_try, 'r--', label=r'$\frac{\partial R_s}{\partial p}$')
        ax1.set_xlabel('Pressure (psi)')
        ax1.set_ylabel('Solution Gas-Oil Ratio (RB/STB)')
        ax2.set_ylabel('Solution Gas-Oil Ratio Derivative')
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        plt.grid()
        plt.show()
        return f

    def plot_mu_g(self):
        px = np.linspace(1, 5000, 1000)
        mu_g_try = []
        dmu_g_try = []
        for i in px:
            mu_g_try.append(prop_fluid.calc_mu_g(self, i))
            dmu_g_try.append(prop_fluid.calc_dmu_g(self, i))
        f, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(px, mu_g_try, 'r-', label=r'$\mu_g$')
        ax2.plot(px, dmu_g_try, 'r--', label=r'$\frac{\partial \mu_g}{\partial p}$')
        ax1.set_xlabel('Pressure (psi)')
        ax1.set_ylabel('Gas Viscosity (cp)')
        ax2.set_ylabel('Gas Viscosity Derivative')
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        plt.grid()
        plt.show()
        return f

    def plot_all(self):
        f1 = prop_fluid.plot_bo(self)
        f2 = prop_fluid.plot_bg(self)
        f3 = prop_fluid.plot_rs(self)
        f4 = prop_fluid.plot_mu_g(self)
        return f1, f2, f3, f4


class prop_grid(object):
    """This describes grid dimension and numbers."""
    def __init__(self, Nx=0, Ny=0, Nz=0):
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz

    def grid_dimension_x(self, Lx):
        return self.Nx / Lx

    def grid_dimension_y(self, Ly):
        return self.Ny / Ly

    def grid_dimension_z(self, Lz):
        return self.Nz / Lz


class prop_res(object):
    """A class that captures reservoir dimension and initial pressure."""
    def __init__(self, Lx=0, Ly=0, Lz=0, press_n=0, sg_n=0, press_n1_k=0, sg_n1_k=0):
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.press_n = press_n
        self.sg_n = sg_n
        self.press_n1_k = press_n1_k
        self.sg_n1_k = sg_n1_k

    def stack_ps(self, press_n, sg_n):
        stacked_vector = []
        for i in range(len(press_n)):
            stacked_vector.append(press_n[i])
            stacked_vector.append(sg_n[i])
        return np.asarray(stacked_vector)


class prop_well(object):
    """Describes well location a flow rate. Also provides conversion from
    cartesian i,j coordinate to grid number"""
    def __init__(self, loc=0, q=0):
        self.loc = loc
        self.q = q

    def index_to_grid(self, Nx):
        return self.loc[1] * Nx + self.loc[0]


class prop_time(object):
    """Describes time-step (assumed constant) and time interval"""
    def __init__(self, tstep=0, timeint=0):
        self.tstep = tstep
        self.timeint = timeint


def load_data(filename):
    """Loads ECLIPSE simulation block pressure data as a comparison"""
    url = 'https://raw.githubusercontent.com/titaristanto/reservoir-simulation/master/eclipse%20bhp.csv'
    df = pd.read_csv(url)

    t = df.loc[:, ['TIME']]  # Time in simulation: DAY
    p = df.loc[:, ['BPR:(18,18,1)']]
    return t, p


def calc_transmissibility_x(k_x, kr_o, mu_o, b_o, params, i, j):
    """Calculates transmissibility in x-direction"""
    # Calculate transmissibility in x-direction. Unit: (md ft psi)/cp
    dx = params['dx']
    dy = params['dy']
    dz = params['dz']
    p_grids = params['p_grids_n1']

    # Arithmetic Average for k
    k_x_avg = (dx[j, i] + dx[j, i + 1]) / (dx[j, i] / k_x[j, i] + dx[j, i + 1] / k_x[j, i + 1])
    A = dy[j, i] * dz[j, i]
    x_l = (dx[j, i] + dx[j, i + 1]) / 2

    fluid_term = upwind(p_grids, [kr_o, b_o, 1 / mu_o], i, j, dir='x')
    return k_x_avg * A / x_l * fluid_term


def calc_transmissibility_y(k_y, kr_o, mu_o, b_o, params, i, j):
    """Calculates transmissibility in y-direction"""
    # Calculate transmissibility in y-direction. Unit: (md ft psi)/cp
    dx = params['dx']
    dy = params['dy']
    dz = params['dz']
    p_grids = params['p_grids_n1']

    # Arithmetic Average for k
    k_x_avg = (dy[j, i] + dy[j + 1, i]) / (dy[j, i] / k_y[j, i] + dy[j + 1, i] / k_y[j + 1, i])
    A = dx[j, i] * dz[j, i]
    x_l = (dy[j, i] + dy[j + 1, i]) / 2

    fluid_term = upwind(p_grids, [kr_o, b_o, 1 / mu_o], i, j, dir='y')
    return k_x_avg * A / x_l * fluid_term


def upwind(p_grids, pars, i, j, dir):
    # upwind parameters based on pressure between two blocks
    if dir == 'x':
        if p_grids[j, i] > p_grids[j, i + 1]:
            mult = 1
            for p in range(len(pars)):
                mult *= pars[p][j, i]
        else:
            mult = 1
            for p in range(len(pars)):
                mult *= pars[p][j, i + 1]
    elif dir == 'y':
        if p_grids[j, i] > p_grids[j + 1, i]:
            mult = 1
            for p in range(len(pars)):
                mult *= pars[p][j, i]
        else:
            mult = 1
            for p in range(len(pars)):
                mult *= pars[p][j + 1, i]
    return mult


def ij_to_grid(i, j, Nx):
    # Convert i,j coordinate to block number
    return (i) + Nx * j


def flip_variables(M, ind):
    # Flip variables (for Residual and Jacobian). Ind==0 for vector, 1 for 2D matrix.
    J = M * 1
    m = J.shape[0]
    for i in range(m):
        if i % 2 == 0:
            if ind == 0:
                J[i:i + 2] = np.flip(J[i:i + 2], 0)
            elif ind == 1:
                J[i:i + 2, :] = np.flip(J[i:i + 2, :], 0)
            else:
                print('Unknown Choice..')
    return J


def construct_T(mat, params):
    # Create matrix T containing connection transmissibilities of all blocks
    k_x = params['k_x']
    k_y = params['k_y']
    b_o = params['b_o']
    b_g = params['b_g']
    mu_o = params['mu_o']
    mu_g = params['mu_g']
    kr_o = params['kr_o']
    kr_g = params['kr_g']
    rs = params['rs']
    p_grids = params['p_grids_n1']

    m = mat.shape[0]
    n = mat.shape[1]
    T = np.zeros((m * n * 2, m * n * 2))
    for j in range(m):
        for i in range(n):
            # 2 neighbors in x direction
            if i < n - 1:
                # Oil D1
                T[(mat[j, i] - 1) * 2, (mat[j, i + 1] - 1) * 2] = calc_transmissibility_x(k_x, kr_o, mu_o, b_o, params,
                                                                                          i, j)
                T[(mat[j, i + 1] - 1) * 2, (mat[j, i] - 1) * 2] = T[(mat[j, i] - 1) * 2, (mat[j, i + 1] - 1) * 2]

                # Gas D3
                T[(mat[j, i] - 1) * 2 + 1, (mat[j, i + 1] - 1) * 2] = calc_transmissibility_x(k_x, kr_g, mu_g, b_g,
                                                                                              params, i, j) + upwind(
                    p_grids, [rs], i, j, dir='x') * calc_transmissibility_x(k_x, kr_o, mu_o, b_o, params, i, j)
                T[(mat[j, i + 1] - 1) * 2 + 1, (mat[j, i] - 1) * 2] = T[
                    (mat[j, i] - 1) * 2 + 1, (mat[j, i + 1] - 1) * 2]
            # 2 neighbors in y direction
            if j < m - 1:
                # Oil D1
                T[(mat[j, i] - 1) * 2, (mat[j + 1, i] - 1) * 2] = calc_transmissibility_y(k_y, kr_o, mu_o, b_o, params,
                                                                                          i, j)
                T[(mat[j + 1, i] - 1) * 2, (mat[j, i] - 1) * 2] = T[(mat[j, i] - 1) * 2, (mat[j + 1, i] - 1) * 2]

                # Gas D3
                T[(mat[j, i] - 1) * 2 + 1, (mat[j + 1, i] - 1) * 2] = calc_transmissibility_y(k_y, kr_g, mu_g, b_g,
                                                                                              params, i, j) + upwind(
                    p_grids, [rs], i, j, dir='y') * calc_transmissibility_y(k_y, kr_o, mu_o, b_o, params, i, j)
                T[(mat[j + 1, i] - 1) * 2 + 1, (mat[j, i] - 1) * 2] = T[
                    (mat[j, i] - 1) * 2 + 1, (mat[j + 1, i] - 1) * 2]

    for k in range(T.shape[0]):
        # For 2 phases only, not generalized to n phases
        if k % 2 == 0:
            T[k, k] = -np.sum(T[k, ::2])
            T[k, k + 1] = -np.sum(T[k, 1::2])
        else:
            T[k, k - 1] = -np.sum(T[k, ::2])
            T[k, k] = -np.sum(T[k, 1::2])
    T = T * 0.001127
    return T


def construct_J(mat, params, props):
    # Construct Jacobian matrix
    k_x = params['k_x']
    k_y = params['k_y']
    b_o = params['b_o']
    db_o = params['db_o']
    b_g = params['b_g']
    db_g = params['db_g']
    mu_o = params['mu_o']
    mu_g = params['mu_g']
    dmu_g = params['dmu_g']
    kr_o = params['kr_o']
    dkr_o = params['dkr_o']
    kr_g = params['kr_g']
    dkr_g = params['dkr_g']
    rs = params['rs']
    drs = params['drs']
    p_grids_n = params['p_grids_n']
    p_grids_n1 = params['p_grids_n1']
    dx = params['dx']
    dy = params['dy']
    dz = params['dz']
    sg_n = params['sg_n']
    sg_n1 = params['sg_n1']
    por = props['rock'].por
    T = params['T']

    m = mat.shape[0]
    n = mat.shape[1]
    J = np.zeros((m * n * 2, m * n * 2))
    for j in range(m):
        for i in range(n):
            C1_i_neg = 0
            C1_i_pos = 0
            C1_j_neg = 0
            C1_j_pos = 0
            C2_i_neg = 0
            C2_i_pos = 0
            C2_j_neg = 0
            C2_j_pos = 0
            C3_i_neg = 0
            C3_i_pos = 0
            C3_j_neg = 0
            C3_j_pos = 0
            C4_i_neg = 0
            C4_i_pos = 0
            C4_j_neg = 0
            C4_j_pos = 0
            ## 2 neighbors in x direction
            # Right block (i+1/2) elements
            if i < n - 1:
                # Oil D1 derivative w.r.t. pressure
                dp_i_pos = (p_grids_n1[j, i + 1] - p_grids_n1[j, i])
                if p_grids_n1[j, i] < p_grids_n1[j, i + 1]:
                    D1_i_pos = dp_i_pos * calc_transmissibility_x(k_x, kr_o, mu_o, db_o, params, i, j)
                    D2_i_pos = dp_i_pos * calc_transmissibility_x(k_x, dkr_o, mu_o, b_o, params, i, j)
                    D3_i_pos_free = dp_i_pos * calc_transmissibility_x(k_x, kr_g, mu_g ** 2,
                                                                       (db_g * mu_g - b_g * dmu_g), params, i, j)
                    D3_i_pos_sol = dp_i_pos * calc_transmissibility_x(k_x, kr_o, mu_o, drs * b_o + db_o * rs, params, i,
                                                                      j)
                    D3_i_pos = D3_i_pos_free + D3_i_pos_sol
                    D4_i_pos = dp_i_pos * (
                                calc_transmissibility_x(k_x, dkr_g, mu_g, b_g, params, i, j) + calc_transmissibility_x(
                            k_x, dkr_o, mu_o, rs * b_o, params, i, j))
                    C1_i_pos = 0
                    C2_i_pos = 0
                    C3_i_pos = 0
                    C4_i_pos = 0
                else:
                    D1_i_pos = 0
                    D2_i_pos = 0
                    D3_i_pos = 0
                    D4_i_pos = 0
                    C1_i_pos = dp_i_pos * calc_transmissibility_x(k_x, kr_o, mu_o, db_o, params, i, j)
                    C2_i_pos = dp_i_pos * calc_transmissibility_x(k_x, dkr_o, mu_o, b_o, params, i, j)
                    C3_i_pos_free = dp_i_pos * calc_transmissibility_x(k_x, kr_g, mu_g ** 2,
                                                                       (db_g * mu_g - b_g * dmu_g), params, i, j)
                    C3_i_pos_sol = dp_i_pos * calc_transmissibility_x(k_x, kr_o, mu_o, drs * b_o + db_o * rs, params, i,
                                                                      j)
                    C3_i_pos = C3_i_pos_free + C3_i_pos_sol
                    C4_i_pos = dp_i_pos * (
                                calc_transmissibility_x(k_x, dkr_g, mu_g, b_g, params, i, j) + calc_transmissibility_x(
                            k_x, dkr_o, mu_o, rs * b_o, params, i, j))
                J[(mat[j, i] - 1) * 2, (mat[j, i + 1] - 1) * 2] = calc_transmissibility_x(k_x, kr_o, mu_o, b_o, params,
                                                                                          i, j) + D1_i_pos

                # Oil D2 derivative w.r.t. sg
                J[(mat[j, i] - 1) * 2, (mat[j, i + 1] - 1) * 2 + 1] = D2_i_pos

                # Gas D3 derivative w.r.t. pressure
                J[(mat[j, i] - 1) * 2 + 1, (mat[j, i + 1] - 1) * 2] = calc_transmissibility_x(k_x, kr_g, mu_g, b_g,
                                                                                              params, i,
                                                                                              j) + calc_transmissibility_x(
                    k_x, kr_o, mu_o, b_o * rs, params, i, j) + D3_i_pos

                # Gas D4 derivative w.r.t. sg
                J[(mat[j, i] - 1) * 2 + 1, (mat[j, i + 1] - 1) * 2 + 1] = D4_i_pos

            # Left block (i-1/2) elements
            if i > 0:
                # Oil D1 derivative w.r.t. pressure
                dp_i_neg = (p_grids_n1[j, i - 1] - p_grids_n1[j, i])
                if p_grids_n1[j, i] < p_grids_n1[j, i - 1]:
                    D1_i_neg = dp_i_neg * calc_transmissibility_x(k_x, kr_o, mu_o, db_o, params, i - 1, j)
                    D2_i_neg = dp_i_neg * calc_transmissibility_x(k_x, dkr_o, mu_o, b_o, params, i - 1, j)
                    D3_i_neg_free = dp_i_neg * calc_transmissibility_x(k_x, kr_g, mu_g ** 2,
                                                                       (db_g * mu_g - b_g * dmu_g), params, i - 1, j)
                    D3_i_neg_sol = dp_i_neg * calc_transmissibility_x(k_x, kr_o, mu_o, drs * b_o + db_o * rs, params,
                                                                      i - 1, j)
                    D3_i_neg = D3_i_neg_free + D3_i_neg_sol
                    D4_i_neg = dp_i_neg * (calc_transmissibility_x(k_x, dkr_g, mu_g, b_g, params, i - 1,
                                                                   j) + calc_transmissibility_x(k_x, dkr_o, mu_o,
                                                                                                rs * b_o, params, i - 1,
                                                                                                j))
                    C1_i_neg = 0
                    C2_i_neg = 0
                    C3_i_neg = 0
                    C4_i_neg = 0
                else:
                    D1_i_neg = 0
                    D2_i_neg = 0
                    D3_i_neg = 0
                    D4_i_neg = 0
                    C1_i_neg = dp_i_neg * calc_transmissibility_x(k_x, kr_o, mu_o, db_o, params, i - 1, j)
                    C2_i_neg = dp_i_neg * calc_transmissibility_x(k_x, dkr_o, mu_o, b_o, params, i - 1, j)
                    C3_i_neg_free = dp_i_neg * calc_transmissibility_x(k_x, kr_g, mu_g ** 2,
                                                                       (db_g * mu_g - b_g * dmu_g), params, i - 1, j)
                    C3_i_neg_sol = dp_i_neg * calc_transmissibility_x(k_x, kr_o, mu_o, drs * b_o + db_o * rs, params,
                                                                      i - 1, j)
                    C3_i_neg = C3_i_neg_free + C3_i_neg_sol
                    C4_i_neg = dp_i_neg * (calc_transmissibility_x(k_x, dkr_g, mu_g, b_g, params, i - 1,
                                                                   j) + calc_transmissibility_x(k_x, dkr_o, mu_o,
                                                                                                rs * b_o, params, i - 1,
                                                                                                j))
                J[(mat[j, i] - 1) * 2, (mat[j, i - 1] - 1) * 2] = calc_transmissibility_x(k_x, kr_o, mu_o, b_o, params,
                                                                                          i - 1, j) + D1_i_neg

                # Oil D2 derivative w.r.t. sg
                J[(mat[j, i] - 1) * 2, (mat[j, i - 1] - 1) * 2 + 1] = D2_i_neg

                # Gas D3 derivative w.r.t. pressure
                J[(mat[j, i] - 1) * 2 + 1, (mat[j, i - 1] - 1) * 2] = calc_transmissibility_x(k_x, kr_g, mu_g, b_g,
                                                                                              params, i - 1,
                                                                                              j) + calc_transmissibility_x(
                    k_x, kr_o, mu_o, b_o * rs, params, i - 1, j) + D3_i_neg

                # Gas D4 derivative w.r.t. sg
                J[(mat[j, i] - 1) * 2 + 1, (mat[j, i - 1] - 1) * 2 + 1] = D4_i_neg

            ## 2 neighbors in y direction
            # Lower block (j+1/2) elements
            if j < m - 1:
                # Oil D1 derivative w.r.t. pressure
                dp_j_pos = (p_grids_n1[j + 1, i] - p_grids_n1[j, i])
                if p_grids_n1[j, i] < p_grids_n1[j + 1, i]:
                    D1_j_pos = dp_j_pos * calc_transmissibility_y(k_y, kr_o, mu_o, db_o, params, i, j)
                    D2_j_pos = dp_j_pos * calc_transmissibility_y(k_y, dkr_o, mu_o, b_o, params, i, j)
                    D3_j_pos_free = dp_j_pos * calc_transmissibility_y(k_y, kr_g, mu_g ** 2,
                                                                       (db_g * mu_g - b_g * dmu_g), params, i, j)
                    D3_j_pos_sol = dp_j_pos * calc_transmissibility_y(k_y, kr_o, mu_o, drs * b_o + db_o * rs, params, i,
                                                                      j)
                    D3_j_pos = D3_j_pos_free + D3_j_pos_sol
                    D4_j_pos = dp_j_pos * (
                                calc_transmissibility_y(k_y, dkr_g, mu_g, b_g, params, i, j) + calc_transmissibility_y(
                            k_y, dkr_o, mu_o, rs * b_o, params, i, j))
                    C1_j_pos = 0
                    C2_j_pos = 0
                    C3_j_pos = 0
                    C4_j_pos = 0
                else:
                    D1_j_pos = 0
                    D2_j_pos = 0
                    D3_j_pos = 0
                    D4_j_pos = 0
                    C1_j_pos = dp_j_pos * calc_transmissibility_y(k_y, kr_o, mu_o, db_o, params, i, j)
                    C2_j_pos = dp_j_pos * calc_transmissibility_y(k_y, dkr_o, mu_o, b_o, params, i, j)
                    C3_j_pos_free = dp_j_pos * calc_transmissibility_y(k_y, kr_g, mu_g ** 2,
                                                                       (db_g * mu_g - b_g * dmu_g), params, i, j)
                    C3_j_pos_sol = dp_j_pos * calc_transmissibility_y(k_y, kr_o, mu_o, drs * b_o + db_o * rs, params, i,
                                                                      j)
                    C3_j_pos = C3_j_pos_free + C3_j_pos_sol
                    C4_j_pos = dp_j_pos * (
                                calc_transmissibility_y(k_y, dkr_g, mu_g, b_g, params, i, j) + calc_transmissibility_y(
                            k_y, dkr_o, mu_o, rs * b_o, params, i, j))
                J[(mat[j, i] - 1) * 2, (mat[j + 1, i] - 1) * 2] = calc_transmissibility_y(k_y, kr_o, mu_o, b_o, params,
                                                                                          i, j) + D1_j_pos

                # Oil D2 derivative w.r.t. sg
                J[(mat[j, i] - 1) * 2, (mat[j + 1, i] - 1) * 2 + 1] = D2_j_pos

                # Gas D3 derivative w.r.t. pressure
                J[(mat[j, i] - 1) * 2 + 1, (mat[j + 1, i] - 1) * 2] = calc_transmissibility_y(k_y, kr_g, mu_g, b_g,
                                                                                              params, i,
                                                                                              j) + calc_transmissibility_y(
                    k_y, kr_o, mu_o, b_o * rs, params, i, j) + D3_j_pos

                # Gas D4 derivative w.r.t. sg
                J[(mat[j, i] - 1) * 2 + 1, (mat[j + 1, i] - 1) * 2 + 1] = D4_j_pos

            # Upper block (j-1/2) elements
            if j > 0:
                # Oil D1 j-1 element
                dp_j_neg = (p_grids_n1[j - 1, i] - p_grids_n1[j, i])
                if p_grids_n1[j, i] < p_grids_n1[j - 1, i]:
                    D1_j_neg = dp_j_neg * calc_transmissibility_y(k_y, kr_o, mu_o, db_o, params, i, j - 1)
                    D2_j_neg = dp_j_neg * calc_transmissibility_y(k_y, dkr_o, mu_o, b_o, params, i, j - 1)
                    D3_j_neg_free = dp_j_neg * calc_transmissibility_y(k_y, kr_g, mu_g ** 2,
                                                                       (db_g * mu_g - b_g * dmu_g), params, i, j - 1)
                    D3_j_neg_sol = dp_j_neg * calc_transmissibility_y(k_y, kr_o, mu_o, drs * b_o + db_o * rs, params, i,
                                                                      j - 1)
                    D3_j_neg = D3_j_neg_free + D3_j_neg_sol
                    D4_j_neg = dp_j_neg * (calc_transmissibility_y(k_y, dkr_g, mu_g, b_g, params, i,
                                                                   j - 1) + calc_transmissibility_y(k_y, dkr_o, mu_o,
                                                                                                    rs * b_o, params, i,
                                                                                                    j - 1))
                    C1_j_neg = 0
                    C2_j_neg = 0
                    C3_j_neg = 0
                    C4_j_neg = 0
                else:
                    D1_j_neg = 0
                    D2_j_neg = 0
                    D3_j_neg = 0
                    D4_j_neg = 0
                    C1_j_neg = dp_j_neg * calc_transmissibility_y(k_y, kr_o, mu_o, db_o, params, i, j - 1)
                    C2_j_neg = dp_j_neg * calc_transmissibility_y(k_y, dkr_o, mu_o, b_o, params, i, j - 1)
                    C3_j_neg_free = dp_j_neg * calc_transmissibility_y(k_y, kr_g, mu_g ** 2,
                                                                       (db_g * mu_g - b_g * dmu_g), params, i, j - 1)
                    C3_j_neg_sol = dp_j_neg * calc_transmissibility_y(k_y, kr_o, mu_o, drs * b_o + db_o * rs, params, i,
                                                                      j - 1)
                    C3_j_neg = C3_j_neg_free + C3_j_neg_sol
                    C4_j_neg = dp_j_neg * (calc_transmissibility_y(k_y, dkr_g, mu_g, b_g, params, i,
                                                                   j - 1) + calc_transmissibility_y(k_y, dkr_o, mu_o,
                                                                                                    rs * b_o, params, i,
                                                                                                    j - 1))
                J[(mat[j, i] - 1) * 2, (mat[j - 1, i] - 1) * 2] = calc_transmissibility_y(k_y, kr_o, mu_o, b_o, params,
                                                                                          i, j - 1) + D1_j_neg

                # Oil D2 derivative w.r.t. sg
                J[(mat[j, i] - 1) * 2, (mat[j - 1, i] - 1) * 2 + 1] = D2_j_neg

                # Gas D3 derivative w.r.t. pressure
                J[(mat[j, i] - 1) * 2 + 1, (mat[j - 1, i] - 1) * 2] = calc_transmissibility_y(k_y, kr_g, mu_g, b_g,
                                                                                              params, i,
                                                                                              j - 1) + calc_transmissibility_y(
                    k_y, kr_o, mu_o, b_o * rs, params, i, j - 1) + D3_j_neg

                # Gas D4 derivative w.r.t. sg
                J[(mat[j, i] - 1) * 2 + 1, (mat[j - 1, i] - 1) * 2 + 1] = D4_j_neg

            ## Main Blocks (diagonal of matrix J)
            acc_par = dx[j, i] * dy[j, i] * dz[j, i] * por / props['time'].tstep / 5.615 / 0.001127

            # Main Diagonal 1
            diag_transmissibility1 = T[(mat[j, i] - 1) * 2, (mat[j, i] - 1) * 2] / 0.001127
            acc1 = acc_par * ((1 - sg_n1[j, i]) * db_o[j, i])
            J[(mat[j, i] - 1) * 2, (
                        mat[j, i] - 1) * 2] = diag_transmissibility1 + C1_i_neg + C1_i_pos + C1_j_neg + C1_j_pos - acc1

            # Main Diagonal 2
            acc2 = -acc_par * b_o[j, i]
            J[(mat[j, i] - 1) * 2, (mat[j, i] - 1) * 2 + 1] = C2_i_neg + C2_i_pos + C2_j_neg + C2_j_pos - acc2

            # Main Diagonal 3
            diag_transmissibility3 = T[(mat[j, i] - 1) * 2 + 1, (mat[j, i] - 1) * 2] / 0.001127
            acc3 = acc_par * (
                        sg_n1[j, i] * db_g[j, i] + (1 - sg_n1[j, i]) * (db_o[j, i] * rs[j, i] + b_o[j, i] * drs[j, i]))
            J[(mat[j, i] - 1) * 2 + 1, (
                        mat[j, i] - 1) * 2] = diag_transmissibility3 + C3_i_neg + C3_i_pos + C3_j_neg + C3_j_pos - acc3

            # Main Diagonal 4
            acc4 = acc_par * (b_g[j, i] - b_o[j, i] * rs[j, i])
            J[(mat[j, i] - 1) * 2 + 1, (mat[j, i] - 1) * 2 + 1] = C4_i_neg + C4_i_pos + C4_j_neg + C4_j_pos - acc4

    J = J * 0.001127
    return J


def construct_D(mat, params, props):
    # Construct accumulation matrix for every block
    b_o = params['b_o']
    b_g = params['b_g']
    rs = params['rs']
    p_grids_n = params['p_grids_n']
    sg_n = params['sg_n']
    sg_n1 = params['sg_n1']
    dx = params['dx']
    dy = params['dy']
    dz = params['dz']
    por = props['rock'].por

    m = mat.shape[0]
    n = mat.shape[1]

    D = np.zeros(m * n * 2)
    for j in range(m):
        for i in range(n):
            Vot = dx[j, i] * dy[j, i] * dz[j, i] / props['time'].tstep * por / 5.615
            D[(mat[j, i] - 1) * 2] = Vot * (
                        (1 - sg_n1[j, i]) * b_o[j, i] - (1 - sg_n[j, i]) * props['fluid'].calc_bo(p_grids_n[j, i]))
            g1 = sg_n1[j, i] * b_g[j, i] - sg_n[j, i] * props['fluid'].calc_bg(p_grids_n[j, i])
            g2 = (1 - sg_n1[j, i]) * rs[j, i] * b_o[j, i] - (1 - sg_n[j, i]) * props['fluid'].calc_bo(p_grids_n[j, i]) * \
                 props['fluid'].calc_rs(p_grids_n[j, i])
            D[(mat[j, i] - 1) * 2 + 1] = Vot * (g1 + g2)
    return D


def run_simulation(props):
    rock = props['rock']
    fluid = props['fluid']
    grid = props['grid']
    res = props['res']
    wells = props['well']
    sim_time = props['time']

    # Distribute properties (and their values) to the grid
    k_x = np.full((grid.Ny, grid.Nx), rock.kx)
    k_y = np.full((grid.Ny, grid.Nx), rock.ky)
    B_o = np.full((grid.Ny, grid.Nx), fluid.calc_b(res.p_init))
    mu = np.full((grid.Ny, grid.Nx), fluid.mu_o)
    p_grids = np.full((grid.Ny * grid.Nx, 1), res.p_init)
    params = {'k_x': k_x, 'k_y': k_y, 'B_o': B_o, 'mu': mu}

    # Construct transmissibility matrix T
    mat = np.reshape(np.arange(1, grid.Ny * grid.Nx + 1), (grid.Ny, grid.Nx))
    T = construct_T(mat, params, props)

    # Create matrix A = transmissibility matrix - accumulation matrix
    D = construct_D(mat, params, props)

    dx = res.Lx / grid.Nx
    dy = res.Lx / grid.Nx
    dz = res.Lx / grid.Nx
    V = dx * dy * dz
    accumulation = V * rock.por * fluid.c_o / 5.615 / (sim_time.tstep)
    A = T - np.eye(T.shape[0]) * accumulation

    # Assign well flow rate to Q matrix
    Q = np.zeros((T.shape[0], 1))
    for well in wells:
        Q[well.index_to_grid(grid.Nx)] = -well.q

    # Calculate right hand side
    p_n = np.full((grid.Ny * grid.Nx, 1), -accumulation * res.p_init)
    b = p_n - Q

    # Variable of interest: pressure in block (18,18)
    p_well_block = []

    # Time-loop
    for t in sim_time.timeint:
        print('evaluating t = %1.1f (days)' % t)
        p_well_block.append(p_grids[wells[0].index_to_grid(grid.Nx)])

        # Calculate pressure at time level n+1
        p_grids = (gmres(A, b))[0]
        p_grids = np.reshape(p_grids, (len(p_grids), 1))

        # Update B, b, and transmissibility matrix
        for i in range(grid.Nx):
            for j in range(grid.Ny):
                B_o[i, j] = fluid.calc_b(p_grids[ij_to_grid(i, j, grid.Nx)])
        params['B_o'] = B_o
        A = construct_T(mat, params, props)
        A = A - np.eye(A.shape[0]) * accumulation

        b = -accumulation * p_grids - Q
    return p_well_block, p_grids


def main():
    # Initialization
    tstep = 2  # day
    timeint = np.arange(0, 10, tstep)
    sim_time = prop_time(tstep=tstep,
                         timeint=timeint)
    rock = prop_rock(kx=np.array([50, 50, 50, 150, 150, 150]),  # permeability in x direction in mD
                     ky=np.array([200, 200, 200, 300, 300, 300]),  # permeability in y direction in mD
                     por=0.22,  # porosity in fraction
                     cr=0)  # 1/psi
    fluid = prop_fluid(c_o=0.8e-5,  # oil compressibility in 1/psi
                       mu_o=2.5,  # oil viscosity in cP
                       rho_o=49.1,  # lbm/ft3
                       p_bub=3500,  # pb in psi
                       p_atm=14.7)  # atmospheric pressure in psi
    grid = prop_grid(Nx=3,
                     Ny=2,
                     Nz=1)  # no of grid blocks in x, y, and z direction
    res = prop_res(Lx=1500,  # reservoir length in ft
                   Ly=1500,  # reservoir width in ft
                   Lz=200,  # reservoir height in ft
                   press_n=np.array([2500, 2525, 2550, 2450, 2475, 2500]),  # block pressure at time level n in psi
                   sg_n=np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5]),  # gas saturation at time level n
                   press_n1_k=np.array([2505, 2530, 2555, 2455, 2480, 2505]),
                   # block pressure at time level n+1, iteration k
                   sg_n1_k=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))  # gas saturation at time level n+1, iteration k
    well1 = prop_well(loc=(1, 1),  # well location
                      q=0)  # well flowrate in STB/D
    props = {'rock': rock, 'fluid': fluid, 'grid': grid, 'res': res, 'well': [well1], 'time': sim_time}

    '''/Reference
    press_n=np.array([2500, 2525, 2550, 2450, 2475, 2500]),  # block pressure at time level n in psi
    sg_n=np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5]),  # gas saturation at time level n
    press_n1_k=np.array([2505, 2530, 2555, 2455, 2480, 2505]),  # block pressure at time level n+1, iteration k
    sg_n1_k=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])) # gas saturation at time level n+1, iteration k
    /end Reference
    
    /Test_Data
    press_n=np.array([2400, 2425, 2450, 2350, 2375, 2400]),  # block pressure at time level n in psi
    sg_n=np.array([0, 0.15, 0.25, 0.35, 0.45, 0.55]),  # gas saturation at time level n
    press_n1_k=np.array([2405, 2430, 2455, 2355, 2380, 2405]),  # block pressure at time level n+1, iteration k
    sg_n1_k=np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65])) # gas saturation at time level n+1, iteration k
    /end_Test Data
    '''

    # Check Fluid and Rock Properties
    # fluid.plot_all()
    # rock.plot_all()

    ## Distribute properties (and their values) to the grid
    b_o = np.zeros(grid.Ny * grid.Nx)
    db_o = np.zeros(grid.Ny * grid.Nx)
    b_g = np.zeros(grid.Ny * grid.Nx)
    db_g = np.zeros(grid.Ny * grid.Nx)
    mu_g = np.zeros(grid.Ny * grid.Nx)
    dmu_g = np.zeros(grid.Ny * grid.Nx)
    rs = np.zeros(grid.Ny * grid.Nx)
    drs = np.zeros(grid.Ny * grid.Nx)
    kr_o = np.zeros(grid.Ny * grid.Nx)
    dkr_o = np.zeros(grid.Ny * grid.Nx)
    kr_g = np.zeros(grid.Ny * grid.Nx)
    dkr_g = np.zeros(grid.Ny * grid.Nx)
    for i in range(grid.Nx * grid.Ny):
        # Fluid and rock properties
        b_o[i] = fluid.calc_bo(res.press_n1_k[i])
        db_o[i] = fluid.calc_dbo(res.press_n1_k[i])
        b_g[i] = fluid.calc_bg(res.press_n1_k[i])
        db_g[i] = fluid.calc_dbg(res.press_n1_k[i])
        mu_o = np.full((grid.Ny, grid.Nx), fluid.mu_o)
        mu_g[i] = fluid.calc_mu_g(res.press_n1_k[i])
        dmu_g[i] = fluid.calc_dmu_g(res.press_n1_k[i])
        rs[i] = fluid.calc_rs(res.press_n1_k[i])
        drs[i] = fluid.calc_drs(res.press_n1_k[i])
        kr_o[i] = rock.calc_kro(res.sg_n1_k[i])
        dkr_o[i] = rock.calc_dkro(res.sg_n1_k[i])
        kr_g[i] = rock.calc_krg(res.sg_n1_k[i])
        dkr_g[i] = rock.calc_dkrg(res.sg_n1_k[i])

        # Grid size
        dx = np.full((grid.Ny, grid.Nx), props['res'].Lx / props['grid'].Nx)
        dy = np.full((grid.Ny, grid.Nx), props['res'].Ly / props['grid'].Ny)
        dz = np.full((grid.Ny, grid.Nx), props['res'].Lz / props['grid'].Nz)
    params = {'k_x': rock.kx, 'k_y': rock.ky, 'b_o': b_o, 'db_o': db_o, 'b_g': b_g, 'db_g': db_g,
              'mu_o': mu_o, 'mu_g': mu_g, 'dmu_g': dmu_g, 'rs': rs, 'drs': drs, 'p_grids_n': res.press_n,
              'p_grids_n1': res.press_n1_k,
              'sg_n': res.sg_n, 'sg_n1': res.sg_n1_k, 'kr_o': kr_o, 'dkr_o': dkr_o, 'kr_g': kr_g, 'dkr_g': dkr_g,
              'dx': dx, 'dy': dy, 'dz': dz}
    for p in params:
        params[p] = np.reshape(params[p], (grid.Ny, grid.Nx))
    mat = np.reshape(np.arange(1, grid.Ny * grid.Nx + 1), (grid.Ny, grid.Nx))

    # Construct matrix T (containing transmissibility terms)
    T = construct_T(mat, params)
    params.update({'T': T})
    p_n1_k = res.stack_ps(res.press_n1_k, res.sg_n1_k)

    # Construct matrix D (containing accumulation terms)
    D = construct_D(mat, params, props)
    p_n = res.stack_ps(res.press_n, res.sg_n)

    # Compute residual matrix
    R = np.dot(T, p_n1_k) - D

    # Compute Jacobian Matrix
    J = construct_J(mat, params, props)

    # Flip variables (pressure and sg) to match the reference case format
    R_flipped = flip_variables(R, 0)
    J_flipped = flip_variables(J, 1)

    # For Reference Case: Report mean absolute error and relative error
    j_dir = 'https://raw.githubusercontent.com/titaristanto/reservoir-simulation/master/reference_J.csv'
    r_dir = 'https://raw.githubusercontent.com/titaristanto/reservoir-simulation/master/reference_R.csv'
    J_reference = pd.read_csv(j_dir, header=None)
    R_reference = pd.read_csv(r_dir, header=None)

    mean_abs_error_R = mean_absolute_error(R_flipped, R_reference)
    mean_abs_error_J = mean_absolute_error(J_flipped, J_reference)

    relative_error_R = norm(np.reshape(R_flipped, (-1, 1)) - R_reference) / norm(R_reference)
    relative_error_J = norm(J_flipped - J_reference) / norm(J_reference)

    print('Mean abs error R =', mean_abs_error_R)
    print('Mean abs error J =', mean_abs_error_J)
    print('Relative error R =', relative_error_R)
    print('Relative error J =', relative_error_J)

    # Print residual and Jacobian into .csv files
    os.chdir('C:\\Users\\E460\\Documents\\Stanford\\Courses\\Winter 17\\Reservoir Simulation\\Phase 3b\\Result')
    np.savetxt('R.csv', R_flipped, delimiter=',')
    np.savetxt('J.csv', J_flipped, delimiter=',')

    # Sparse matrix
    J_ref_sparse = sparse.csr_matrix(J_reference)
    io.mmwrite('sparse_J_ref.csv', J_ref_sparse)
    J_result_sparse = sparse.csr_matrix(J_flipped)
    io.mmwrite('sparse_J_result.csv', J_result_sparse)

    R_ref_sparse = sparse.csr_matrix(R_reference)
    io.mmwrite('sparse_R_ref.csv', R_ref_sparse)
    R_result_sparse = sparse.csr_matrix(R_flipped)
    io.mmwrite('sparse_R_result.csv', R_result_sparse)


if __name__ == '__main__':
    main()
