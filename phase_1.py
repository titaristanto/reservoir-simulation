from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import gmres
import os, io, requests

class prop_rock(object):
    def __init__(self, kx=0, ky=0, por=0, cr=0):
        self.kx = kx
        self.ky = ky
        self.por = por
        self.cr = cr

class prop_fluid(object):
    def __init__(self, c_o=0, mu_o=0,rho_o=0):
        self.c_o = c_o
        self.mu_o = mu_o
        self.rho_o = rho_o
    def calc_b(self,p):
        return 1/(1+self.c_o*(p-14.7))

class prop_grid(object):
    def __init__(self, Nx=0, Ny=0, Nz=0):
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
    def grid_dimension_x(self,Lx):
        return self.Nx/Lx
    def grid_dimension_y(self,Ly):
        return self.Ny/Ly
    def grid_dimension_z(self,Lz):
        return self.Nz/Lz

class prop_res(object):
    def __init__(self, Lx=0, Ly=0, Lz=0, p_init=0):
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.p_init=p_init

class prop_well(object):
    def __init__(self, loc=0, q=0):
        self.loc = loc
        self.q = q

    def index_to_grid(self,Nx):
        return self.loc[1]*Nx+self.loc[0]

class prop_time(object):
    def __init__(self, tstep=0, timeint=0):
        self.tstep = tstep
        self.timeint = timeint

def load_data(filename):
    url='https://raw.githubusercontent.com/titaristanto/reservoir-simulation/master/eclipse%20bhp.csv'
    df = pd.read_csv(url)

    t=df.loc[:,['TIME']] # Time in simulation: DAY
    p=df.loc[:,['BPR:(18,18,1)']]
    return t,p

def calc_transmissibility_block_x(k_x,mu,B_o,props,i,j):
    dx=props['res'].Lx/props['grid'].Nx
    dy=props['res'].Ly/props['grid'].Ny
    dz=props['res'].Lz/props['grid'].Nz

    k_x=k_x[j,i]
    mu=mu[j,i]
    B_o=B_o[j,i]
    return k_x*dy*dz/mu/B_o/dx

def calc_transmissibility_block_y(k_y,mu,B_o,props,i,j):
    dx=props['res'].Lx/props['grid'].Nx
    dy=props['res'].Ly/props['grid'].Ny
    dz=props['res'].Lz/props['grid'].Nz

    k_y=k_y[j,i]
    mu=mu[j,i]
    B_o=B_o[j,i]
    return k_y*dx*dz/mu/B_o/dy

def calc_transmissibility_x(k_x,mu,B_o,props,i,j):
    dx=props['res'].Lx/props['grid'].Nx
    dy=props['res'].Ly/props['grid'].Ny
    dz=props['res'].Lz/props['grid'].Nz

    k_x=(k_x[j,i]+k_x[j,i+1])/2
    mu=(mu[j,i]+mu[j,i+1])/2
    B_o=(B_o[j,i]+B_o[j,i+1])/2
    return k_x*dy*dz/mu/B_o/dx

def calc_transmissibility_y(k_y,mu,B_o,props,i,j):
    dx=props['res'].Lx/props['grid'].Nx
    dy=props['res'].Ly/props['grid'].Ny
    dz=props['res'].Lz/props['grid'].Nz

    k_y=(k_y[j,i]+k_y[j+1,i])/2
    mu=(mu[j,i]+mu[j+1,i])/2
    B_o=(B_o[j,i]+B_o[j+1,i])/2
    return k_y*dx*dz/mu/B_o/dy

def ij_to_grid(i,j,Nx):
    return (i)+Nx*j

def construct_T(mat, params, props):
    k_x=params['k_x']
    k_y=params['k_y']
    B_o=params['B_o']
    mu=params['mu']

    m=mat.shape[0]
    n=mat.shape[1]
    A=np.zeros((m*n,m*n))
    for j in range(m):
        for i in range(n):
            # 2 neighbors in x direction
            if i<n-1:
                A[mat[j,i]-1,mat[j,i+1]-1]=calc_transmissibility_x(k_x,mu,B_o,props,i,j)
                A[mat[j,i+1]-1,mat[j,i]-1]=A[mat[j,i]-1,mat[j,i+1]-1]
            # 2 neighbors in y direction
            if j<m-1:
                A[mat[j,i]-1,mat[j+1,i]-1]=calc_transmissibility_y(k_y,mu,B_o,props,i,j)
                A[mat[j+1,i]-1,mat[j,i]-1]=A[mat[j,i]-1,mat[j+1,i]-1]

    for k in range(A.shape[0]):
        p=np.sum(A[k,:])*-1
        A[k,k]=p
    return A/887.5

def run_simulation(props):
    rock=props['rock']
    fluid=props['fluid']
    grid=props['grid']
    res=props['res']
    wells=props['well']
    sim_time=props['time']

    # Distribute properties (and their values) to the grid
    k_x=np.full((grid.Ny,grid.Nx),rock.kx)
    k_y=np.full((grid.Ny,grid.Nx),rock.ky)
    B_o=np.full((grid.Ny,grid.Nx),fluid.calc_b(res.p_init))
    mu=np.full((grid.Ny,grid.Nx),fluid.mu_o)
    p_grids=np.full((grid.Ny*grid.Nx,1),res.p_init)
    params={'k_x':k_x,'k_y':k_y,'B_o':B_o,'mu':mu}

    # Construct transmissibility matrix T
    mat=np.reshape(np.arange(1,grid.Ny*grid.Nx+1),(grid.Ny,grid.Nx))
    T=construct_T(mat, params, props)

    # Create matrix A = transmissibility matrix - accumulation matrix
    dx=res.Lx/grid.Nx
    dy=res.Lx/grid.Nx
    dz=res.Lx/grid.Nx
    V=dx*dy*dz
    accumulation=V*rock.por*fluid.c_o/5.615/(sim_time.tstep)
    A=T-np.eye(T.shape[0])*accumulation

    # Assign well flow rate to Q matrix
    Q=np.zeros((T.shape[0],1))
    for well in wells:
        Q[well.index_to_grid(grid.Nx)]=-well.q

    # Calculate right hand side
    p_n=np.full((grid.Ny*grid.Nx,1),-accumulation*res.p_init)
    b=p_n-Q

    # Variable of interest: pressure in block (18,18)
    p_well_block=[]

    # Time-loop
    for t in sim_time.timeint:
        print('evaluating t = %1.1f (days)' % t)
        p_well_block.append(p_grids[wells[0].index_to_grid(grid.Nx)])

        # Calculate pressure at time level n+1
        p_grids=(gmres(A,b))[0]
        p_grids=np.reshape(p_grids,(len(p_grids),1))

        # Update B, b, and transmissibility matrix
        for i in range(grid.Nx):
            for j in range(grid.Ny):
                B_o[i,j]=fluid.calc_b(p_grids[ij_to_grid(i,j,grid.Nx)])
        params['B_o']=B_o
        A=construct_T(mat, params, props)
        A=A-np.eye(A.shape[0])*accumulation

        b=-accumulation*p_grids-Q
    return p_well_block, p_grids

def plot_pressure(t,p_pred,label,color):
    # Plotting pressure v time
    plt.plot(t, p_pred, color=color, markeredgecolor=color,label=label)
    plt.xlabel("Time (days)")
    plt.ylabel("Block Pressure Cell (18,18)", fontsize=9)
    plt.legend(loc="best", prop=dict(size=8))
    plt.xlim(0,max(t))
    plt.ylim(0,max(p_pred))
    plt.grid(True)
    plt.draw()

def spatial_map(p_2D,title):
    plt.matshow(p_2D)
    plt.colorbar()
    plt.xlabel('grid in x-direction')
    plt.ylabel('grid in y-direction')
    plt.title(title)
    plt.draw()

def derivatives(t,p_pred,p_act,title):
    t_act=np.linspace(0,400,len(p_act))
    dp_act=abs(p_act[0]-p_act)
    dp_pred=abs(p_pred[0]-p_pred)

    # Calculate Derivatives
    p_der_act=np.zeros(len(p_act)-2)
    p_der_pred=np.zeros(len(p_pred)-2)
    for i in range(1,len(p_act)-1):
        p_der_act[i-1]=t_act[i]/(t_act[i+1]-t_act[i-1])*abs(dp_act[i+1]-dp_act[i-1])
    for i in range(1,len(p_pred)-1):
        p_der_pred[i-1]=t[i]/(t[i+1]-t[i-1])*abs(dp_pred[i+1]-dp_pred[i-1])
    plot_derivatives(t_act,t,dp_act,dp_pred,p_der_act,p_der_pred,title)

def plot_derivatives(t_act,t,dp_act,dp_pred,p_der_act,p_der_pred,title):
    plt.figure()
    plt.loglog(t_act, dp_act, 'k-',linewidth=3,label='Actual dP')
    plt.loglog(t, dp_pred, 'ro',label='Predicted dP')
    plt.loglog(t_act[1:-1], p_der_act, 'k*',linewidth=3,label='Actual Derivative')
    plt.loglog(t[1:-1], p_der_pred, 'gx',label='Predicted Derivative')
    plt.title(title)
    plt.xlabel('dt (hours)')
    plt.ylabel('dP & Derivative')
    plt.legend(loc="best", prop=dict(size=12))
    plt.grid()
    plt.draw()

def main():
    # Initialization
    tstep=1 # day
    timeint=np.arange(0,401,tstep)
    sim_time=prop_time(tstep=tstep,
                       timeint=timeint)
    rock=prop_rock(kx=200, # permeability in x direction in mD
                   ky=100, # permeability in y direction in mD
                   por=0.25, # porosity in fraction
                   cr=0) # 1/psi
    fluid=prop_fluid(c_o=1.2 * 10 ** -5, # oil compressibility in 1/psi
                     mu_o=2, # oil viscosity in cP
                     rho_o=49.1) # lbm/ft3
    grid=prop_grid(Nx=35,
                   Ny=35,
                   Nz=1) # no of grid blocks in x, y , and z
    res=prop_res(Lx=3500, # reservoir length in ft
                 Ly=3500, # reservoir width in ft
                 Lz=100, # reservoir height in ft
                 p_init=6000) # initial block pressure in psi
    well1=prop_well(loc=(18, 18), # well location
                   q=2000) # well flowrate in STB/D
    props={'rock':rock,'fluid':fluid,'grid':grid,'res':res,'well':[well1],'time':sim_time}

    # Load data from Eclipse
    t_ecl,p_ecl=load_data('eclipse bhp.csv')

    ### Main Case: 1 producer
    # Run simulation
    print('Running Main Case: 1 producer')
    p_well_block, p_grids=run_simulation(props)

    # Plotting
    plt.figure()
    plot_pressure(timeint,p_well_block,label='Phase-1 Simulator',color='red')
    plot_pressure(t_ecl.values,p_ecl.values,label='Eclipse',color='black')
    plt.title('Main Case: 1 producer')

    p_2D=np.reshape(p_grids,(grid.Nx,grid.Ny))
    spatial_map(p_2D,'Main Case: 1 producer')
    plt.title('Main Case: 1 producer')

    derivatives(timeint,np.matrix(p_well_block),p_ecl.values,'Main Case: 1 producer')

    ## Additional Case: 3 producers
    # Define 2 more producers
    well2=prop_well(loc=(3, 5), q=300)
    well3=prop_well(loc=(33, 30), q=2500)
    props['well']=[well1,well2,well3]

    # Run simulation
    print('Running Additional Case: 3 producers')
    p_well_block, p_grids=run_simulation(props)

    # Plotting
    plt.figure()
    plot_pressure(timeint,p_well_block,label='Phase-1 Simulator',color='red')
    plot_pressure(t_ecl.values,p_ecl.values,label='Eclipse',color='black')
    plt.title('Additional Case: 3 producers')

    p_2D=np.reshape(p_grids,(grid.Nx,grid.Ny))
    spatial_map(p_2D,'Additional Case: 3 producers')

    derivatives(timeint,np.matrix(p_well_block),p_ecl.values,'Additional Case: 3 producers')

    ## Additional Case: 1 producer 1 injector
    # Define 1 injector
    well4=prop_well(loc=(3, 5), q=-2000)
    props['well']=[well1,well4]

    # Run simulation
    print('Running Additional Case: 1 producer 1 injector')
    p_well_block, p_grids=run_simulation(props)

    # Plotting
    plt.figure()
    plot_pressure(timeint,p_well_block,label='Phase-1 Simulator',color='red')
    plot_pressure(t_ecl.values,p_ecl.values,label='Eclipse',color='black')
    plt.title('Additional Case: 1 producer 1 injector')

    p_2D=np.reshape(p_grids,(grid.Nx,grid.Ny))
    spatial_map(p_2D,'Additional Case: 1 producer 1 injector')

    derivatives(timeint,np.matrix(p_well_block),p_ecl.values,'Additional Case: 1 producer 1 injector')
    plt.show(block=True)

if __name__ == '__main__':
    main()
