# single_phase_homogeneous_isothermal_2D_reservoir

This script is an implementation of a discretized single-phase homogeneous isothermal 2D reservoir 
equation: <img src="https://render.githubusercontent.com/render/math?math=T p^{(n+1)}=D(p^{(n+1)}-p^{(n)})+Q">

, where T is a transmissibility matrix, D is an accumulation
matrix, and Q is a source/sink matrix (which represents well(s)). p^(n+1) is a vector of unknown 
(in this case, pressure) at time level n and p^(n) is the same variable at current time level n.
This case assumes no capillary pressure, isotropic permeability, and constant viscosity. 
Well treatment is not included. This code treats the unknown (pressure) implicitly and transmissibilities 
explicitly.

Generalized Minimal Residual iteration (GMRES) is used to solve the linear matrix system, 
which is an iterative method for the numerical solution of a nonsymmetric system of 
linear equations. This function solves for p at time level n+1. This new pressure vector 
updates formation volume factor B_o in every grid, which also updates transmissibility matrix, 
so then we can solve for pressure at the next time level. This process continues until 
it reaches the end of simulation time.

The main simulation loop in this file calls the `run_simulation()` function for
simulating the reservoir dynamics. The `main()` function runs three different example cases: 
- 1 producer
- 3 producers
- 1 producer and 1 injector

Variable of interest demonstrated here includes block pressure (in the middle of the reservoir)
and spatial pressure.

# complete_two_phase_heterogeneous_isothermal_2D_reservoir

This script handles two-phase problem (oil and gas), heterogeneous grid properties, and isothermal 2D reservoir. Both pressure and transmissibilities are treated implicitly and solved using GMRES. 

Implementation examples are provided on the script:
- BHP control demo
- Oil rate and BHP control demo
- Timestep interval study
- Grid sensitivity study
- Heterogeneous permeability demo
- Two producers demo
