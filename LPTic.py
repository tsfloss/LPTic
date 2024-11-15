import numpy as np

def linear_growth(z, Omega_m):
    E = lambda z, Omega_m: np.sqrt(Omega_m*(1.+z)**3. + (1.-Omega_m))
    
    zs_0 = np.logspace(-4,4,1000)
    zs_z = np.logspace(np.log10(z),4,1000)
    
    integrand_0 = (1.+zs_0) / (100.*E(zs_0, Omega_m))**3.
    integrand_z = (1.+zs_z) / (100.*E(zs_z, Omega_m))**3.
    
    D_0 = 5./2. * Omega_m * E(1e-4, Omega_m) * np.trapezoid(integrand_0, x=zs_0)
    D_z = 5./2. * Omega_m * E(z, Omega_m) * np.trapezoid(integrand_z, x=zs_z)
    
    return D_z / D_0

def LPTic(grid, BoxSize, z, Omega_m, kLin, PLin, n_LPT=2, seed=0):
    """
    Generate initial conditions for cosmological simulations using Lagrangian Perturbation Theory (LPT).

    Parameters
    ----------
    grid : int
        The number of grid points along each axis (grid resolution).
    BoxSize : float
        The physical size of the simulation box (in units of Mpc/h).
    z : float
        The redshift at which to generate the initial conditions.
    Omega_m : float
        Matter density parameter at redshift zero (Ωₘ₀).
    kLin : array_like
        1D array of wave numbers (in units of Mpc/h) corresponding to the linear power spectrum at redshift zero.
    PLin : array_like
        1D array of linear power spectrum values (in units of (Mpc/h)^3) corresponding to kLin.
    n_LPT : int
        Order of LPT to use (must be either 1 (Zeldovich/1LPT) or 2 (2LPT))
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    delta_r : ndarray
        Real-space linear density field at redshift zero.
    particle_pos : ndarray
        Array of particle positions at redshift z after applying 1LPT and 2LPT displacements, shape (N_particles, 3).

    Notes
    -----
    This function generates initial conditions using first- or second-order Lagrangian
    perturbation theory (LPT). It computes the displacement fields and applies
    them to particles initially placed on a regular grid.
    """

    assert n_LPT in (1,2), "n_LPT must be either 1 or 2"

    kF = 2*np.pi/BoxSize
    cell_size = BoxSize/grid
    
    D = linear_growth(z+1e-4, Omega_m)

    kx = 2*np.pi * np.fft.fftfreq(grid, BoxSize/grid)
    kz = 2*np.pi * np.fft.rfftfreq(grid, BoxSize/grid)
    kmesh = np.array(np.meshgrid(kx, kx, kz, indexing='ij'))
    kgrid = np.sqrt(kmesh[0]**2 + kmesh[1]**2 + kmesh[2]**2)

    sqrtPgrid = np.sqrt(np.interp(kgrid, kLin, PLin) / cell_size**3.)
    sqrtPgrid[0,0,0] = 0.

    np.random.seed(seed)
    delta_r = np.random.normal(0, 1, (grid, grid, grid))
    delta_k = np.fft.rfftn(delta_r, norm='ortho') * sqrtPgrid
    delta_r = np.fft.irfftn(delta_k, norm='ortho')

    kgrid[0,0,0] = 1.
    inv_k2 = 1/kgrid**2.
    phi_k = - delta_k * inv_k2

    #### 1LPT ####
    Psi_1LPT_k = - 1.j * phi_k * kmesh
    Psi_1LPT_r = np.fft.irfftn(Psi_1LPT_k, norm='ortho', axes=(-3,-2,-1))

    #### 2LPT ####
    if n_LPT > 1:
        phi_dxdx = 1.j * np.einsum("ijkl,mjkl->imjkl", kmesh, Psi_1LPT_k)
    
        # Pad to prevent aliasing
        phi_dxdx = np.fft.fftshift(phi_dxdx, axes=(-3,-2))
        phi_dxdx = np.pad(phi_dxdx, ((0,0), (0,0), (grid//4, grid//4), (grid//4, grid//4), (0, grid//4)),constant_values=0.+0.j)
        phi_dxdx = np.fft.ifftshift(phi_dxdx, axes=(-3,-2))
        phi_dxdx = np.fft.irfftn(phi_dxdx, axes=(-3,-2,-1), norm='ortho')
    
        # Compute 2LPT potential
        phi_2LPT = phi_dxdx[0,0]*phi_dxdx[1,1] - phi_dxdx[0,1]**2.
        phi_2LPT+= phi_dxdx[0,0]*phi_dxdx[2,2] - phi_dxdx[0,2]**2.
        phi_2LPT+= phi_dxdx[2,2]*phi_dxdx[1,1] - phi_dxdx[2,1]**2.
        phi_2LPT = np.fft.rfftn(phi_2LPT, axes=(-3,-2,-1))
    
        # Downsample after antialiased products
        phi_2LPT = np.fft.fftshift(phi_2LPT, axes=(-3,-2))
        phi_2LPT = phi_2LPT[grid//4:-grid//4, grid//4:-grid//4,:-grid//4]
        phi_2LPT = np.fft.ifftshift(phi_2LPT, axes=(-3,-2))    
    
        Psi_2LPT_r = -3/7 * np.fft.irfftn(- 1.j * phi_2LPT * kmesh * inv_k2, axes=(-3,-2,-1),)

    #### Displace Particles ####
    x = np.arange(0,grid)*cell_size
    particle_pos = np.array(np.meshgrid(x, x, x, indexing='ij'), dtype=np.float32)

    particle_pos += D * Psi_1LPT_r
    if n_LPT > 1:
        particle_pos += D**2. * Psi_2LPT_r

    particle_pos[particle_pos < 0.] += BoxSize
    particle_pos[particle_pos > BoxSize] -= BoxSize
    
    return delta_r, particle_pos.reshape(3,-1).T