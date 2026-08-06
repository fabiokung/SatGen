import numba
numba.config.DISABLE_JIT = False
from numba import njit, prange
import numpy as np
from scipy.interpolate import CubicHermiteSpline, InterpolatedUnivariateSpline
import time
from tqdm import tqdm
from scipy.integrate import quad
from scipy.optimize import brentq

_G_NB = np.float64(4.4985e-06)  # [kpc^3 Gyr^-2 Msun^-1]

# Analytic relation between Plummer R0 and 3-D half-mass radius r_half:
#   M(<r) = M * r^3 / (r^2 + R0^2)^(3/2)  =>  r_half = R0 / sqrt(2^(2/3) - 1)
#   Inverted: R0 = r_half * sqrt(2^(2/3) - 1)
_PLUMMER_R0_FROM_RHALF = np.sqrt(2.0 ** (2.0 / 3.0) - 1.0)  # ≈ 0.7664

# Yoshida (1990) 4th-order symplectic coefficients
# Composition of 3 leapfrog sub-steps; 3 force evals per sub-step; O(dt^4) phase error.
_W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))                         # ≈  1.3512
_W0 = -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0))      # ≈ -1.7024
_YC = np.array([_W1 / 2.0,
                (_W0 + _W1) / 2.0,
                (_W0 + _W1) / 2.0,
                _W1 / 2.0])          # drift coefficients (4 drifts)
_YD = np.array([_W1, _W0, _W1])     # kick coefficients  (3 kicks)

# Standalone copies of the constants and cosmology needed by NFW, so this
# module doesn't depend on SatGen's config.py/cosmo.py (fiducial values match
# config.py: h=0.7, Om=0.3, OL=0.7, rhoc0=277.5 h^2 Msun kpc^-3).
_COSMO_H  = 0.7
_COSMO_OM = 0.3
_COSMO_OL = 0.7
_RHOC0 = 277.5
_FOUR_PI = 4. * np.pi
_FOUR_PI_G = 4. * np.pi * _G_NB
_THREE_PI_OVER_16G = 3. * np.pi / (16. * _G_NB)
_EPS = 1e-3

def _rhoc(z, h=_COSMO_H, Om=_COSMO_OM, OL=_COSMO_OL):
    """Critical density [Msun kpc^-3] at redshift z."""
    return _RHOC0 * h**2 * (Om*(1.+z)**3 + OL)

_x_interpolator = None
_x_interpolator_min = None
_x_interpolator_max = None

def initialize_subhalos(profile, Msub, fc, rmax=None, seed=None):
    """
    Initialize subhalo positions and velocities given a profile.

    Positions are drawn from the Dehnen density distribution via analytic
    CDF inversion.  Velocities are drawn from an isotropic Gaussian with
    radial dispersion sigma(r) given by the profile's velocity-dispersion
    method (assuming sigma_r = sigma_theta = sigma_phi).

    Parameters
    ----------
    profile : with the same conventions as the SatGen profiles
        profile of the host (UFD).  Must expose .rs, .Mh, .M(r),
        and .sigma(r).
    Msub : float
        Mass of each subhalo [M_sun].
    fc : float
        Subhalo mass fraction (0 < fc <= 1).
    rmax : float, optional
        Truncation radius [kpc] for position sampling.  If None the virial
        radius stored in profile.rh is used when available; otherwise
        sampling is restricted to 100 * profile.rs.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pos : ndarray, shape (Nsub, 3)
        Cartesian positions [kpc].
    vel : ndarray, shape (Nsub, 3)
        Cartesian velocities [kpc/Gyr].
    """
    rng = np.random.default_rng(seed)

    MUFD = profile.Mh

    Nsub = int(fc * MUFD / Msub)
    if Nsub == 0:
        return (np.empty((0, 3)), np.empty((0, 3)), np.empty(0))

    rs = profile.rs

    # --- truncation radius ---
    if rmax is None:
        rmax = getattr(profile, 'rh', 100. * rs)

    # CDF at truncation: P(<rmax) = (x_max / (1 + x_max))^3
    # x_max = rmax / rs
    # cdf_max = (x_max / (1. + x_max)) ** 3.

    # --- sample radii via numeric CDF inversion ---
    # normalization:
    cdf_max = profile.M(rmax)/MUFD
    # CDF: u = (x / (1+x))^3  =>  x = u^(1/3) / (1 - u^(1/3))
    u = rng.uniform(0., cdf_max, Nsub)
    cdf_inv = np.vectorize(lambda p: brentq(lambda x: profile.M(x)/MUFD - p, 0., rmax))
    r_samp = cdf_inv(u)

    # --- sample angular positions uniformly on the sphere ---
    cos_theta = rng.uniform(-1., 1., Nsub)
    sin_theta = np.sqrt(1. - cos_theta ** 2.)
    phi = rng.uniform(0., 2. * np.pi, Nsub)

    pos = np.column_stack([
        r_samp * sin_theta * np.cos(phi),
        r_samp * sin_theta * np.sin(phi),
        r_samp * cos_theta,
    ])

    # --- sample velocities ---
    # sigma(r) is not yet implemented as a standalone function; we call the
    # profile method which provides the isotropic radial velocity dispersion.
    sigma_r = profile.sigma(r_samp)   # shape (Nsub,)

    # Draw v_r, v_theta, v_phi independently from N(0, sigma_r)
    v_r     = rng.normal(0., sigma_r)
    v_theta = rng.normal(0., sigma_r)
    v_phi   = rng.normal(0., sigma_r)

    # Convert spherical velocity components to Cartesian
    vel = np.column_stack([
        v_r * sin_theta * np.cos(phi) + v_theta * cos_theta * np.cos(phi) - v_phi * np.sin(phi),
        v_r * sin_theta * np.sin(phi) + v_theta * cos_theta * np.sin(phi) + v_phi * np.cos(phi),
        v_r * cos_theta              - v_theta * sin_theta,
    ])

    return pos, vel

# NFW class taken from SatGen
class NFW(object):
    """
    Class that implements the Navarro, Frenk, & White (1997) profile:

        rho(R,z) = rho_crit * delta_char / [(r/r_s) * (1+r/r_s)^2]
                 = rho_0 / [(r/r_s) * (1+r/r_s)^2]
    
    in a cylindrical frame (R,phi,z), where 
    
        r = sqrt(R^2 + z^2)
        r_s: scale radius, at which d ln rho(r) / d ln(r) = -2
        rho_crit: critical density of the Universe
        delta_char = Delta_halo / 3 * c^3 / f(c), where c = R_vir / r_s 
            is the concentration parameter
    
    Syntax:
    
        halo = NFW(M,c,Delta=200.,z=0.,sf=1.)
        
    where 
    
        M: halo mass [M_sun], where halo is defined as spherical 
            overdensity of Delta times critical density (float) 
        c: halo concentration (float)
        Delta: average overdensity of the halo, in multiples of the 
            critical density of the Universe (float)
            (default 200.)
        z: redshift (float) (default 0.)
        sf: Suppression factor used for reducing the overall
                density of the halo while preserving its shape, used
                when a disk is added in order to preserve total mass
                of the system
    
    Attributes:
    
        .Mh: halo mass [M_sun]
        .ch: halo concentration
        .Deltah: spherical overdensity wrt instantaneous critical density
        .z: redshift
        .rhoc: critical density [M_sun kpc^-3]
        .rhoh: average density of halo [M_sun kpc^-3]
        .rh: halo radius within which density is Delta times rhoc [kpc]
        .rs: scale radius [kpc]
        .rmax: radius at which maximum circular velocity is reached [kpc]
        .Vmax: maximum circular velocity [kpc/Gyr]
        .s001: logarithmic density slope at 0.01 halo radius
        
    Methods:
    
        .rho(R,z=0.): density [M_sun kpc^-3] at radius r=sqrt(R^2+z^2)
        .s(R,z=0.): logarithmic density slope at radius r=sqrt(R^2+z^2)
        .M(R,z=0.): mass [M_sun] enclosed in radius r=sqrt(R^2+z^2)
        .rhobar(R,z=0.): mean density [M_sun kpc^-3] within radius 
            r=sqrt(R^2+z^2)
        .tdyn(R,z=0.): dyn. time [Gyr] within radius r = sqrt(R^2+z^2)
        .Phi(R,z=0.): potential [(kpc/Gyr)^2] at radius r=sqrt(R^2+z^2)
        .fgrav(R,z): grav. acceleration [(kpc/Gyr)^2 kpc^-1] at (R,z) 
        .Vcirc(R,z=0.): circ. vel. [kpc/Gyr] at radius r=sqrt(R^2+z^2)
        .sigma(R,z=0.): vel. disp. [kpc/Gyr] at radius r=sqrt(R^2+z^2)      
        .d2Phidr2(R,z=0.): second radial derivative of potential [1/Gyr^2]
            at radius r=sqrt(R^2+r^2)
    
    HISTORY: Arthur Fangzhou Jiang (2016-10-24, HUJI)
             Arthur Fangzhou Jiang (2016-10-30, HUJI)
             Arthur Fangzhou Jiang (2019-08-24, HUJI)
    """
    def __init__(self,M,c,Delta=200.,z=0.,sf=1.):
        """
        Initialize NFW profile.
        
        Syntax:
        
            halo = NFW(M,c,Delta=200.,z=0.,sf=1.)
        
        where
        
            M: halo mass [M_sun] (float), 
            c: halo concentration (float),        
            Delta: spherical overdensity with respect to the critical 
                density of the universe (default is 200.)         
            z: redshift (float)
            sf: Suppression factor used for reducing the overall
                density of the halo while preserving its shape, used
                when a disk is added in order to preserve total mass
                of the system
        """
        # input attributes
        self.Mh = M 
        self.ch = c
        self.Deltah = Delta
        self.z = z
        self.sf = sf
        #
        # derived attributes
        self.rhoc = _rhoc(z)
        self.rhoh = self.Deltah * self.rhoc
        self.rh = (3.*self.Mh / (_FOUR_PI*self.rhoh))**(1./3.)
        self.rs = self.rh / self.ch
        self.rmax = self.rs * 2.163
        self.rho0 = self.sf*self.rhoc*self.Deltah/3.*self.ch**3./self.f(self.ch)
        self.Phi0 = -_FOUR_PI_G*self.rho0*self.rs**2.
        self.Vmax = self.Vcirc(self.rmax)
        self.s001 = self.s(0.01*self.rh)
    def f(self,x):
        """
        Auxiliary method for NFW profile: f(x) = ln(1+x) - x/(1+x)
    
        Syntax:
    
            .f(x)
        
        where
        
            x: dimensionless radius r/r_s (float or array)
        """
        return np.log(1.+x) - x/(1.+x) 
    def rho(self,R,z=0.):
        """
        Density [M_sun kpc^-3] at radius r = sqrt(R^2 + z^2). 
            
        Syntax:
        
            .rho(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R**2.+z**2.) 
        x = r / self.rs
        return self.rho0 / (x * (1.+x)**2.)
    def s(self,R,z=0.):
        """
        Logarithmic density slope 
            
            - d ln rho / d ln r 
        
        at radius r = sqrt(R^2 + z^2). 
        
        Syntax:
        
            .s(R,z=0.)

        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R**2.+z**2.) 
        x = r / self.rs
        return 1. + 2*x / (1.+x)
    def M(self,R,z=0.):
        """
        Mass [M_sun] within radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .M(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)       
        """
        r = np.sqrt(R**2.+z**2.)
        x = r/self.rs
        return _FOUR_PI*self.rho0*self.rs**3. * self.f(x)
    def rhobar(self,R,z=0.):
        """
        Average density [M_sun kpc^-3] within radius r = sqrt(R^2 + z^2). 
            
        Syntax:
        
            .rhobar(R,z=0.)
        
        where 
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)   
        """
        r = np.sqrt(R**2.+z**2.)
        x = r/self.rs
        return 3.*self.rho0 * self.f(x)/x**3.
    def tdyn(self,R,z=0.):
        """
        Dynamical time [Gyr] within radius r = sqrt(R^2 + z^2).

        Syntax:
        
            .tdyn(R,z=0.)
            
        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)     
        """
        return np.sqrt(_THREE_PI_OVER_16G / self.rhobar(R,z))
    def Phi(self,R,z=0.):
        """
        Potential [(kpc/Gyr)^2] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .Phi(R,z=0.)  

        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        x = r/self.rs
        return self.Phi0 * np.log(1.+x)/x
    def otherMassDefinition(self,Delta=200.):
        """
        Computes the mass, radius, and concentration of the fixed,
        physical halo under a new spherical overdensity definition.
        Since rho0 is fixed, determines the cnew=rnew/rs that solves:

            rho0 = [Delta * rhoc / 3] * (rnew/rs)**3 / f(rnew/rs)**3

        Implementation based on Benedikt Diemer's COLOSSUS code.

        Syntax:

            .otherMassDefinition(Delta=200.)

        where
            Delta: Spherical overdensity in units of the critical
                   overdensity at the redshift that the halo was
                   initialized at (float).

        Return:

            Mnew: Mass within new overdensity (Msun, float)
            rnew: Radius corresponding to new overdensity (kpc, float)
            cnew: Concentration relative to new overdensity.
        """

        global _x_interpolator
        global _x_interpolator_min
        global _x_interpolator_max

        if _x_interpolator is None:
            table_x = np.logspace(4.0, -4.0, 1000)
            table_y = self.f(table_x) * 3.0 / table_x**3
            _x_interpolator = InterpolatedUnivariateSpline(table_y,
                                                          table_x, k=3)
            knots = _x_interpolator.get_knots()
            _x_interpolator_min = knots[0]
            _x_interpolator_max = knots[-1]

        dens_threshold = Delta * self.rhoc
        y = dens_threshold / self.rho0

        if(y < _x_interpolator_min):
            raise Exception("Requested overdensity %.2e cannot be evaluated\
                             for scale density %.2e, out of range." \
                             % (y, _x_interpolator_min))
        elif(y > _x_interpolator_max):
            raise Exception("Requested overdensity %.2e cannot be evaluated\
                             for scale density %.2e, out of range." \
                             % (y, _x_interpolator_max))

        cnew = _x_interpolator(y)
        rnew = cnew * self.rs
        Mnew = self.M(rnew)
        return Mnew, rnew, cnew
    def fgrav(self,R,z):
        """
        gravitational acceleration [(kpc/Gyr)^2 kpc^-1] at location (R,z)
        
            [- d Phi(R,z) / d R, 0, - d Phi(R,z) / d z]
        
        Syntax:
            
            .fgrav(R,z)
            
        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
        
        Note that unlike the other methods, where z is optional with a 
        default of 0, here z must be specified.
        
        Return:
        
            R-component of gravitational acceleration
            phi-component of gravitational acceleration
            z-component of gravitational acceleration
        """
        r = np.sqrt(R**2.+z**2.)
        x = r / self.rs   
        fac = self.Phi0 * (self.f(x)/x) / r**2.
        return fac*R, fac*0., fac*z
    def Vcirc(self,R,z=0.):
        """
        Circular velocity [kpc/Gyr] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .Vcirc(R,z=0.)
            
        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        return np.sqrt(r*-self.fgrav(r,0.)[0])
    def sigma(self,R,z=0.):
        """
        Velocity dispersion [kpc/Gyr] at radius r = sqrt(R^2 + z^2), 
        assuming isotropic velicity dispersion tensor, and following the 
        Zentner & Bullock (2003) fitting function:
        
            sigma(x) = V_max 1.4393 x^0.345 / (1 + 1.1756 x^0.725)
            
        where x = r/r_s.
        
        Syntax:
            
            .sigma(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        x = r / self.rs
        return self.Vmax*1.4393*x**0.354/(1.+1.1756*x**0.725)
    def sigma_accurate(self,R,z=0.,beta=0.):
        """
        Velocity dispersion [kpc/Gyr].
        
        Syntax:
            
            .sigma(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
            beta: anisotropy parameter (default=0., i.e., isotropic)
        """
        r = np.sqrt(R**2.+z**2.)
        x = r / self.rs
        if isinstance(x,list) or isinstance(x,np.ndarray):
            I = []
            for xx in x:
                II = quad(self.dIdx_sigma, xx, np.inf,args=(beta,))[0]
                I.append(II)
            I = np.array(I)
        else:
            I = quad(self.dIdx_sigma, x, np.inf,args=(beta,))[0]
        f = self.f(x)
        sigmasqr = -self.Phi0 / x**(2.*beta-1) *(1.+x)**2 * I
        return np.sqrt(sigmasqr)
    def dIdx_sigma(self,x,beta):
        """
        Integrand for the integral in the velocity dispersion.
        """
        f = self.f(x)
        return x**(2.*beta-3.) * f / (1.+x)**2
    def dlnsigmasqrdlnr_accurate(self,R,z=0.,beta=0.):
        """
        d ln sigma^2 / d ln r
        """
        r = np.sqrt(R**2.+z**2.)
        r1 = r * (1.+_EPS)
        r2 = r * (1.-_EPS)
        y1 = np.log(self.sigma_accurate(r1))
        y2 = np.log(self.sigma_accurate(r2))
        return (y1-y2)/(r1-r2)
    def d2Phidr2(self,R,z=0.):
        """
        Second radial derivative of the gravitational potential [1/Gyr^2] 
        computed at r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .d2Phidr2(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)       
        """
        r = np.sqrt(R**2.+z**2.)
        x = r/self.rs

        f = (2.*np.log(1. + x) - x*(2. + 3.*x)/(1. + x)**2.) / r**3.
        return self.Phi0 * self.rs * f

def Plummer(r, M, R0star):
    return 3. * M/(4.*np.pi * R0star**3)  / (1 + (r/R0star)**2)**(5/2)

def sigmastar_Plummer(r, R0star, Mstar, profile):
    """Radial velocity dispersion of Plummer stars in a Dehnen host potential.

    Solves the isotropic Jeans equation:
        sigma_r^2(r) = (1 / rho_*(r)) * int_r^inf rho_*(r') * G*M(r') / r'^2 dr'
    where rho_* is the Plummer profile and M(r) is from the Dehnen host.
    """
    def integrand(rp):
        return Plummer(rp, Mstar, R0star) * _G_NB * profile.M(rp) / rp**2
    integral, _ = quad(integrand, r, np.inf, limit=80)
    rho_r = Plummer(r, Mstar, R0star)
    return np.sqrt(max(0.0, integral / rho_r)) if rho_r > 0.0 else 0.0

def init_plummer(Mstar, R0star, profile, m_star=1.0, seed=None):
    """
    Initialize N = Mstar / m_star stars following the Plummer profile.

    Positions are sampled via the analytic inverse CDF of the Plummer
    cumulative mass profile (derived from stellar_heating.Plummer):

        u = M(<r)/Mstar = (r/R0)^3 / (1 + (r/R0)^2)^(3/2)
        => r = R0star * u^(1/3) / sqrt(1 - u^(2/3)),  u ~ Uniform(0, 1)

    Velocities are drawn isotropically; each Cartesian component is sampled
    from N(0, sigmastar_Plummer(r, R0star, Mstar, profile)).

    Parameters
    ----------
    Mstar   : float        -- total stellar mass [Msun]
    R0star  : float        -- Plummer scale radius [kpc]
    profile : Dehnen/NFW   -- host potential used in sigmastar_Plummer
    m_star  : float        -- mass per particle [Msun]; default 1.0
    seed    : int or None  -- random seed

    Returns
    -------
    pos : ndarray (N, 3)  -- Cartesian positions [kpc]
    vel : ndarray (N, 3)  -- Cartesian velocities [kpc/Gyr]
    """
    rng = np.random.default_rng(seed)
    N   = int(round(Mstar / m_star))

    # --- positions: analytic inverse CDF of the Plummer mass profile ---
    u   = rng.uniform(0.0, 1.0, N)
    r   = R0star * u**(1.0/3.0) / np.sqrt(1.0 - u**(2.0/3.0))

    cos_theta = rng.uniform(-1.0, 1.0, N)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi       = rng.uniform(0.0, 2.0*np.pi, N)

    pos = np.column_stack([
        r * sin_theta * np.cos(phi),
        r * sin_theta * np.sin(phi),
        r * cos_theta,
    ])

    # --- velocities: isotropic Gaussian with sigma = sigmastar_Plummer(r) ---
    sigma = np.array([sigmastar_Plummer(ri, R0star, Mstar, profile) for ri in r])
    vel   = rng.normal(0.0, sigma[:, None], size=(N, 3))

    return pos, vel

def estimateProfile(posArray, massArray, r_bins):
    r = np.linalg.norm(posArray, axis=1)
    mass_in_bins, _ = np.histogram(r, bins=r_bins, weights=massArray)
    shell_volumes = (4/3) * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
    return mass_in_bins / shell_volumes

def upsample(xvMACHO, timesteps, Nsteps):
    """
    Upsample a MACHO trajectory to a finer time grid using CubicHermiteSpline.

    Parameters
    ----------
    xvMACHO   : (N_MACHO, N_step+1, 6)  cylindrical [R, phi, z, VR, Vphi, Vz]
    timesteps : (N_step,)                original output times [Gyr] (excluding t=0)
    Nsteps    : int                      desired number of output steps

    Returns
    -------
    xvMACHO_new   : (N_MACHO, Nsteps+1, 6)  upsampled cylindrical trajectory
    timesteps_new : (Nsteps,)                new output times [Gyr] (excluding t=0)
    """
    tmax = timesteps[-1]
    t_knots = np.concatenate([[0.], timesteps])  # (N_step+1,) — include t=0

    # Convert cylindrical to Cartesian positions and velocities
    R_m, phi_m, z_m    = xvMACHO[:, :, 0], xvMACHO[:, :, 1], xvMACHO[:, :, 2]
    VR_m, Vphi_m, Vz_m = xvMACHO[:, :, 3], xvMACHO[:, :, 4], xvMACHO[:, :, 5]

    cart_pos = np.stack([
        R_m * np.cos(phi_m),
        R_m * np.sin(phi_m),
        z_m,
    ], axis=-1).transpose(1, 0, 2)  # (N_step+1, N_MACHO, 3)

    cart_vel = np.stack([
        VR_m * np.cos(phi_m) - Vphi_m * np.sin(phi_m),
        VR_m * np.sin(phi_m) + Vphi_m * np.cos(phi_m),
        Vz_m,
    ], axis=-1).transpose(1, 0, 2)  # (N_step+1, N_MACHO, 3)

    # Single spline over all MACHOs and coordinates simultaneously
    spline = CubicHermiteSpline(t_knots, cart_pos, cart_vel)

    # Evaluate positions and velocities at new fine grid
    timesteps_new = np.linspace(0., tmax, Nsteps + 1)[1:]       # (Nsteps,)
    t_eval        = np.concatenate([[0.], timesteps_new])         # (Nsteps+1,)

    pos_new = spline(t_eval)     # (Nsteps+1, N_MACHO, 3)
    vel_new = spline(t_eval, 1)  # first derivative = Cartesian velocities

    # Convert back to cylindrical
    x_n, y_n, z_n   = pos_new[:, :, 0], pos_new[:, :, 1], pos_new[:, :, 2]
    vx_n, vy_n, vz_n = vel_new[:, :, 0], vel_new[:, :, 1], vel_new[:, :, 2]

    R_n   = np.sqrt(x_n**2 + y_n**2)
    phi_n = np.arctan2(y_n, x_n)
    VR_n   = (x_n * vx_n + y_n * vy_n) / R_n
    Vphi_n = (x_n * vy_n - y_n * vx_n) / R_n

    xvMACHO_new = np.stack(
        [R_n, phi_n, z_n, VR_n, Vphi_n, vz_n], axis=-1
    ).transpose(1, 0, 2)  # (N_MACHO, Nsteps+1, 6)

    return xvMACHO_new, timesteps_new

@njit(cache=True)
def _dekel_M_enc_nb(r, Mh, rs, c, alpha):
    """Dekel+ (2016) enclosed mass at radius r from the subhalo centre."""
    if r < 1.0e-10:
        r = 1.0e-10
    if rs < 1.0e-10:
        return 0.0
    x = r / rs
    sqrt_x = x ** 0.5
    chi_x = sqrt_x / (1.0 + sqrt_x)
    sqrt_c = c ** 0.5
    chi_c = sqrt_c / (1.0 + sqrt_c)
    return Mh * (chi_x / chi_c) ** (2.0 * (3.0 - alpha))


@njit(parallel=True, cache=True)
def _accel_cart_clumps_nb(pos, macho_cart, sub_Mh, sub_rs, sub_c, sub_alpha,
                           log_r_grid, M_enc_grid):
    """Acceleration kernel with per-subhalo Dekel enclosed-mass forces.

    Subhalo force uses G * M_Dekel(<r_star-sub) / r^2, not a softened point mass.
    Host force is unchanged: log-linear lookup on the precomputed M_enc grid.
    """
    N_stars   = pos.shape[0]
    N_macho   = macho_cart.shape[0]
    N_grid    = log_r_grid.shape[0]
    log_r_min = log_r_grid[0]
    inv_dlr   = (N_grid - 1) / (log_r_grid[N_grid - 1] - log_r_min)
    accel     = np.empty((N_stars, 3))

    for s in prange(N_stars):
        x = pos[s, 0]; y = pos[s, 1]; z = pos[s, 2]
        r2 = x * x + y * y + z * z
        r  = r2 ** 0.5
        if r < 1e-10:
            r = 1e-10

        # Host: log-linear interpolation of M_enc(r)
        log_r = np.log(r)
        t = (log_r - log_r_min) * inv_dlr
        if t < 0.0:
            t = 0.0
        elif t > N_grid - 1 - 1e-12:
            t = N_grid - 1 - 1e-12
        i0   = int(t)
        frac = t - i0
        M_r  = (1.0 - frac) * M_enc_grid[i0] + frac * M_enc_grid[i0 + 1]

        fac_h = -_G_NB * M_r / (r2 * r)
        ax = fac_h * x
        ay = fac_h * y
        az = fac_h * z

        # Subhalo contributions: Dekel enclosed mass at star–subhalo separation
        for j in range(N_macho):
            if sub_Mh[j] == 0.0:
                continue
            dx  = x - macho_cart[j, 0]
            dy  = y - macho_cart[j, 1]
            dz  = z - macho_cart[j, 2]
            rsj = (dx * dx + dy * dy + dz * dz) ** 0.5
            if rsj < 1e-10:
                rsj = 1e-10
            M_sub = _dekel_M_enc_nb(rsj, sub_Mh[j], sub_rs[j], sub_c[j], sub_alpha[j])
            fac_m = -_G_NB * M_sub / (rsj * rsj * rsj)
            ax += fac_m * dx
            ay += fac_m * dy
            az += fac_m * dz

        accel[s, 0] = ax
        accel[s, 1] = ay
        accel[s, 2] = az

    return accel


@njit(cache=True)
def _yoshida4_step_clumps_nb(pos_c, vel_c, macho_cart,
                               sub_Mh, sub_rs, sub_c, sub_alpha,
                               log_r_grid, M_enc_grid,
                               sub_dt, YC, YD, n_sub):
    """n_sub Yoshida-4 sub-steps using Dekel subhalo profiles."""
    for _ in range(n_sub):
        pos_c = pos_c + YC[0] * vel_c * sub_dt
        a     = _accel_cart_clumps_nb(pos_c, macho_cart, sub_Mh, sub_rs, sub_c, sub_alpha,
                                       log_r_grid, M_enc_grid)
        vel_c = vel_c + YD[0] * a * sub_dt

        pos_c = pos_c + YC[1] * vel_c * sub_dt
        a     = _accel_cart_clumps_nb(pos_c, macho_cart, sub_Mh, sub_rs, sub_c, sub_alpha,
                                       log_r_grid, M_enc_grid)
        vel_c = vel_c + YD[1] * a * sub_dt

        pos_c = pos_c + YC[2] * vel_c * sub_dt
        a     = _accel_cart_clumps_nb(pos_c, macho_cart, sub_Mh, sub_rs, sub_c, sub_alpha,
                                       log_r_grid, M_enc_grid)
        vel_c = vel_c + YD[2] * a * sub_dt

        pos_c = pos_c + YC[3] * vel_c * sub_dt
    return pos_c, vel_c


@njit(parallel=True, cache=True)
def _accel_cart_nb(pos, macho_cart, mMA, log_r_grid, M_enc_grid, softening_sq):
    """Numba-parallel (prange over stars) acceleration kernel.

    Host enclosed mass is looked up via log-linear interpolation on a
    precomputed grid so the kernel stays free of Python objects.
    """
    N_stars   = pos.shape[0]
    N_macho   = macho_cart.shape[0]
    N_grid    = log_r_grid.shape[0]
    log_r_min = log_r_grid[0]
    inv_dlr   = (N_grid - 1) / (log_r_grid[N_grid - 1] - log_r_min)
    accel     = np.empty((N_stars, 3))

    for s in prange(N_stars):
        x = pos[s, 0]; y = pos[s, 1]; z = pos[s, 2]
        r2 = x * x + y * y + z * z
        r  = r2 ** 0.5
        if r < 1e-10:
            r = 1e-10

        # Host: log-linear interpolation of M_enc(r)
        log_r = np.log(r)
        t = (log_r - log_r_min) * inv_dlr
        if t < 0.0:
            t = 0.0
        elif t > N_grid - 1 - 1e-12:
            t = N_grid - 1 - 1e-12
        i0   = int(t)
        frac = t - i0
        M_r  = (1.0 - frac) * M_enc_grid[i0] + frac * M_enc_grid[i0 + 1]

        fac_h = -_G_NB * M_r / (r2 * r)
        ax = fac_h * x
        ay = fac_h * y
        az = fac_h * z

        # MACHO point-mass contributions (serial, typical N_macho is small)
        for j in range(N_macho):
            dx  = x - macho_cart[j, 0]
            dy  = y - macho_cart[j, 1]
            dz  = z - macho_cart[j, 2]
            r2m = dx * dx + dy * dy + dz * dz + softening_sq
            r3m = r2m * (r2m ** 0.5)
            fac_m = -_G_NB * mMA / r3m
            ax += fac_m * dx
            ay += fac_m * dy
            az += fac_m * dz

        accel[s, 0] = ax
        accel[s, 1] = ay
        accel[s, 2] = az

    return accel


@njit(cache=True)
def _yoshida4_step_nb(pos_c, vel_c, macho_cart, mMA,
                      log_r_grid, M_enc_grid, softening_sq,
                      sub_dt, YC, YD, n_sub):
    """n_sub Yoshida-4 sub-steps; calls parallel _accel_cart_nb each kick."""
    for _ in range(n_sub):
        pos_c = pos_c + YC[0] * vel_c * sub_dt
        a     = _accel_cart_nb(pos_c, macho_cart, mMA, log_r_grid, M_enc_grid, softening_sq)
        vel_c = vel_c + YD[0] * a * sub_dt

        pos_c = pos_c + YC[1] * vel_c * sub_dt
        a     = _accel_cart_nb(pos_c, macho_cart, mMA, log_r_grid, M_enc_grid, softening_sq)
        vel_c = vel_c + YD[1] * a * sub_dt

        pos_c = pos_c + YC[2] * vel_c * sub_dt
        a     = _accel_cart_nb(pos_c, macho_cart, mMA, log_r_grid, M_enc_grid, softening_sq)
        vel_c = vel_c + YD[2] * a * sub_dt

        pos_c = pos_c + YC[3] * vel_c * sub_dt   # final drift
    return pos_c, vel_c

def _compute_n_substeps(host, dt_outer, safety=100.0):
    """Minimum Yoshida sub-steps per outer timestep.

    Yoshida-4 is stable for  omega_max * dt_sub < ~1.28 (harmonic oscillator).
    We use the conservative bound  omega_max * dt_sub < 1 / safety.
    """
    omega_max = np.sqrt(_G_NB * host.Mh / host.rs**3)   # Gyr^-1
    dt_stable = 1.0 / (omega_max * safety)
    return max(1, int(np.ceil(dt_outer / dt_stable)))

def stellar_simulation_nb(xvMACHO, posStar, velStar, mMA, host, timesteps, r0i,
                          softening=0.0, n_substeps=None, Nsteps=None, earlyStop=True):
    """Numba-parallel drop-in replacement for stellar_simulation_fast.

    Identical algorithm (Yoshida-4 + optional CHS upsampling) but the
    per-step acceleration is compiled with numba and parallelised over stars
    via prange.  The host M(<r) is precomputed on a 1000-point log grid and
    looked up with log-linear interpolation inside the JIT kernel.

    First call triggers JIT compilation (~few seconds); subsequent calls are
    fast.  Signature is identical to stellar_simulation_fast.
    """
    bins   = np.logspace(-4, 0, 40)
    Nstars = len(posStar)

    # Optional upsampling (same as stellar_simulation_fast)
    if Nsteps is not None and Nsteps > len(timesteps):
        xvMACHO, timesteps = upsample(xvMACHO, timesteps, Nsteps)

    Nsteps_actual = len(timesteps)

    R0starList   = np.zeros(Nsteps_actual + 1)
    HLRadiusList = np.zeros(Nsteps_actual + 1)
    R0starList[0]   = r0i
    HLRadiusList[0] = np.percentile(np.linalg.norm(posStar, axis=1), 50)

    # Pre-convert MACHO cylindrical -> Cartesian (once)
    R_m, phi_m, z_m = xvMACHO[:, :, 0], xvMACHO[:, :, 1], xvMACHO[:, :, 2]
    macho_cart_all  = np.ascontiguousarray(
        np.stack([R_m * np.cos(phi_m), R_m * np.sin(phi_m), z_m], axis=-1),
        dtype=np.float64,
    )  # (N_MACHO, N_step+1, 3)

    # Precompute host M_enc on a log-spaced grid for the numba kernel
    r_nb      = np.logspace(-5, 2, 1000, dtype=np.float64)
    M_enc_nb  = np.ascontiguousarray(host.M(r_nb), dtype=np.float64)
    log_r_nb  = np.ascontiguousarray(np.log(r_nb),  dtype=np.float64)
    soft_sq   = np.float64(softening ** 2)

    # Sub-step count (same heuristic as stellar_simulation_fast)
    dt_outer = float(timesteps[0])
    if n_substeps is None:
        n_substeps = _compute_n_substeps(host, dt_outer)
    sub_dt    = dt_outer / n_substeps
    omega_max = np.sqrt(_G_NB * host.Mh / host.rs**3)
    print(f'  omega_max = {omega_max:.1f} Gyr⁻¹   '
          f'dt_outer = {1e3*dt_outer:.1f} Myr   '
          f'n_substeps = {n_substeps}   '
          f'sub_dt = {1e3*sub_dt:.2f} Myr   '
          f'total force evals/step = {3*n_substeps}')

    pos_c = np.ascontiguousarray(posStar, dtype=np.float64)
    vel_c = np.ascontiguousarray(velStar, dtype=np.float64)
    YC_nb = np.ascontiguousarray(_YC, dtype=np.float64)
    YD_nb = np.ascontiguousarray(_YD, dtype=np.float64)

    pos_hist = np.empty((Nstars, Nsteps_actual + 1, 3))
    vel_hist = np.empty((Nstars, Nsteps_actual + 1, 3))
    pos_hist[:, 0, :] = pos_c
    vel_hist[:, 0, :] = vel_c

    # Warm up: trigger JIT compilation before the timed loop
    print('>>> compiling numba kernels (first call may take a few seconds) ...')
    _dummy_pos, _dummy_vel = _yoshida4_step_nb(
        pos_c.copy(), vel_c.copy(),
        np.ascontiguousarray(macho_cart_all[:, 0, :], dtype=np.float64),
        np.float64(mMA), log_r_nb, M_enc_nb, soft_sq,
        np.float64(sub_dt), YC_nb, YD_nb, 1,
    )
    print('>>> evolving (Yoshida-4, numba-parallel) ...')
    tprevious = 0.0
    t1 = time.time()

    for i, t in tqdm(enumerate(timesteps)):
        dt_i     = t - tprevious
        sub_dt_i = np.float64(dt_i / n_substeps)
        macho_c  = np.ascontiguousarray(macho_cart_all[:, i, :], dtype=np.float64)

        pos_c, vel_c = _yoshida4_step_nb(
            pos_c, vel_c, macho_c,
            np.float64(mMA), log_r_nb, M_enc_nb, soft_sq,
            sub_dt_i, YC_nb, YD_nb, n_substeps,
        )

        tprevious = t
        pos_hist[:, i + 1, :] = pos_c
        vel_hist[:, i + 1, :] = vel_c

        rcurrent          = np.linalg.norm(pos_c, axis=1)
        r_half            = np.percentile(rcurrent, 50)
        HLRadiusList[i+1] = r_half
        R0starList[i+1]   = r_half * _PLUMMER_R0_FROM_RHALF

        if R0starList[i+1] > 50 * r0i and HLRadiusList[i+1] > 50 * r0i and earlyStop:
            print(f"Warning: at t={t:.2f} Gyr, R0*={R0starList[i+1]:.4f} kpc "
                  f"and r_half={HLRadiusList[i+1]:.4f} kpc are both >50x r0i. Stopping.")
            R0starList[i+2:]     = R0starList[i+1]
            HLRadiusList[i+2:]   = HLRadiusList[i+1]
            pos_hist[:, i+2:, :] = pos_c[:, None, :]
            vel_hist[:, i+2:, :] = vel_c[:, None, :]
            break

    t2 = time.time()
    print(f'    time = {t2 - t1:.4f} s')
    print(f"r0i    = {r0i:.4f} kpc")
    print(f"R0star = {R0starList[-1]:.4f} kpc  (x{R0starList[-1] / r0i:.3f})")

    # Vectorised cylindrical conversion (same as stellar_simulation_fast)
    x, y, z    = pos_hist[:, :, 0], pos_hist[:, :, 1], pos_hist[:, :, 2]
    vx, vy, vz = vel_hist[:, :, 0], vel_hist[:, :, 1], vel_hist[:, :, 2]
    R_h   = np.sqrt(x**2 + y**2)
    phi_h = np.arctan2(y, x)
    VR_h  = (x * vx + y * vy) / np.maximum(R_h, 1e-10)
    Vp_h  = (x * vy - y * vx) / np.maximum(R_h, 1e-10)
    xvList = np.stack([R_h, phi_h, z, VR_h, Vp_h, vz], axis=-1)

    dens_final = estimateProfile(pos_c, np.ones(Nstars), bins)

    return R0starList, HLRadiusList, dens_final, xvList, timesteps

def stellar_simulation_no_internal_stars_nb(xvMACHO, posStar, velStar, mMA, host, timesteps, r0i,
                          softening=0.0, n_substeps=None, Nsteps=None, earlyStop=True, ratio=1.0):
    """Same as stellar_simulation_nb, but MACHOs interior to the stellar
    system are excluded from the force on the stars each step: only MACHOs
    with r > ratio * R0star(t) contribute.
    """
    bins   = np.logspace(-4, 0, 40)
    Nstars = len(posStar)

    # Optional upsampling (same as stellar_simulation_fast)
    if Nsteps is not None and Nsteps > len(timesteps):
        xvMACHO, timesteps = upsample(xvMACHO, timesteps, Nsteps)

    Nsteps_actual = len(timesteps)

    R0starList   = np.zeros(Nsteps_actual + 1)
    HLRadiusList = np.zeros(Nsteps_actual + 1)
    R0starList[0]   = r0i
    HLRadiusList[0] = np.percentile(np.linalg.norm(posStar, axis=1), 50)

    # Pre-convert MACHO cylindrical -> Cartesian (once)
    R_m, phi_m, z_m = xvMACHO[:, :, 0], xvMACHO[:, :, 1], xvMACHO[:, :, 2]
    macho_cart_all  = np.ascontiguousarray(
        np.stack([R_m * np.cos(phi_m), R_m * np.sin(phi_m), z_m], axis=-1),
        dtype=np.float64,
    )  # (N_MACHO, N_step+1, 3)

    # Precompute host M_enc on a log-spaced grid for the numba kernel
    r_nb      = np.logspace(-5, 2, 1000, dtype=np.float64)
    M_enc_nb  = np.ascontiguousarray(host.M(r_nb), dtype=np.float64)
    log_r_nb  = np.ascontiguousarray(np.log(r_nb),  dtype=np.float64)
    soft_sq   = np.float64(softening ** 2)

    # Sub-step count (same heuristic as stellar_simulation_fast)
    dt_outer = float(timesteps[0])
    if n_substeps is None:
        n_substeps = _compute_n_substeps(host, dt_outer)
    sub_dt    = dt_outer / n_substeps
    omega_max = np.sqrt(_G_NB * host.Mh / host.rs**3)
    print(f'  omega_max = {omega_max:.1f} Gyr⁻¹   '
          f'dt_outer = {1e3*dt_outer:.1f} Myr   '
          f'n_substeps = {n_substeps}   '
          f'sub_dt = {1e3*sub_dt:.2f} Myr   '
          f'total force evals/step = {3*n_substeps}')

    pos_c = np.ascontiguousarray(posStar, dtype=np.float64)
    vel_c = np.ascontiguousarray(velStar, dtype=np.float64)
    YC_nb = np.ascontiguousarray(_YC, dtype=np.float64)
    YD_nb = np.ascontiguousarray(_YD, dtype=np.float64)

    pos_hist = np.empty((Nstars, Nsteps_actual + 1, 3))
    vel_hist = np.empty((Nstars, Nsteps_actual + 1, 3))
    pos_hist[:, 0, :] = pos_c
    vel_hist[:, 0, :] = vel_c

    # Warm up: trigger JIT compilation before the timed loop
    print('>>> compiling numba kernels (first call may take a few seconds) ...')
    _dummy_pos, _dummy_vel = _yoshida4_step_nb(
        pos_c.copy(), vel_c.copy(),
        np.ascontiguousarray(macho_cart_all[:, 0, :], dtype=np.float64),
        np.float64(mMA), log_r_nb, M_enc_nb, soft_sq,
        np.float64(sub_dt), YC_nb, YD_nb, 1,
    )
    print('>>> evolving (Yoshida-4, numba-parallel) ...')
    tprevious = 0.0
    t1 = time.time()

    for i, t in tqdm(enumerate(timesteps)):
        dt_i     = t - tprevious
        sub_dt_i = np.float64(dt_i / n_substeps)
        curr_macho = macho_cart_all[:, i, :]
        curr_macho_r = np.sqrt(curr_macho[:,0]**2 + curr_macho[:,1]**2 + curr_macho[:,2]**2)
        macho_c  = np.ascontiguousarray(curr_macho[curr_macho_r > ratio*R0starList[i]], dtype=np.float64)

        pos_c, vel_c = _yoshida4_step_nb(
            pos_c, vel_c, macho_c,
            np.float64(mMA), log_r_nb, M_enc_nb, soft_sq,
            sub_dt_i, YC_nb, YD_nb, n_substeps,
        )

        tprevious = t
        pos_hist[:, i + 1, :] = pos_c
        vel_hist[:, i + 1, :] = vel_c

        rcurrent          = np.linalg.norm(pos_c, axis=1)
        r_half            = np.percentile(rcurrent, 50)
        HLRadiusList[i+1] = r_half
        R0starList[i+1]   = r_half * _PLUMMER_R0_FROM_RHALF

        if R0starList[i+1] > 50 * r0i and HLRadiusList[i+1] > 50 * r0i and earlyStop:
            print(f"Warning: at t={t:.2f} Gyr, R0*={R0starList[i+1]:.4f} kpc "
                  f"and r_half={HLRadiusList[i+1]:.4f} kpc are both >50x r0i. Stopping.")
            R0starList[i+2:]     = R0starList[i+1]
            HLRadiusList[i+2:]   = HLRadiusList[i+1]
            pos_hist[:, i+2:, :] = pos_c[:, None, :]
            vel_hist[:, i+2:, :] = vel_c[:, None, :]
            break

    t2 = time.time()
    print(f'    time = {t2 - t1:.4f} s')
    print(f"r0i    = {r0i:.4f} kpc")
    print(f"R0star = {R0starList[-1]:.4f} kpc  (x{R0starList[-1] / r0i:.3f})")

    # Vectorised cylindrical conversion (same as stellar_simulation_fast)
    x, y, z    = pos_hist[:, :, 0], pos_hist[:, :, 1], pos_hist[:, :, 2]
    vx, vy, vz = vel_hist[:, :, 0], vel_hist[:, :, 1], vel_hist[:, :, 2]
    R_h   = np.sqrt(x**2 + y**2)
    phi_h = np.arctan2(y, x)
    VR_h  = (x * vx + y * vy) / np.maximum(R_h, 1e-10)
    Vp_h  = (x * vy - y * vx) / np.maximum(R_h, 1e-10)
    xvList = np.stack([R_h, phi_h, z, VR_h, Vp_h, vz], axis=-1)

    dens_final = estimateProfile(pos_c, np.ones(Nstars), bins)

    return R0starList, HLRadiusList, dens_final, xvList, timesteps

def _dekel_M_enc_grid(r_grid, Mh, rs, c, alpha):
    """Dekel+ (2016) enclosed mass evaluated at each radius in r_grid."""
    x     = r_grid / rs
    chi_x = x**0.5 / (1.0 + x**0.5)
    chi_c = c**0.5  / (1.0 + c**0.5)
    return Mh * (chi_x / chi_c) ** (2.0 * (3.0 - alpha))


def stellar_simulation_clumps_nb(xvMACHO, posStar, velStar,
                                  subhalo_M, subhalo_c, subhalo_alpha,
                                  subhalo_rh,
                                  host, timesteps, r0i,
                                  host_Mh=None, host_rh=None,
                                  host_c=None, host_alpha=None,
                                  n_substeps=None, Nsteps=None, earlyStop=True):
    """Yoshida-4 stellar N-body with time-varying Dekel subhalo profiles.

    Each subhalo's gravitational force on stars is computed from its Dekel
    enclosed mass at the star–subhalo separation, rather than a point mass.
    Profile parameters are linearly interpolated when the trajectory is
    upsampled; the trajectory itself is upsampled with CubicHermiteSpline.

    Parameters
    ----------
    xvMACHO      : (N_sub, N_step+1, 6)   cylindrical [R,phi,z,VR,Vphi,Vz]
    posStar      : (N_star, 3)             initial star positions [kpc]
    velStar      : (N_star, 3)             initial star velocities [kpc/Gyr]
    subhalo_M    : (N_sub, N_step+1)       virial mass [Msun] at each input step
    subhalo_c    : (N_sub, N_step+1)       Dekel concentration at each input step
    subhalo_alpha: (N_sub, N_step+1)       Dekel inner slope at each input step
    subhalo_rh   : (N_sub, N_step+1)       virial radius of subhalos [kpc]
    host         : Dekel/NFW profile       background host potential (used as
                                           fallback when host_* arrays are None)
    timesteps    : (N_step,)               output times [Gyr] (excluding t=0)
    r0i          : float                   initial Plummer scale radius [kpc]
    host_Mh      : (N_step+1,) or None     host virial mass [Msun] at each input
                                           step (including t=0); if provided the
                                           host potential evolves in time
    host_rh      : (N_step+1,) or None     host virial radius [kpc]
    host_c       : (N_step+1,) or None     host Dekel concentration
    host_alpha   : (N_step+1,) or None     host Dekel inner slope
    n_substeps   : int or None             Yoshida sub-steps per outer step
    Nsteps       : int or None             upsample trajectory to this many steps
    earlyStop    : bool                    stop early if stellar system dissolves

    Returns
    -------
    R0starList, HLRadiusList, dens_final, xvList, timesteps
    """
    bins   = np.logspace(-4, 0, 40)
    Nstars = len(posStar)
    N_sub  = xvMACHO.shape[0]

    subhalo_M     = np.asarray(subhalo_M,     dtype=np.float64)
    subhalo_c     = np.asarray(subhalo_c,     dtype=np.float64)
    subhalo_alpha = np.asarray(subhalo_alpha, dtype=np.float64)
    subhalo_rh = np.asarray(subhalo_rh, dtype=np.float64)
    subhalo_rs = subhalo_rh / subhalo_c

    evolving_host = (host_Mh is not None)
    if evolving_host:
        host_Mh    = np.asarray(host_Mh,    dtype=np.float64)
        host_rh    = np.asarray(host_rh,    dtype=np.float64)
        host_c     = np.asarray(host_c,     dtype=np.float64)
        host_alpha = np.asarray(host_alpha, dtype=np.float64)

    # Upsample trajectory (CHS) and profile params (linear) to a finer grid
    if Nsteps is not None and Nsteps > len(timesteps):
        t_knots_orig = np.concatenate([[0.], timesteps])
        xvMACHO, timesteps = upsample(xvMACHO, timesteps, Nsteps)
        t_fine = np.concatenate([[0.], timesteps])
        subhalo_M     = np.vstack([np.interp(t_fine, t_knots_orig, subhalo_M[j])
                                   for j in range(N_sub)])
        subhalo_c     = np.vstack([np.interp(t_fine, t_knots_orig, subhalo_c[j])
                                   for j in range(N_sub)])
        subhalo_alpha = np.vstack([np.interp(t_fine, t_knots_orig, subhalo_alpha[j])
                                   for j in range(N_sub)])
        subhalo_rs    = np.vstack([np.interp(t_fine, t_knots_orig, subhalo_rs[j])
                                   for j in range(N_sub)])
        if evolving_host:
            host_Mh    = np.interp(t_fine, t_knots_orig, host_Mh)
            host_rh    = np.interp(t_fine, t_knots_orig, host_rh)
            host_c     = np.interp(t_fine, t_knots_orig, host_c)
            host_alpha = np.interp(t_fine, t_knots_orig, host_alpha)

    Nsteps_actual = len(timesteps)


    R0starList   = np.zeros(Nsteps_actual + 1)
    HLRadiusList = np.zeros(Nsteps_actual + 1)
    R0starList[0]   = r0i
    HLRadiusList[0] = np.percentile(np.linalg.norm(posStar, axis=1), 50)

    # Pre-convert subhalo cylindrical -> Cartesian (once)
    R_m, phi_m, z_m = xvMACHO[:, :, 0], xvMACHO[:, :, 1], xvMACHO[:, :, 2]
    macho_cart_all  = np.ascontiguousarray(
        np.stack([R_m * np.cos(phi_m), R_m * np.sin(phi_m), z_m], axis=-1),
        dtype=np.float64,
    )  # (N_sub, N_step+1, 3)

    # Precompute host M_enc on a log-spaced grid for the numba kernel
    r_nb     = np.logspace(-5, 2, 1000, dtype=np.float64)
    log_r_nb = np.ascontiguousarray(np.log(r_nb), dtype=np.float64)
    if evolving_host:
        M_enc_nb = np.ascontiguousarray(
            _dekel_M_enc_grid(r_nb, host_Mh[0], host_rh[0] / host_c[0],
                              host_c[0], host_alpha[0]), dtype=np.float64)
    else:
        M_enc_nb = np.ascontiguousarray(host.M(r_nb), dtype=np.float64)

    dt_outer = float(timesteps[0])
    if n_substeps is None:
        if evolving_host:
            host_rs_all  = host_rh / host_c
            omega_all    = np.sqrt(_G_NB * host_Mh / host_rs_all**3)
            dt_stable    = 1.0 / (omega_all.max() * 100.0)
            n_substeps   = max(1, int(np.ceil(dt_outer / dt_stable)))
        else:
            n_substeps = _compute_n_substeps(host, dt_outer)
    sub_dt = dt_outer / n_substeps
    if evolving_host:
        omega_max = np.sqrt(_G_NB * host_Mh[0] / (host_rh[0] / host_c[0])**3)
    else:
        omega_max = np.sqrt(_G_NB * host.Mh / host.rs**3)
    print(f'  omega_max = {omega_max:.1f} Gyr⁻¹   '
          f'dt_outer = {1e3*dt_outer:.1f} Myr   '
          f'n_substeps = {n_substeps}   '
          f'sub_dt = {1e3*sub_dt:.2f} Myr   '
          f'total force evals/step = {3*n_substeps}')

    pos_c = np.ascontiguousarray(posStar, dtype=np.float64)
    vel_c = np.ascontiguousarray(velStar, dtype=np.float64)
    YC_nb = np.ascontiguousarray(_YC, dtype=np.float64)
    YD_nb = np.ascontiguousarray(_YD, dtype=np.float64)

    pos_hist = np.empty((Nstars, Nsteps_actual + 1, 3))
    vel_hist = np.empty((Nstars, Nsteps_actual + 1, 3))
    pos_hist[:, 0, :] = pos_c
    vel_hist[:, 0, :] = vel_c

    # Warm up: trigger JIT compilation before the timed loop
    print('>>> compiling numba kernels (first call may take a few seconds) ...')
    _dummy_pos, _dummy_vel = _yoshida4_step_clumps_nb(
        pos_c.copy(), vel_c.copy(),
        np.ascontiguousarray(macho_cart_all[:, 0, :], dtype=np.float64),
        np.ascontiguousarray(subhalo_M[:, 0],     dtype=np.float64),
        np.ascontiguousarray(subhalo_rs[:, 0],    dtype=np.float64),
        np.ascontiguousarray(subhalo_c[:, 0],     dtype=np.float64),
        np.ascontiguousarray(subhalo_alpha[:, 0], dtype=np.float64),
        log_r_nb, M_enc_nb,
        np.float64(sub_dt), YC_nb, YD_nb, 1,
    )
    print('>>> evolving (Yoshida-4, numba-parallel, Dekel subhalos) ...')
    tprevious = 0.0
    t1 = time.time()

    for i, t in tqdm(enumerate(timesteps)):
        dt_i     = t - tprevious
        sub_dt_i = np.float64(dt_i / n_substeps)
        macho_c  = np.ascontiguousarray(macho_cart_all[:, i, :], dtype=np.float64)

        if evolving_host:
            M_enc_nb = np.ascontiguousarray(
                _dekel_M_enc_grid(r_nb, host_Mh[i], host_rh[i] / host_c[i],
                                  host_c[i], host_alpha[i]), dtype=np.float64)

        pos_c, vel_c = _yoshida4_step_clumps_nb(
            pos_c, vel_c, macho_c,
            np.ascontiguousarray(subhalo_M[:, i],     dtype=np.float64),
            np.ascontiguousarray(subhalo_rs[:, i],    dtype=np.float64),
            np.ascontiguousarray(subhalo_c[:, i],     dtype=np.float64),
            np.ascontiguousarray(subhalo_alpha[:, i], dtype=np.float64),
            log_r_nb, M_enc_nb,
            sub_dt_i, YC_nb, YD_nb, n_substeps,
        )

        tprevious = t
        pos_hist[:, i + 1, :] = pos_c
        vel_hist[:, i + 1, :] = vel_c

        rcurrent          = np.linalg.norm(pos_c, axis=1)
        r_half            = np.percentile(rcurrent, 50)
        HLRadiusList[i+1] = r_half
        R0starList[i+1]   = r_half * _PLUMMER_R0_FROM_RHALF

        if R0starList[i+1] > 50 * r0i and HLRadiusList[i+1] > 50 * r0i and earlyStop:
            print(f"Warning: at t={t:.2f} Gyr, R0*={R0starList[i+1]:.4f} kpc "
                  f"and r_half={HLRadiusList[i+1]:.4f} kpc are both >50x r0i. Stopping.")
            R0starList[i+2:]     = R0starList[i+1]
            HLRadiusList[i+2:]   = HLRadiusList[i+1]
            pos_hist[:, i+2:, :] = pos_c[:, None, :]
            vel_hist[:, i+2:, :] = vel_c[:, None, :]
            break

    t2 = time.time()
    print(f'    time = {t2 - t1:.4f} s')
    print(f"r0i    = {r0i:.4f} kpc")
    print(f"R0star = {R0starList[-1]:.4f} kpc  (x{R0starList[-1] / r0i:.3f})")

    x, y, z    = pos_hist[:, :, 0], pos_hist[:, :, 1], pos_hist[:, :, 2]
    vx, vy, vz = vel_hist[:, :, 0], vel_hist[:, :, 1], vel_hist[:, :, 2]
    R_h   = np.sqrt(x**2 + y**2)
    phi_h = np.arctan2(y, x)
    VR_h  = (x * vx + y * vy) / np.maximum(R_h, 1e-10)
    Vp_h  = (x * vy - y * vx) / np.maximum(R_h, 1e-10)
    xvList = np.stack([R_h, phi_h, z, VR_h, Vp_h, vz], axis=-1)

    dens_final = estimateProfile(pos_c, np.ones(Nstars), bins)

    return R0starList, HLRadiusList, dens_final, xvList, timesteps


@njit(parallel=True, cache=True)
def _accel_cart_sub_pointmass_nb(pos, macho_cart, sub_Mh,
                                  log_r_grid, M_enc_grid, softening_sq):
    """Acceleration kernel with per-subhalo point-mass forces.

    Subhalo force uses G * M_j / (r^2 + eps^2)^(3/2).  Host force is the
    same log-linear M_enc lookup as all other kernels.
    """
    N_stars   = pos.shape[0]
    N_macho   = macho_cart.shape[0]
    N_grid    = log_r_grid.shape[0]
    log_r_min = log_r_grid[0]
    inv_dlr   = (N_grid - 1) / (log_r_grid[N_grid - 1] - log_r_min)
    accel     = np.empty((N_stars, 3))

    for s in prange(N_stars):
        x = pos[s, 0]; y = pos[s, 1]; z = pos[s, 2]
        r2 = x * x + y * y + z * z
        r  = r2 ** 0.5
        if r < 1e-10:
            r = 1e-10

        # Host: log-linear interpolation of M_enc(r)
        log_r = np.log(r)
        t = (log_r - log_r_min) * inv_dlr
        if t < 0.0:
            t = 0.0
        elif t > N_grid - 1 - 1e-12:
            t = N_grid - 1 - 1e-12
        i0   = int(t)
        frac = t - i0
        M_r  = (1.0 - frac) * M_enc_grid[i0] + frac * M_enc_grid[i0 + 1]

        fac_h = -_G_NB * M_r / (r2 * r)
        ax = fac_h * x
        ay = fac_h * y
        az = fac_h * z

        # Subhalo point-mass contributions
        for j in range(N_macho):
            if sub_Mh[j] == 0.0:
                continue
            dx  = x - macho_cart[j, 0]
            dy  = y - macho_cart[j, 1]
            dz  = z - macho_cart[j, 2]
            r2m = dx * dx + dy * dy + dz * dz + softening_sq
            r3m = r2m * (r2m ** 0.5)
            fac_m = -_G_NB * sub_Mh[j] / r3m
            ax += fac_m * dx
            ay += fac_m * dy
            az += fac_m * dz

        accel[s, 0] = ax
        accel[s, 1] = ay
        accel[s, 2] = az

    return accel


@njit(cache=True)
def _yoshida4_step_sub_pointmass_nb(pos_c, vel_c, macho_cart, sub_Mh,
                                     log_r_grid, M_enc_grid, softening_sq,
                                     sub_dt, YC, YD, n_sub):
    """n_sub Yoshida-4 sub-steps using point-mass subhalos."""
    for _ in range(n_sub):
        pos_c = pos_c + YC[0] * vel_c * sub_dt
        a     = _accel_cart_sub_pointmass_nb(pos_c, macho_cart, sub_Mh,
                                              log_r_grid, M_enc_grid, softening_sq)
        vel_c = vel_c + YD[0] * a * sub_dt

        pos_c = pos_c + YC[1] * vel_c * sub_dt
        a     = _accel_cart_sub_pointmass_nb(pos_c, macho_cart, sub_Mh,
                                              log_r_grid, M_enc_grid, softening_sq)
        vel_c = vel_c + YD[1] * a * sub_dt

        pos_c = pos_c + YC[2] * vel_c * sub_dt
        a     = _accel_cart_sub_pointmass_nb(pos_c, macho_cart, sub_Mh,
                                              log_r_grid, M_enc_grid, softening_sq)
        vel_c = vel_c + YD[2] * a * sub_dt

        pos_c = pos_c + YC[3] * vel_c * sub_dt
    return pos_c, vel_c


def stellar_simulation_clumps_pointmass_nb(xvMACHO, posStar, velStar,
                                            subhalo_M,
                                            host, timesteps, r0i,
                                            host_Mh=None, host_rh=None,
                                            host_c=None, host_alpha=None,
                                            softening=0.0,
                                            n_substeps=None, Nsteps=None,
                                            earlyStop=True):
    """Yoshida-4 stellar N-body with subhalos treated as point masses.

    Identical to stellar_simulation_clumps_nb but subhalos are pure point
    masses (no Dekel profile needed).  The host potential still evolves in
    time when host_* arrays are provided.

    Parameters
    ----------
    xvMACHO   : (N_sub, N_step+1, 6)   cylindrical [R,phi,z,VR,Vphi,Vz]
    posStar   : (N_star, 3)             initial star positions [kpc]
    velStar   : (N_star, 3)             initial star velocities [kpc/Gyr]
    subhalo_M : (N_sub, N_step+1)       total subhalo mass [Msun] at each step
    host      : Dekel/NFW profile       background host potential (fallback
                                        when host_* arrays are None)
    timesteps : (N_step,)               output times [Gyr] (excluding t=0)
    r0i       : float                   initial Plummer scale radius [kpc]
    host_Mh   : (N_step+1,) or None    host virial mass [Msun] (incl. t=0)
    host_rh   : (N_step+1,) or None    host virial radius [kpc]
    host_c    : (N_step+1,) or None    host Dekel concentration
    host_alpha: (N_step+1,) or None    host Dekel inner slope
    softening : float                   Plummer softening length [kpc]
    n_substeps: int or None             Yoshida sub-steps per outer step
    Nsteps    : int or None             upsample trajectory to this many steps
    earlyStop : bool                    stop early if stellar system dissolves

    Returns
    -------
    R0starList, HLRadiusList, dens_final, xvList, timesteps
    """
    bins   = np.logspace(-4, 0, 40)
    Nstars = len(posStar)
    N_sub  = xvMACHO.shape[0]

    subhalo_M = np.asarray(subhalo_M, dtype=np.float64)

    evolving_host = (host_Mh is not None)
    if evolving_host:
        host_Mh    = np.asarray(host_Mh,    dtype=np.float64)
        host_rh    = np.asarray(host_rh,    dtype=np.float64)
        host_c     = np.asarray(host_c,     dtype=np.float64)
        host_alpha = np.asarray(host_alpha, dtype=np.float64)

    if Nsteps is not None and Nsteps > len(timesteps):
        t_knots_orig = np.concatenate([[0.], timesteps])
        xvMACHO, timesteps = upsample(xvMACHO, timesteps, Nsteps)
        t_fine    = np.concatenate([[0.], timesteps])
        subhalo_M = np.vstack([np.interp(t_fine, t_knots_orig, subhalo_M[j])
                                for j in range(N_sub)])
        if evolving_host:
            host_Mh    = np.interp(t_fine, t_knots_orig, host_Mh)
            host_rh    = np.interp(t_fine, t_knots_orig, host_rh)
            host_c     = np.interp(t_fine, t_knots_orig, host_c)
            host_alpha = np.interp(t_fine, t_knots_orig, host_alpha)

    Nsteps_actual = len(timesteps)

    R0starList   = np.zeros(Nsteps_actual + 1)
    HLRadiusList = np.zeros(Nsteps_actual + 1)
    R0starList[0]   = r0i
    HLRadiusList[0] = np.percentile(np.linalg.norm(posStar, axis=1), 50)

    R_m, phi_m, z_m = xvMACHO[:, :, 0], xvMACHO[:, :, 1], xvMACHO[:, :, 2]
    macho_cart_all  = np.ascontiguousarray(
        np.stack([R_m * np.cos(phi_m), R_m * np.sin(phi_m), z_m], axis=-1),
        dtype=np.float64,
    )  # (N_sub, N_step+1, 3)

    r_nb     = np.logspace(-5, 2, 1000, dtype=np.float64)
    log_r_nb = np.ascontiguousarray(np.log(r_nb), dtype=np.float64)
    if evolving_host:
        M_enc_nb = np.ascontiguousarray(
            _dekel_M_enc_grid(r_nb, host_Mh[0], host_rh[0] / host_c[0],
                              host_c[0], host_alpha[0]), dtype=np.float64)
    else:
        M_enc_nb = np.ascontiguousarray(host.M(r_nb), dtype=np.float64)
    soft_sq = np.float64(softening ** 2)

    dt_outer = float(timesteps[0])
    if n_substeps is None:
        if evolving_host:
            host_rs_all = host_rh / host_c
            omega_all   = np.sqrt(_G_NB * host_Mh / host_rs_all**3)
            dt_stable   = 1.0 / (omega_all.max() * 100.0)
            n_substeps  = max(1, int(np.ceil(dt_outer / dt_stable)))
        else:
            n_substeps = _compute_n_substeps(host, dt_outer)
    sub_dt = dt_outer / n_substeps
    if evolving_host:
        omega_max = np.sqrt(_G_NB * host_Mh[0] / (host_rh[0] / host_c[0])**3)
    else:
        omega_max = np.sqrt(_G_NB * host.Mh / host.rs**3)
    print(f'  omega_max = {omega_max:.1f} Gyr⁻¹   '
          f'dt_outer = {1e3*dt_outer:.1f} Myr   '
          f'n_substeps = {n_substeps}   '
          f'sub_dt = {1e3*sub_dt:.2f} Myr   '
          f'total force evals/step = {3*n_substeps}')

    pos_c = np.ascontiguousarray(posStar, dtype=np.float64)
    vel_c = np.ascontiguousarray(velStar, dtype=np.float64)
    YC_nb = np.ascontiguousarray(_YC, dtype=np.float64)
    YD_nb = np.ascontiguousarray(_YD, dtype=np.float64)

    pos_hist = np.empty((Nstars, Nsteps_actual + 1, 3))
    vel_hist = np.empty((Nstars, Nsteps_actual + 1, 3))
    pos_hist[:, 0, :] = pos_c
    vel_hist[:, 0, :] = vel_c

    print('>>> compiling numba kernels (first call may take a few seconds) ...')
    _dummy_pos, _dummy_vel = _yoshida4_step_sub_pointmass_nb(
        pos_c.copy(), vel_c.copy(),
        np.ascontiguousarray(macho_cart_all[:, 0, :], dtype=np.float64),
        np.ascontiguousarray(subhalo_M[:, 0], dtype=np.float64),
        log_r_nb, M_enc_nb, soft_sq,
        np.float64(sub_dt), YC_nb, YD_nb, 1,
    )
    print('>>> evolving (Yoshida-4, numba-parallel, point-mass subhalos) ...')
    tprevious = 0.0
    t1 = time.time()

    for i, t in tqdm(enumerate(timesteps)):
        dt_i     = t - tprevious
        sub_dt_i = np.float64(dt_i / n_substeps)
        macho_c  = np.ascontiguousarray(macho_cart_all[:, i, :], dtype=np.float64)

        if evolving_host:
            M_enc_nb = np.ascontiguousarray(
                _dekel_M_enc_grid(r_nb, host_Mh[i], host_rh[i] / host_c[i],
                                  host_c[i], host_alpha[i]), dtype=np.float64)

        pos_c, vel_c = _yoshida4_step_sub_pointmass_nb(
            pos_c, vel_c, macho_c,
            np.ascontiguousarray(subhalo_M[:, i], dtype=np.float64),
            log_r_nb, M_enc_nb, soft_sq,
            sub_dt_i, YC_nb, YD_nb, n_substeps,
        )

        tprevious = t
        pos_hist[:, i + 1, :] = pos_c
        vel_hist[:, i + 1, :] = vel_c

        rcurrent          = np.linalg.norm(pos_c, axis=1)
        r_half            = np.percentile(rcurrent, 50)
        HLRadiusList[i+1] = r_half
        R0starList[i+1]   = r_half * _PLUMMER_R0_FROM_RHALF

        if R0starList[i+1] > 50 * r0i and HLRadiusList[i+1] > 50 * r0i and earlyStop:
            print(f"Warning: at t={t:.2f} Gyr, R0*={R0starList[i+1]:.4f} kpc "
                  f"and r_half={HLRadiusList[i+1]:.4f} kpc are both >50x r0i. Stopping.")
            R0starList[i+2:]     = R0starList[i+1]
            HLRadiusList[i+2:]   = HLRadiusList[i+1]
            pos_hist[:, i+2:, :] = pos_c[:, None, :]
            vel_hist[:, i+2:, :] = vel_c[:, None, :]
            break

    t2 = time.time()
    print(f'    time = {t2 - t1:.4f} s')
    print(f"r0i    = {r0i:.4f} kpc")
    print(f"R0star = {R0starList[-1]:.4f} kpc  (x{R0starList[-1] / r0i:.3f})")

    x, y, z    = pos_hist[:, :, 0], pos_hist[:, :, 1], pos_hist[:, :, 2]
    vx, vy, vz = vel_hist[:, :, 0], vel_hist[:, :, 1], vel_hist[:, :, 2]
    R_h   = np.sqrt(x**2 + y**2)
    phi_h = np.arctan2(y, x)
    VR_h  = (x * vx + y * vy) / np.maximum(R_h, 1e-10)
    Vp_h  = (x * vy - y * vx) / np.maximum(R_h, 1e-10)
    xvList = np.stack([R_h, phi_h, z, VR_h, Vp_h, vz], axis=-1)

    dens_final = estimateProfile(pos_c, np.ones(Nstars), bins)

    return R0starList, HLRadiusList, dens_final, xvList, timesteps