# SatGen imports
# other imports
import numpy as np
from scipy.differentiate import derivative
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import brentq, minimize

import config as cfg


def tidalTensor(hostProfile, coords):
    x,y,z = coords
    r = np.sqrt(x**2+y**2+z**2)
    Mh = hostProfile.M(r)
    rho = hostProfile.rho(r)
    rarb = np.array([[x**2,x*y,x*z],[x*y,y**2,y*z],[x*z,z*y,z**2]])
    return cfg.G*Mh/r**3 * (3*rarb/r**2 - np.identity(3))- 4*np.pi*cfg.G*rho*rarb/r**2

class NumericProfile(object):
    def __init__(self, ri, Mr):
        self.ri = ri
        self.Mr = Mr
        self.rh = np.max(ri)
        self.Mh = np.max(Mr)

        # compute density profile from masses
        self.MInt = InterpolatedUnivariateSpline(self.ri, self.Mr, ext='const')
        self.rhovals = derivative(self.MInt, self.ri).df/(4.0*np.pi*self.ri**2)
        self.rhoInt = InterpolatedUnivariateSpline(self.ri, self.rhovals, ext='zeros')

        self.rmax = minimize(lambda x: -self.Vcirc(x), self.rh/2., method='L-BFGS-B', bounds=((self.ri[0],self.rh),)).x[0]
        self.Vmax = self.Vcirc(self.rmax)

    def rho(self, R, z=0.):
        r = np.sqrt(R**2 + z**2)

        return self.rhoInt(r)

    def M(self, R, z=0.):
        r = np.sqrt(R**2 + z**2)

        return self.MInt(r)

    def rhobar(self, R, z=0.):
        r = np.sqrt(R**2.+z**2.)
        return self.M(r)/(cfg.FourPiOverThree*r**3)

    def tdyn(self,R,z=0.):
        return np.sqrt(cfg.ThreePiOverSixteenG / self.rhobar(R,z))

    def Phi(self,R,z=0.):
        r = np.sqrt(R**2.+z**2.)
        #phi1 = -cfg.G*self.M(r)/r
        phi1 = -cfg.G*self.Mh*quad(lambda x: 1/x**2, self.rh, np.inf)[0]

        if isinstance(r,list) or isinstance(r,np.ndarray):
            phiList =[]
            for i in range(len(r)):
                #phiList.append(-cfg.G*4*np.pi*quad(lambda x: x*self.rho(x), r[i], self.rh)[0])
                phiList.append(-cfg.G*quad(lambda x: self.M(x)/x**2, r[i], self.rh)[0])
            phi2 = np.array(phiList)
        else:
            phi2 = -cfg.G*quad(lambda x: self.M(x)/x**2, r, self.rh)[0]

        return phi1 + phi2

    def fgrav(self,R):
        r = np.sqrt(R**2.+z**2.)
        fac = -cfg.G*self.M(r)/r**2
        return fac*R/r, 0, fac*z/r

    def Vcirc(self,R,z=0.):
        r = np.sqrt(R**2 + z**2)
        return np.sqrt(cfg.G*self.M(r)/r)

    def sigma(self,R,z=0.):
        r = np.sqrt(R**2.+z**2.)
        if isinstance(r,list) or isinstance(r,np.ndarray):
            intList =[]
            for i in range(len(r)):
                intList.append(cfg.G*quad(lambda x: self.rho(x)*self.M(x)/x**2,r[i],self.rh)[0])
            integ = np.array(intList)
        else:
            integ = cfg.G*quad(lambda x: self.rho(x)*self.M(x)/x**2,r,self.rh)[0]
        return np.sqrt(integ/self.rho(r))

def heat_profile(profile: NumericProfile, eps, count_per_decade=100):
    """
    Apply monotonic heating algorithm to a NumericProfile.

    Parameters
    ----------
    profile : NumericProfile
        Initial halo profile
    eps : function
        Specific energy injection function eps(r)
    count_per_decade : int
        Radial resolution

    Returns
    -------
    NumericProfile
        Heated profile
    """

    G = cfg.G

    rvir = profile.rh

    # --------------------------------------------------
    # Radial grid
    # --------------------------------------------------
    rmin = min(profile.ri)
    rmax = 10.0 * rvir

    decades = np.log10(rmax / rmin)
    count_r = int(decades * count_per_decade + 1)

    ri = np.logspace(np.log10(rmin), np.log10(rmax), count_r)

    # --------------------------------------------------
    # Arrays
    # --------------------------------------------------
    Menc = profile.M(ri)
    perturb = np.zeros_like(ri)

    # --------------------------------------------------
    # Compute perturbation outer → inner
    # --------------------------------------------------
    for i in reversed(range(count_r)):

        r = ri[i]
        m = Menc[i]

        if m > 0:
            perturb[i] = 2.0 * eps(r) * r / (G * m)
        else:
            perturb[i] = 0.0

        # enforce monotonic shell ordering
        if i < count_r - 1:

            limit = (
                1.0
                - r / ri[i + 1]
                * (Menc[i] / Menc[i + 1]) ** (-1.0 / 3.0)
                * (1.0 - perturb[i + 1])
            )

            perturb[i] = min(perturb[i], limit)

    # --------------------------------------------------
    # Final energies
    # --------------------------------------------------
    Ef = G * Menc / ri * (-1.0 + perturb)

    # --------------------------------------------------
    # Shell masses
    # --------------------------------------------------
    Mshell = np.zeros_like(Menc)
    Mshell[0] = Menc[0]
    Mshell[1:] = np.diff(Menc)

    # --------------------------------------------------
    # Bound shells
    # --------------------------------------------------
    bound = (Ef < 0) & (Mshell > 0)

    rf = np.full_like(ri, np.inf)
    rf[bound] = -G * Menc[bound] / Ef[bound]

    # --------------------------------------------------
    # If halo destroyed
    # --------------------------------------------------
    if np.count_nonzero(bound) <= 2:
        raise RuntimeError("Heating unbinds the halo")

    r_bound = rf[bound]
    M_bound = Menc[bound]

    # --------------------------------------------------
    # Sort by radius (important after shell expansion)
    # TODO(Fabio): Investigate further. This is suspicious, it indicates there can be shell crossing.
    # --------------------------------------------------
    order = np.argsort(r_bound)

    r_bound = r_bound[order]
    M_bound = M_bound[order]

    # --------------------------------------------------
    # Remove duplicates if needed
    # --------------------------------------------------
    r_unique, idx = np.unique(r_bound, return_index=True)

    r_bound = r_unique
    M_bound = M_bound[idx]

    # --------------------------------------------------
    # Return new NumericProfile
    # --------------------------------------------------
    return NumericProfile(r_bound, M_bound)
