import warnings
import numpy as np
from scipy import interpolate
from scipy.special import kn, kve

##############################################
# electron-ion heating functions

def qie_func(Te,Ti,ne):

    # compute dimensionless temperatures
    theta_e = (1.68637e-10)*Te
    theta_i = (9.18426e-14)*Ti

    # arguments of the Bessel functions; note that (theta_e+theta_i)/(theta_e*theta_i) = a + b
    a = 1.0/theta_e
    b = 1.0/theta_i

    # evaluate using the exponentially scaled Bessel functions kve(n,x) = kn(n,x)*exp(x);
    # the exp(-(a+b)) factors cancel between numerator and denominator
    qie = (5.624e-32)*((ne**2.0)*(Ti - Te))*(((((2.0*((theta_e + theta_i)**2.0)) + 1.0) / (theta_e + theta_i))*kve(1,a+b)) + (2.0*kve(0,a+b)))/(kve(2,a)*kve(2,b))

    return qie

##############################################
# electron advection function

def gammaCV(theta_e):
    return (20.0*(2.0 + (8.0*theta_e) + (5.0*(theta_e**2.0)))) / (3.0*(8.0 + (40.0*theta_e) + (25.0*(theta_e**2.0))))

##############################################
# relativistic Maxwellian function

def relMax(xM):
    return (4.0505/(xM**(1.0/6.0)))*(1.0 + (0.40/(xM**(1.0/4.0))) + (0.5316/(xM**(1.0/2.0))))*np.exp(-1.8899*(xM**(1.0/3.0)))

##############################################
# electron temperature profile

def Te_profile(r,Te0,t,req,K):
    """
    Piecewise electron temperature profile.  Inside req the flow is two-temperature and
    Te = Te0 * r**(-(1-t)); at and beyond req the electrons are locked to the ions,
    Te = Ti = K/r, where K = 6.66e12 K * beta * c3 / (2.08 (1+beta)) is the common
    temperature normalization (NY95 eq. 2.16 with Ti = Te).  Continuous at req by
    construction of t.
    """
    r = np.asarray(r, dtype=float)
    return np.where(r <= req, Te0/(r**(1.0-t)), K/r)

def Ti_profile(r,Te,beta,c3):
    """Ion temperature from NY95 eq. (2.16): Ti + 1.08 Te = 6.66e12 K * beta c3/((1+beta) r)."""
    r = np.asarray(r, dtype=float)
    return ((6.66e12)*((1.0+beta)**(-1.0))*beta*c3*(r**(-1.0))) - (1.08*Te)

##############################################
# synchrotron functions

def find_critical_freq(nu_arr,volume,surface,n_bisect=10):

    diff = volume - surface
    
    # first check if the intersection is contained in the sampled points
    zero_idx = np.where(diff == 0)[0]
    if (zero_idx.size > 0):
        i = zero_idx[0]
        nu_crit = nu_arr[i]
        Lnu_crit = volume[i]
        return nu_crit, Lnu_crit

    # otherwise look for a sign change, which should bracket an intersection
    sign_change_idx = np.where(diff[:-1]*diff[1:] < 0)[0]

    # if the sign never changes, then fall back to the point of minimum absolute difference
    if (sign_change_idx.size == 0):
        warnings.warn('no synchrotron self-absorption crossing was bracketed within the '
                      'sampled frequency range; falling back to the point of closest '
                      'approach, which may not be a real critical frequency',
                      RuntimeWarning, stacklevel=2)
        i = np.argmin(np.abs(diff))
        nu_crit = nu_arr[i]
        Lnu_crit = 0.5*(volume[i] + surface[i])
        return nu_crit, Lnu_crit

    # otherwise carry out a bisection search within the identified interval
    i = sign_change_idx[0]
    x_lo = nu_arr[i]
    x_hi = nu_arr[i+1]
    y_lo = diff[i]
    y_hi = diff[i+1]
    for _ in range(n_bisect):
        x_mid = 0.5*(x_lo + x_hi)
        y_mid = np.interp(x_mid,nu_arr,diff)

        if (y_mid == 0.0):
            x_lo = x_hi = x_mid
            break

        if ((y_lo*y_mid) <= 0):
            x_hi = x_mid
            y_hi = y_mid
        else:
            x_lo = x_mid
            y_lo = y_mid

    nu_crit = 0.5*(x_lo + x_hi)
    vol_crit = np.interp(nu_crit, nu_arr, volume)
    surf_crit = np.interp(nu_crit, nu_arr, surface)
    Lnu_crit = 0.5*(vol_crit + surf_crit)

    return nu_crit, Lnu_crit

def synch_branches(nu_arr,Te_eval,r_eval,m,mdot,s,alpha,beta,c1,c3):
    """
    Optically-thin ("volume") and optically-thick ("surface") synchrotron luminosity
    densities at a single radius r_eval where the electron temperature is Te_eval.  Their
    intersection defines the critical frequency nu_c(r_eval) below which the synchrotron
    emission is self-absorbed.  Both branches use the same uniform-sphere convention
    (NY95 eq. 3.13 / M97 eq. 19): the volume branch is the emissivity times the volume of
    a sphere of radius R_eval, and the surface branch is Rayleigh-Jeans emission from the
    surface of that same sphere.
    """

    # dimensionless temperature
    theta_e = (1.68637e-10)*Te_eval

    # gyro frequency
    nu_b = (3.998e15)*((1+beta)**(-1.0/2.0))*(alpha**(-1.0/2.0))*(c1**(-1.0/2.0))*(c3**(1.0/2.0))*(m**(-1.0/2.0))*(mdot**(1.0/2.0))*(r_eval**((-5.0/4.0) + (s/2.0)))

    # dimensionless frequency
    xM = 2.0*nu_arr/(3.0*nu_b*(theta_e**2.0))

    # optically-thin and optically-thick branches
    volume = (1.896e8)*(relMax(xM)/kn(2,1.0/theta_e))*(alpha**(-1.0))*(c1**(-1.0))*(m**2.0)*mdot*nu_arr*(r_eval**((3.0/2.0) + s))
    surface = (1.058e-24)*(nu_arr**2.0)*Te_eval*(m**2.0)*(r_eval**2.0)

    return volume, surface

def compute_crit_freq(nu_arr,Te_eval,r_eval,m,mdot,s,alpha,beta,c1,c3):
    """Critical (self-absorption) frequency and luminosity density at radius r_eval."""

    volume, surface = synch_branches(nu_arr,Te_eval,r_eval,m,mdot,s,alpha,beta,c1,c3)
    nu_c, L_c = find_critical_freq(nu_arr,volume,surface)

    return nu_c, L_c

def compute_peak_freq(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3,T_synch_min=1.0e8):

    # if even the hottest gas in the zone is cooler than T_synch_min, the thermal
    # synchrotron formulae do not apply anywhere in it
    if (Te0/(rmin**(1.0-t))) < T_synch_min:
        return 0.0, 0.0

    # the peak frequency is the critical frequency at the innermost radius
    return compute_crit_freq(nu_arr,Te0/(rmin**(1.0-t)),rmin,m,mdot,s,alpha,beta,c1,c3)

def compute_min_freq(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3,T_synch_min=1.0e8):

    # if even the hottest gas in the zone is cooler than T_synch_min, the thermal
    # synchrotron formulae do not apply anywhere in it
    if (Te0/(rmin**(1.0-t))) < T_synch_min:
        return 0.0, 0.0

    # The outer synchrotron anchor is placed no further out than where the temperature
    # profile falls to T_synch_min; this keeps the anchor inside the validity range of
    # the thermal synchrotron formulae during the Te0 search, where trial profiles can
    # be very cold.  When this branch is reached, t < 1 is guaranteed; t = 1 makes the
    # profile flat, and the zone-top check above has already passed.
    if (Te0/(rmax**(1.0-t))) < T_synch_min:
        r_anchor = (Te0/T_synch_min)**(1.0/(1.0-t))
    else:
        r_anchor = rmax

    # the minimum frequency is the critical frequency at the outer edge of the
    # two-temperature zone (rmax here, capped to r_anchor), evaluated on the inner
    # temperature profile
    return compute_crit_freq(nu_arr,Te0/(r_anchor**(1.0-t)),r_anchor,m,mdot,s,alpha,beta,c1,c3)

def compute_synch_power(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3,T_synch_min=1.0e8):

    # critical frequencies at the inner and outer edges
    nu_p, L_p = compute_peak_freq(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3,T_synch_min)

    # no synchrotron if the whole zone is below the validity floor
    if L_p <= 0.0:
        return 0.0

    nu_min, Lnu_min = compute_min_freq(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3,T_synch_min)

    # integrate the spectrum
    Lnu_synch = compute_synch_spectrum(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3,nu_p,L_p,nu_min,Lnu_min)
    P_synch = np.sum(0.5*(Lnu_synch[1:] + Lnu_synch[0:-1])*(nu_arr[1:] - nu_arr[0:-1]))

    return P_synch

def compute_synch_spectrum(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3,nu_p,L_p,nu_min,Lnu_min,nu_eq=None,Lnu_eq=None):
    """
    Piecewise synchrotron spectrum (P21 eq. A18): Rayleigh-Jeans below nu_min, a power law
    connecting the critical points, and the optically-thin emissivity at rmin above nu_p.

    If (nu_eq, Lnu_eq) is given, the flow has an outer one-temperature zone: nu_min is then
    the critical frequency at the outer edge of the flow and nu_eq the critical frequency
    at the outer edge of the two-temperature zone, and the power law is broken at nu_eq
    (M97 sec. 4.1: the one-temperature zone with Te ~ 1/r gives a steeper radio slope).
    """

    # construct synchrotron spetrum
    Lnu_synch = np.zeros_like(nu_arr)

    # synchrotron emission below nu_min is blackbody
    index = (nu_arr <= nu_min)
    Lnu_synch[index] = Lnu_min*((nu_arr[index]/nu_min)**2.0)

    # synchrotron emission above nu_p is Maxwellian
    index = (nu_arr >= nu_p)

    volume, surface = synch_branches(nu_arr,Te0/(rmin**(1.0-t)),rmin,m,mdot,s,alpha,beta,c1,c3)
    Lnu_synch[index] = volume[index]

    if nu_eq is None:

        # single power law between nu_min and nu_p, connecting the two critical points
        # exactly (P21 eq. A18)
        pl_exp = np.log(L_p/Lnu_min) / np.log(nu_p/nu_min)
        index = ((nu_arr > nu_min) & (nu_arr < nu_p))
        Lnu_synch[index] = Lnu_min*((nu_arr[index]/nu_min)**pl_exp)

    else:

        # outer (one-temperature) zone: nu_min -> nu_eq
        pl_exp_out = np.log(Lnu_eq/Lnu_min) / np.log(nu_eq/nu_min)
        index = ((nu_arr > nu_min) & (nu_arr < nu_eq))
        Lnu_synch[index] = Lnu_min*((nu_arr[index]/nu_min)**pl_exp_out)

        # inner (two-temperature) zone: nu_eq -> nu_p
        pl_exp_in = np.log(L_p/Lnu_eq) / np.log(nu_p/nu_eq)
        index = ((nu_arr >= nu_eq) & (nu_arr < nu_p))
        Lnu_synch[index] = Lnu_eq*((nu_arr[index]/nu_eq)**pl_exp_in)

    return Lnu_synch

##############################################
# inverse Compton functions

def compute_compt_spectrum(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3,nu_p,L_p):

    # optical depth to electron scattering
    # NOTE: the coefficient here follows M97 Eq. 31; P21 Eq. A25 is wrong by a factor of 2
    tau_es = 6.205*(alpha**(-1.0))*(c1**(-1.0))*mdot*rmin**(-(1.0/2.0) + s)

    # electron temperature at rmin
    Te_rmin = Te0 / (rmin**(1.0-t))

    # dimensionless temperature
    theta_e_rmin = (1.68637e-10)*Te_rmin

    # amplification factor
    A = 1.0 + (4.0*theta_e_rmin) + (16.0*(theta_e_rmin**2.0))

    # power-law index
    alpha_c = -np.log(tau_es) / np.log(A)

    # peak frequency
    nu_f = (6.251e10)*Te_rmin

    # construct Compton spectrum, adding exponential cutoffs at nu_p and nu_f
    with np.errstate(over='ignore'):
        log_Lnu_compt = (np.log(L_p) - (alpha_c*np.log(np.maximum(nu_arr,nu_p)/nu_p))
                         - ((nu_arr/(0.5*nu_f))**2.0)      # exponential cutoff at nu_f
                         - ((nu_arr/(1.0*nu_p))**-4.0))    # exponential cutoff at nu_p
        Lnu_compt = np.exp(log_Lnu_compt)

    return Lnu_compt

def compute_compt_power(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3,T_synch_min=1.0e8):

    # optical depth to electron scattering
    tau_es = 6.205*(alpha**(-1.0))*(c1**(-1.0))*mdot*rmin**(-(1.0/2.0) + s)

    # electron temperature at rmin
    Te_rmin = Te0 / (rmin**(1.0-t))

    # dimensionless temperature
    theta_e_rmin = (1.68637e-10)*Te_rmin

    # amplification factor
    A = 1.0 + (4.0*theta_e_rmin) + (16.0*(theta_e_rmin**2.0))

    # power-law index
    alpha_c = -np.log(tau_es) / np.log(A)

    # peak frequency
    nu_f = (6.251e10)*Te_rmin

    # get peak synchrotron frequency
    nu_p, L_p = compute_peak_freq(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3,T_synch_min)

    # no Comptonization without synchrotron seed photons
    if L_p <= 0.0:
        return 0.0

    # compute total power
    Lnu_compt = compute_compt_spectrum(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3,nu_p,L_p)
    P_compt = np.sum(0.5*(Lnu_compt[1:] + Lnu_compt[0:-1])*(nu_arr[1:] - nu_arr[0:-1]))

    return P_compt

##############################################
# bremsstrahlung F(theta) function

def bremsF(theta_e):

    # If scalar, make into 1D array
    theta_e = np.asarray(theta_e)
    scalar_input = False
    if theta_e.ndim == 0:
        theta_e = theta_e[np.newaxis]  
        scalar_input = True

    # initialize array
    F = np.zeros_like(theta_e)

    # first branch
    ind1 = (theta_e <= 1.0)
    F[ind1] = (4.0*np.sqrt(2.0*theta_e[ind1] / (np.pi**3.0))*(1.0 + (1.781*(theta_e[ind1]**1.34)))) + (1.73*(theta_e[ind1]**(3.0/2.0))*(1.0 + (1.1*theta_e[ind1]) + (theta_e[ind1]**2.0) - (1.25*(theta_e[ind1]**(5.0/2.0)))))

    # second branch
    ind2 = (theta_e > 1.0)
    F[ind2] = ((9.0*theta_e[ind2]/(2.0*np.pi))*(np.log((1.123*theta_e[ind2]) + 0.48) + 1.5)) + (2.30*theta_e[ind2]*(np.log(1.123*theta_e[ind2]) + 1.28))

    if scalar_input:
        return np.squeeze(F)
    return F

def compute_brems_power(r,Te,m,mdot,s,alpha,c1):
    """Volume-integrated bremsstrahlung power (P21 eq. A22) on the radial grid r with
    electron temperature profile Te (same shape as r)."""

    # dimensionless temperature
    theta_e = (1.68637e-10)*Te

    # integrate over volume
    integrand = (4.776e34)*(alpha**(-2.0))*(c1**(-2.0))*m*(mdot**2.0)*bremsF(theta_e)*(r**(-1.0 + (2.0*s)))
    P_brems = np.sum(0.5*(integrand[1:] + integrand[0:-1])*(r[1:] - r[0:-1]))

    return P_brems

def compute_brems_spectrum(nu_arr,r,Te,m,mdot,s,alpha,c1):
    """Bremsstrahlung spectrum (P21 eq. A23, with the r-exponent written in terms of the
    local Te so that any temperature profile can be used) on the radial grid r."""

    # dimensionless temperature
    theta_e = (1.68637e-10)*Te

    # radial part is frequency-independent, so pull it out of the frequency loop; the
    # 1/Te factor normalizes the flat-plus-exponential spectral shape to q_brems
    pref = (2.292e24)*(alpha**(-2.0))*(c1**(-2.0))*m*(mdot**2.0)*bremsF(theta_e)*(Te**(-1.0))*(r**(-1.0 + (2.0*s)))
    integrand = pref[None,:]*np.exp(-(4.799e-11)*(nu_arr[:,None]/Te[None,:]))
    Lnu_brems = np.sum(0.5*(integrand[:,1:] + integrand[:,0:-1])*(r[1:] - r[0:-1])[None,:], axis=1)

    return Lnu_brems

##############################################
# full-flow synchrotron spectrum

def assemble_synch_spectrum(nu_arr,Te0,t,rmin,rmax,req,m,mdot,s,alpha,beta,c1,c3,T_synch_min=1.0e8):
    """
    Synchrotron spectrum of the whole flow.  The two-temperature zone [rmin, r_bal] with
    r_bal = min(req, rmax) supplies the peak (rmin) and the inner critical point (r_bal).
    If the flow extends beyond req, the one-temperature zone [req, rmax] with Te = K/r
    adds a further, lower critical point at its outer edge, and the radio spectrum is a
    broken power law.  Following M97 (sec. 4.1) the outer anchor is placed no further out
    than where Te falls to T_synch_min, below which the thermal synchrotron formulae are
    not meaningful.

    Returns (Lnu_synch, nu_p, L_p, nu_bal, L_bal, nu_min, L_min); the last two equal the
    middle two when there is no outer zone.  If req <= rmin the entire flow is
    one-temperature and the spectrum is a single zone anchored at rmin and at r_syn =
    min(rmax, Te0/T_synch_min); if even Te(rmin) = Te0/rmin falls below T_synch_min, the
    synchrotron emission is omitted entirely (zeros, with nu_p = L_p = 0).
    """

    K = (6.66e12)*beta*c3/(2.08*(1.0+beta))
    r_bal = min(req,rmax)

    # if the equalization radius lies at or inside the inner edge, the entire flow is
    # one-temperature (Te = Te0/r): the spectrum is a single zone running from the inner
    # edge to the synchrotron validity radius r_syn
    if req <= rmin:
        r_syn = min(rmax, K/T_synch_min)
        if r_syn <= rmin:
            warnings.warn('the entire flow is cooler than T_synch_min = '+str(T_synch_min)
                          +' K; thermal synchrotron emission (and its Comptonization) is omitted',
                          RuntimeWarning, stacklevel=2)
            zeros = np.zeros_like(nu_arr)
            return zeros, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        nu_p, L_p = compute_crit_freq(nu_arr,K/rmin,rmin,m,mdot,s,alpha,beta,c1,c3)
        nu_min, L_min = compute_crit_freq(nu_arr,K/r_syn,r_syn,m,mdot,s,alpha,beta,c1,c3)
        Lnu = compute_synch_spectrum(nu_arr,K,0.0,rmin,r_syn,m,mdot,s,alpha,beta,c1,c3,nu_p,L_p,nu_min,L_min)
        return Lnu, nu_p, L_p, nu_p, L_p, nu_min, L_min

    nu_p, L_p = compute_peak_freq(nu_arr,Te0,t,rmin,r_bal,m,mdot,s,alpha,beta,c1,c3,T_synch_min)
    nu_bal, L_bal = compute_min_freq(nu_arr,Te0,t,rmin,r_bal,m,mdot,s,alpha,beta,c1,c3,T_synch_min)

    # if even the innermost gas is below T_synch_min then there is no thermal synchrotron
    # (and no Compton seed photons) anywhere in the flow
    if L_p <= 0.0:
        warnings.warn('the entire two-temperature zone is cooler than T_synch_min = '+str(T_synch_min)
                      +' K; thermal synchrotron emission (and its Comptonization) is omitted',
                      RuntimeWarning, stacklevel=2)
        zeros = np.zeros_like(nu_arr)
        return zeros, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    if req < rmax:
        # outer one-temperature zone: anchor at its outer edge (capped where Te = T_synch_min)
        r_syn = min(rmax, K/T_synch_min)
        if r_syn > req:
            nu_min, L_min = compute_crit_freq(nu_arr,K/r_syn,r_syn,m,mdot,s,alpha,beta,c1,c3)
            Lnu = compute_synch_spectrum(nu_arr,Te0,t,rmin,r_bal,m,mdot,s,alpha,beta,c1,c3,nu_p,L_p,nu_min,L_min,nu_eq=nu_bal,Lnu_eq=L_bal)
            return Lnu, nu_p, L_p, nu_bal, L_bal, nu_min, L_min

    Lnu = compute_synch_spectrum(nu_arr,Te0,t,rmin,r_bal,m,mdot,s,alpha,beta,c1,c3,nu_p,L_p,nu_bal,L_bal)
    return Lnu, nu_p, L_p, nu_bal, L_bal, nu_bal, L_bal

##############################################
# integrated power budget

def compute_powers(nu_arr,r,Te,Te0,t,f,rmin,rmax,req,m,mdot,s,alpha,beta,c1,c3,T_synch_min=1.0e8):
    """
    Total viscous dissipation Q+ (P21 eq. A10) over the whole flow [rmin, rmax], and total
    radiated power Q- (P21 eq. A2) including any outer one-temperature zone.  NY95 define
    the advected fraction as f = 1 - Q-/Q+.  r and Te are the full radial grid and the
    piecewise temperature profile on it.
    """

    # BB99-consistent generalization of NY95 eq. 2.5
    if s == 1:
        Qplus = (9.430e38)*(f**(-1.0))*(((1.0+beta)**(-1.0)) + ((2.0*s)/3.0))*c3*m*mdot*np.log(rmax/rmin)
    else:
        Qplus = (9.430e38)*(f**(-1.0))*(((1.0+beta)**(-1.0)) + ((2.0*s)/3.0))*c3*m*mdot*((1.0-s)**(-1.0))*((rmin**(-1.0+s)) - (rmax**(-1.0+s)))

    Lnu_synch = assemble_synch_spectrum(nu_arr,Te0,t,rmin,rmax,req,m,mdot,s,alpha,beta,c1,c3,T_synch_min)[0]
    P_synch = np.sum(0.5*(Lnu_synch[1:] + Lnu_synch[0:-1])*(nu_arr[1:] - nu_arr[0:-1]))

    # in a fully one-temperature flow that is everywhere cooler than T_synch_min, there
    # are no synchrotron seed photons, so the Compton contribution is omitted along with
    # the synchrotron (assemble_synch_spectrum returns zeros there)
    if (req <= rmin) and ((Te0/(rmin**(1.0-t))) < T_synch_min):
        P_compt = 0.0
    else:
        P_compt = compute_compt_power(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3,T_synch_min)

    Qminus = (P_synch + P_compt
              + compute_brems_power(r,Te,m,mdot,s,alpha,c1))

    return Qplus, Qminus

##############################################
# electron temperature solver

def solve_temperature(nu_arr,r,m,mdot,f,s,alpha,beta,lambda_w,delta,rmin,rmax,req,T_synch_min,N_Te,logTe0_lo,logTe0_hi,tol_logTe0):
    """
    Solve the electron energy balance (P21 eq. A1) for the electron temperature
    normalization Te0, at a fixed advected fraction f.

    The structure constants c1 and c3 are computed here rather than by the caller.  They
    follow the BB99-consistent generalization of the NY95 self-similar solution to a flow
    with Mdot ~ r**s and a wind: the entropy-balance coefficient is X = epsilon' + 2s/(3f)
    with epsilon' = (1/f)(5/3-gamma)/(gamma-1), the ratio k = c1/c3 is set by the
    angular-momentum balance with the wind removing lambda_w times the local specific
    angular momentum, and the radial-momentum coefficient becomes (5/2 - s).  Setting
    s = 0 recovers the NY95 solution exactly.

    Returns Te0 (K), t (the Te power-law index), c1, c3, and a flag indicating whether the
    solution landed on the boundary of the search interval.

    The balance is a statement about the two-temperature zone, so all integrals run over
    the grid r, which must span [rmin, min(req, rmax)]; rmax here is the outer edge of that
    zone (i.e. the caller passes min(req, rmax)), while req sets the temperature index t.

    If req <= rmin there is no two-temperature zone at all: the entire flow is
    one-temperature with Te = Ti = Te0/r, which is the t = 0 limit of the
    parameterization, and so (Te0, 0) is returned without solving any balance (the grid r
    is then unused).
    """

    # derived quantities, including BB99-consistent self-similar structure
    gamma = (8.0 + (5.0*beta)) / (6.0 + (3.0*beta))
    epsilon_prime = (1.0/f)*(((5.0/3.0) - gamma) / (gamma - 1.0))
    X = epsilon_prime + ((2.0*s)/(3.0*f))

    # the viscous torque diverges as lambda_w approaches (2s+1)/(2s) from below, and the
    # required torque would be negative beyond it
    if lambda_w < 0.0:
        raise RuntimeError('lambda_w must be non-negative.')
    if (s > 0.0) and (lambda_w >= (((2.0*s) + 1.0)/(2.0*s))):
        raise RuntimeError('lambda_w must be smaller than (2s+1)/(2s) = '+str(((2.0*s) + 1.0)/(2.0*s))+' for s = '+str(s)+'.')

    k = (3.0/2.0)*(((2.0*s) + 1.0)/(((2.0*s) + 1.0) - (2.0*s*lambda_w)))
    D = ((5.0/2.0) - s) + ((2.0/3.0)*k*X)

    c3 = 2.0/(D + np.sqrt((D**2.0) + (2.0*(alpha**2.0)*(k**2.0))))
    c1 = k*c3

    # if the equalization radius lies at or inside the inner edge, the entire flow is
    # one-temperature with Te = Ti = Te0/r
    if req <= rmin:
        Te0 = (6.66e12)*beta*c3/(2.08*(1.0+beta))
        return Te0, 0.0, c1, c3, False

    # determine the electron temperature
    def electron_energy_balance(Te0):

        # solve for Te power-law index from the requirement Te(req) = Ti(req)
        t = (1.0 / np.log(req))*np.log((6.66e12)*beta*c3/(2.08*Te0*(1.0+beta)))

        # electron number density radial profile
        ne = (3.158e19)*(alpha**-1.0)*(c1**-1.0)*(m**-1.0)*mdot*(r**((-3.0/2.0) + s))

        # Te and Ti radial profiles
        Te = Te0 / (r**(1.0-t))
        Ti = ((6.66e12)*((1.0+beta)**(-1.0))*beta*c3*(r**(-1.0))) - (1.08*Te)

        # dimensionless temperature
        theta_e = (1.68637e-10)*Te

        ##############################################
        # heating ####################################
        ##############################################

        # viscous heating rate
        if s == 1:
            Qplus = (9.430e38)*(f**(-1.0))*(((1.0+beta)**(-1.0)) + ((2.0*s)/3.0))*c3*m*mdot*np.log(rmax/rmin)
        else:
            Qplus = (9.430e38)*(f**(-1.0))*(((1.0+beta)**(-1.0)) + ((2.0*s)/3.0))*c3*m*mdot*((1.0-s)**(-1.0))*((rmin**(-1.0+s)) - (rmax**(-1.0+s)))

        # electron-ion heating rate
        qie = qie_func(Te,Ti,ne)

        # integrate over volume
        integrand = (3.236e17)*(m**3.0)*qie*(r**2.0)
        Qie = np.sum(0.5*(integrand[1:] + integrand[0:-1])*(r[1:] - r[0:-1]))

        ##############################################
        # cooling ####################################
        ##############################################

        # electron advection
        integrand = (1.013e26)*m*mdot*Te0*(((1.0 - t)/(gammaCV(theta_e) - 1.0)) - (3.0/2.0) + s)*(r**(s+t-2.0))
        Qadve = np.sum(0.5*(integrand[1:] + integrand[0:-1])*(r[1:] - r[0:-1]))

        # eliminate negative advected electron energy
        if Qadve <= 0.0:
            Qadve = 0.0

        # synchrotron emission
        P_synch = compute_synch_power(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3,T_synch_min)

        # inverse Compton emission
        P_compt = compute_compt_power(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3,T_synch_min)

        # bremsstrahlung emission
        P_brems = compute_brems_power(r,Te,m,mdot,s,alpha,c1)

        # total electron heating and cooling
        heating = Qie + (delta*Qplus)
        cooling = Qadve + P_synch + P_compt + P_brems

        return heating - cooling

    # physically allowed Te0 interval
    Te0_t0 = (6.66e12)*beta*c3/(2.08*(1.0+beta))
    Te0_t1 = Te0_t0/req

    # boundaries of search space in log10(Te0)
    Te0_lo = max(10.0**logTe0_lo, Te0_t1)
    Te0_hi = min(10.0**logTe0_hi, Te0_t0)

    if Te0_lo >= Te0_hi:
        raise RuntimeError('No valid Te0 search interval with 0 <= t <= 1.')

    # compute heating-cooling on boundary
    balance_lo = electron_energy_balance(Te0_lo)
    balance_hi = electron_energy_balance(Te0_hi)

    # check to make sure nothing broke
    if (not np.isfinite(balance_lo)) or (not np.isfinite(balance_hi)):
        raise RuntimeError('Non-finite heating-cooling balance at the Te0 search boundaries.')

    # deal with edge cases
    on_boundary = False
    if balance_lo == 0.0:
        Te0 = Te0_lo
        on_boundary = True
    elif balance_hi == 0.0:
        Te0 = Te0_hi
        on_boundary = True
    elif np.sign(balance_lo) == np.sign(balance_hi):
        warnings.warn('heating-cooling does not change sign between 1e'+str(logTe0_lo)+' K and 1e'+str(logTe0_hi)+' K. Returning the boundary with the smaller residual.', RuntimeWarning, stacklevel=2)
        if np.abs(balance_lo) <= np.abs(balance_hi):
            Te0 = Te0_lo
        else:
            Te0 = Te0_hi
        on_boundary = True

    # otherwise, binary search
    else:

        for _ in range(N_Te):

            logTe0_mid = 0.5*(logTe0_lo + logTe0_hi)
            Te0_mid = 10.0**logTe0_mid
            balance_mid = electron_energy_balance(Te0_mid)

            if not np.isfinite(balance_mid):
                raise RuntimeError('Non-finite heating-cooling balance during Te0 binary search.')

            if (balance_mid == 0.0) or ((logTe0_hi - logTe0_lo) <= tol_logTe0):
                logTe0_lo = logTe0_mid
                logTe0_hi = logTe0_mid
                break

            if np.sign(balance_mid) == np.sign(balance_lo):
                logTe0_lo = logTe0_mid
                balance_lo = balance_mid
            else:
                logTe0_hi = logTe0_mid
                balance_hi = balance_mid

        Te0 = 10.0**(0.5*(logTe0_lo + logTe0_hi))

    # solve for Te power-law index
    t = (1.0 / np.log(req))*np.log((6.66e12)*beta*c3/(2.08*Te0*(1.0+beta)))

    return Te0, t, c1, c3, on_boundary

##############################################
# SED main function

def SED(nu,m,mdot,verbose_return=False,verbose=True,s=0.5,alpha=0.2,beta=10.0,f=1.0,delta=0.3,lambda_w=1.0,
    rmin=3.0,rmax=1.0e5,req=1.0e3,T_synch_min=1.0e8,numin=1.0e2,numax=1.0e22,
    N_Te=100,N_r=30,N_nu=20000,logTe0_lo=8.0,logTe0_hi=12.0,tol_logTe0=1.0e-6,
    solve_f=False,N_f=100,tol_f=1.0e-4,damp_f=1.0,f_min=1.0e-3):
    """
    Compute the SED of an advection-dominated accretion flow (ADAF), following Appendix A
    of Pesce et al. (2021), which follows Mahadevan (1997, M97) and Narayan & Yi (1995, NY95).
    
    Inputs:
    nu: array of frequencies at which to compute the SED, in Hz
    m: mass of BH, in solar masses
    mdot: mass accretion rate at r = 1, in units of Mdot_Edd = L_Edd/(eta*c**2), with eta = 0.1

        Note: mdot is not a luminosity ratio.  The flow is radiatively inefficient, so
        L_bol/L_Edd falls 1-4 orders of magnitude below mdot.  The accretion rate varies
        with radius as Mdot(r) = mdot*Mdot_Edd*r**s, so the rate at the inner edge is
        mdot*rmin**s (~ 1.73*mdot for the default rmin = 3, s = 0.5).

    Optional inputs:
    verbose_return: if True, also return Te0 and the individual SED components
    verbose: if True, print the converged electron temperature
    s: power-law index of the radial accretion-rate profile, Mdot ~ r**s
    alpha: viscosity parameter
    beta: plasma beta = gas pressure / magnetic pressure.  Note: this differs from M97, whose
          beta is gas pressure / total pressure; beta_M97 = beta/(1+beta).
    f: fraction of the viscously dissipated energy that is advected.  NY95 define this as
       f = 1 - Q^-/Q^+, so f = 1 corresponds to no radiative losses at all; it is a good
       approximation only at low accretion rates (see solve_f).
    delta: fraction of the viscous heating deposited directly into the electrons
    lambda_w: specific angular momentum carried away by the wind, in units of the local
         disk value Omega*R**2 (BB99).  lambda_w = 1 is a gasdynamical wind that carries
         off its own angular momentum and exerts no reaction torque on the disk (BB99
         case iv); lambda_w > 1 represents magnetized winds, with the disk Bernoulli
         parameter turning negative for lambda_w > (2s+1)/(3s).  Must satisfy
         0 <= lambda_w < (2s+1)/(2s); irrelevant when s = 0.
    rmin, rmax: inner and outer radii of the flow, in Schwarzschild radii
    req: radius at which the ion and electron temperatures become equal, in Schwarzschild
         radii.  Inside req the flow is two-temperature with Te = Te0 r**(-(1-t)); at and
         beyond req the electrons are locked to the ions, Te = Ti ~ 1/r.  Defaults to 1000,
         the NY95/M97 value, independently of rmax; pass req=None to tie it to rmax instead
         (which reproduces the P21 model for any rmax).  req < rmax adds an outer one-
         temperature zone whose emission (mainly bremsstrahlung, plus the low-frequency
         synchrotron tail) is included in the SED but, being powered by the ions, not in the
         electron energy balance that fixes Te0.  req > rmax is allowed and describes a flow
         that is truncated before the temperatures equilibrate.  req <= rmin is also
         allowed: the equalization radius then lies at or inside the inner edge, the
         entire flow is one-temperature with Te = Ti = Te0/r (the t = 0 limit of
         the parameterization), and no electron energy balance is solved.
    T_synch_min: electron temperature below which thermal synchrotron emission is not
         counted, in K.  Following M97 (sec. 4.1), the outer synchrotron anchor -- in the
         one-temperature zone and in the two-temperature zone alike -- is placed no
         further out than where the electron temperature falls to this value, and a zone
         that is everywhere cooler than this emits no synchrotron (or Compton) at all.
    numin, numax: limits of the internal frequency grid, in Hz; requested frequencies
                  outside this range are returned as zero
    N_Te: maximum number of bisection iterations for Te0
    N_r: number of radial grid points
    N_nu: number of frequency grid points
    logTe0_lo, logTe0_hi: log10 bounds of the Te0 search, in K
    tol_logTe0: convergence tolerance of the Te0 search, in dex
    solve_f: if True, treat the input f as a starting guess and iterate the outer loop
             f <- 1 - Q^-/Q^+ until it is self-consistent, as NY95 do.  This costs one
             extra electron-temperature solve per iteration (typically 2-5, up to ~20 near
             the critical accretion rate) and raises RuntimeError where no advection-
             dominated solution exists.  Default False, which keeps f fixed at its input.
    N_f: maximum number of outer iterations when solve_f is True
    tol_f: convergence tolerance on f when solve_f is True
    damp_f: damping applied to the f update; 1.0 is undamped (fastest), lower it if the
            iteration oscillates
    f_min: value of the implied f at or below which no advection-dominated solution is
           deemed to exist

    Returns:
    Lnu: total luminosity density as a function of frequency, in erg/s/Hz
    nu_p: peak synchrotron frequency, in Hz
    if verbose_return is set to True, additionally:
        Te0: Normalization of the electron temperature profile, in K.  The profile is
             Te(r) = Te0/r**(1-t), so Te0 is its value extrapolated to r = 1, which lies
             inside rmin; Te0 is hotter than every electron in the flow.  The hottest
             physical temperature is Te(rmin) = Te0/rmin**(1-t).  In the one-temperature
             regime (req <= rmin), t = 0 by construction, so Te(r) = Te0/r.
        Lnu_synch, Lnu_compt, Lnu_brems: the individual components, in erg/s/Hz
        f_implied: the advected fraction implied by the computed radiative losses,
                   1 - Q^-/Q^+.  The input f should match this for self-consistency;
                   f = 1 is a good approximation only for mdot << 1e-3.
    """

    ##############################################
    # warnings

    if (np.log10(mdot) >= -1.7):
        warnings.warn('the input accretion rate is larger than the maximum log(mdot) = -1.7, which will yield unphysical results', RuntimeWarning, stacklevel=2)
    if ((nu < numin).sum() > 0):
        warnings.warn('the minimum input frequency is smaller than the minimum internally-computed frequency of '+str(numin)+' Hz', RuntimeWarning, stacklevel=2)
    if ((nu > numax).sum() > 0):
        warnings.warn('the maximum input frequency is larger than the maximum internally-computed frequency of '+str(numax)+' Hz', RuntimeWarning, stacklevel=2)

    ##############################################
    # radii

    if req is None:
        req = rmax
    if rmax <= rmin:
        raise RuntimeError('rmax must exceed rmin.')

    # if req <= rmin, the equalization radius lies at or inside the inner edge and the
    # entire flow is one-temperature (Te = Ti = Te0/r); no electron energy balance is solved
    one_temperature = (req <= rmin)

    # outer edge of the two-temperature zone, over which the electron balance is solved
    r_bal = min(req,rmax)

    ##############################################
    # required arrays

    if one_temperature:
        # no two-temperature zone; a single grid covers the whole (one-temperature) flow
        r = 10.0**np.linspace(np.log10(rmin),np.log10(rmax),N_r)
        r_full = r
    else:
        r = 10.0**np.linspace(np.log10(rmin),np.log10(r_bal),N_r)
        if req < rmax:
            # outer one-temperature zone; req is duplicated at the join, which the trapezoid
            # rule handles as a zero-width interval
            r_full = np.concatenate([r, 10.0**np.linspace(np.log10(req),np.log10(rmax),N_r)])
        else:
            r_full = r
    nu_arr = 10.0**np.linspace(np.log10(numin),np.log10(numax),N_nu)

    ##############################################
    # determine the electron temperature, and optionally the advected fraction f

    if not solve_f:

        Te0, t, c1, c3, on_boundary = solve_temperature(nu_arr,r,m,mdot,f,s,alpha,beta,lambda_w,delta,rmin,r_bal,req,T_synch_min,N_Te,logTe0_lo,logTe0_hi,tol_logTe0)

    else:

        # NY95 define f as the fraction of the viscously dissipated energy that is advected,
        # so self-consistency requires f = 1 - Q^-/Q^+.  Solve the fixed point by simple
        # iteration: each pass re-solves the electron energy balance at the current f, then
        # updates f from the resulting radiative losses.
        f_converged = False
        for _ in range(N_f):

            Te0, t, c1, c3, on_boundary = solve_temperature(nu_arr,r,m,mdot,f,s,alpha,beta,lambda_w,delta,rmin,r_bal,req,T_synch_min,N_Te,logTe0_lo,logTe0_hi,tol_logTe0)
            K = (6.66e12)*beta*c3/(2.08*(1.0+beta))
            Qplus, Qminus = compute_powers(nu_arr,r_full,Te_profile(r_full,Te0,t,req,K),Te0,t,f,rmin,rmax,req,m,mdot,s,alpha,beta,c1,c3,T_synch_min)
            f_target = 1.0 - (Qminus/Qplus)

            # below this the flow radiates more than viscosity supplies and no
            # advection-dominated solution exists (this is NY95's critical accretion rate)
            if f_target <= f_min:
                raise RuntimeError('No self-consistent advected fraction: the implied f = '
                                   +str(np.round(f_target,3))+' has fallen to or below f_min = '
                                   +str(f_min)+', so the accretion rate is above the critical '
                                   'value for an advection-dominated solution.')

            # Te0 was solved at this f, so if f already reproduces itself we are done
            if np.abs(f_target - f) <= tol_f:
                f_converged = True
                break

            f = ((1.0 - damp_f)*f) + (damp_f*min(f_target,1.0))

        if not f_converged:
            warnings.warn('the self-consistent solve for f did not converge to within tol_f = '
                          +str(tol_f)+' in N_f = '+str(N_f)+' iterations; the returned f is '
                          +str(np.round(f,4)), RuntimeWarning, stacklevel=2)

    # print out the solved-for temperature
    if verbose:
        if one_temperature:
            print('req <= rmin: the flow is one-temperature everywhere; '
                  +'Te(rmin) = Ti(rmin) = '+str(np.round(Te0/(rmin*1.0e9),2))+' GK.')
        else:
            print('Electron temperature normalization is '+str(np.round(Te0/(1.0e9),2))+' GK at r = 1; '
                  +'Te(rmin) = '+str(np.round(Te0/((rmin**(1.0-t))*1.0e9),2))+' GK.')

    # check for extreme temperature values
    if on_boundary:
        warnings.warn('the self-consistently identified temperature is '+str(np.round((Te0/(1.0e9)),2))+' GK, which is on the boundary of the tested temperature range.', RuntimeWarning, stacklevel=2)

    ##############################################
    # check the self-consistency of the assumed advected fraction f

    # Recompute the implied value of f and warn if it is badly violated;
    # f = 1 is the low-accretion-rate approximation
    K = (6.66e12)*beta*c3/(2.08*(1.0+beta))
    Te_full = Te_profile(r_full,Te0,t,req,K)
    Qplus_final, Qminus_final = compute_powers(nu_arr,r_full,Te_full,Te0,t,f,rmin,rmax,req,m,mdot,s,alpha,beta,c1,c3,T_synch_min)
    f_implied = 1.0 - (Qminus_final/Qplus_final)
    if f_implied <= 0.0:
        warnings.warn('the flow radiates more energy than viscosity dissipates (implied '
                      'advected fraction f = '+str(np.round(f_implied,3))+' <= 0); the '
                      'accretion rate is above the critical value for an advection-'
                      'dominated solution and the result is unphysical',
                      RuntimeWarning, stacklevel=2)
    elif f_implied < 0.9*f:
        warnings.warn('the assumed advected fraction f = '+str(f)+' is inconsistent with '
                      'the computed radiative losses, which imply f = '
                      +str(np.round(f_implied,3))+'; f = 1 is only a good approximation at '
                      'low accretion rates', RuntimeWarning, stacklevel=2)

    ##############################################
    # construct the spectrum

    # determine critical synchrotron frequencies and luminosities
    Lnu_synch_full, nu_p, L_p, nu_bal, L_bal, nu_min, Lnu_min = assemble_synch_spectrum(nu_arr,Te0,t,rmin,rmax,req,m,mdot,s,alpha,beta,c1,c3,T_synch_min)

    # the component arrays contain exact zeros, whose log10 is -inf; that is intended
    # (10**-inf = 0), so silence only the divide warning it raises, and only here
    errstate_ctx = np.errstate(divide='ignore')
    errstate_ctx.__enter__()

    # synchrotron emission
    synch_interpolator = interpolate.interp1d(np.log10(nu_arr), np.log10(Lnu_synch_full),kind='linear',bounds_error=False,fill_value=-np.inf)
    Lnu_synch = 10.0**synch_interpolator(np.log10(nu))

    # inverse Compton emission (omitted when there are no synchrotron seed photons,
    # i.e. when assemble_synch_spectrum returned zeros with L_p = 0)
    if L_p > 0.0:
        Lnu_compt_full = compute_compt_spectrum(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3,nu_p,L_p)
    else:
        Lnu_compt_full = np.zeros_like(nu_arr)
    compt_interpolator = interpolate.interp1d(np.log10(nu_arr), np.log10(Lnu_compt_full),kind='linear',bounds_error=False,fill_value=-np.inf)
    Lnu_compt = 10.0**compt_interpolator(np.log10(nu))

    # bremsstrahlung emission
    Lnu_brems_full = compute_brems_spectrum(nu_arr,r_full,Te_full,m,mdot,s,alpha,c1)
    brems_interpolator = interpolate.interp1d(np.log10(nu_arr), np.log10(Lnu_brems_full),kind='linear',bounds_error=False,fill_value=-np.inf)
    Lnu_brems = 10.0**brems_interpolator(np.log10(nu))

    errstate_ctx.__exit__(None, None, None)
    
    # combine
    Lnu = Lnu_synch + Lnu_compt + Lnu_brems

    ##############################################
    
    if verbose_return:
        return Lnu, nu_p, Te0, Lnu_synch, Lnu_compt, Lnu_brems, f_implied
    else:
        return Lnu, nu_p
