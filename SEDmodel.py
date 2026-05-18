import numpy as np
from scipy import interpolate
from scipy.special import kn
np.seterr(divide='ignore')

##############################################
# electron-ion heating functions

def qie_func(Te,Ti,ne):

    # compute dimensionless temperatures
    theta_e = (1.68637e-10)*Te
    theta_i = (9.18426e-14)*Ti

    qie = (5.624e-32)*(((ne**2.0)*(Ti - Te))/(kn(2,1.0/theta_e)*kn(2,1.0/theta_i)))*(((((2.0*((theta_e + theta_i)**2.0)) + 1.0) / (theta_e + theta_i))*kn(1,(theta_e+theta_i)/(theta_e*theta_i))) + (2.0*kn(0,(theta_e+theta_i)/(theta_e*theta_i))))

    return qie

def qie_func_approx(Te,Ti,ne):

    # compute dimensionless temperatures
    theta_e = (1.68637e-10)*Te
    theta_i = (9.18426e-14)*Ti

    qie = (5.624e-32)*(((ne**2.0)*(Ti - Te))/kn(2,1.0/theta_e))*(2.0 + (2.0*theta_e) + (1.0/theta_e))*np.exp(-1.0/theta_e)

    return qie

def qie_wrapper(Te,Ti,ne,r,rthresh=50.0):

    # initialize array
    qie = np.zeros_like(Te)

    # for small r, use the exact expression:
    ind = (r <= rthresh)
    qie[ind] = qie_func(Te[ind],Ti[ind],ne[ind])

    # for large r, use the approximation:
    ind = (r > rthresh)
    qie[ind] = qie_func_approx(Te[ind],Ti[ind],ne[ind])

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

def compute_peak_freq(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3):

    # Te
    Te = Te0 / (rmin**(1.0-t))

    # dimensionless temperature
    theta_e = (1.68637e-10)*Te

    # electron number density
    ne = (3.158e19)*(alpha**-1.0)*(c1**-1.0)*(m**-1.0)*mdot*(rmin**((-3.0/2.0) + s))

    # gyro frequency
    nu_b = (3.998e15)*((1+beta)**(-1.0/2.0))*(alpha**(-1.0/2.0))*(c1**(-1.0/2.0))*(c3**(1.0/2.0))*(m**(-1.0/2.0))*(mdot**(1.0/2.0))*(rmin**((-5.0/4.0) + (s/2.0)))

    # dimensionless frequency
    xM = 2.0*nu_arr/(3.0*nu_b*(theta_e**2.0))

    # find critical frequency at this radius
    volume = (1.896e8)*(relMax(xM)/kn(2,1.0/theta_e))*(alpha**(-1.0))*(c1**(-1.0))*(m**2.0)*mdot*nu_arr*(rmin**((3.0/2.0) + s))
    surface = (1.058e-24)*(nu_arr**2.0)*Te0*(m**2.0)*(rmin**(1.0+t))

    # critical frequency
    nu_p, L_p = find_critical_freq(nu_arr,volume,surface)

    return nu_p, L_p

def compute_min_freq(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3):

    # Te
    Te = Te0 / (rmax**(1.0-t))

    # dimensionless temperature
    theta_e = (1.68637e-10)*Te

    # electron number density
    ne = (3.158e19)*(alpha**-1.0)*(c1**-1.0)*(m**-1.0)*mdot*(rmax**((-3.0/2.0) + s))

    # gyro frequency
    nu_b = (3.998e15)*((1+beta)**(-1.0/2.0))*(alpha**(-1.0/2.0))*(c1**(-1.0/2.0))*(c3**(1.0/2.0))*(m**(-1.0/2.0))*(mdot**(1.0/2.0))*(rmax**((-5.0/4.0) + (s/2.0)))

    # dimensionless frequency
    xM = 2.0*nu_arr/(3.0*nu_b*(theta_e**2.0))

    # find critical frequency at this radius
    volume = (1.896e8)*(relMax(xM)/kn(2,1.0/theta_e))*(alpha**(-1.0))*(c1**(-1.0))*(m**2.0)*mdot*nu_arr*(rmax**((3.0/2.0) + s))
    surface = (1.058e-24)*(nu_arr**2.0)*Te0*(m**2.0)*(rmax**(1.0+t))

    # critical frequency
    nu_min, Lnu_min = find_critical_freq(nu_arr,volume,surface)

    return nu_min, Lnu_min

def compute_synch_power(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3):

    ##############################################
    # get critical frequency at which synchrotron becomes optically thin at rmin

    # Te
    Te = Te0 / (rmin**(1.0-t))

    # dimensionless temperature
    theta_e = (1.68637e-10)*Te

    # gyro frequency
    nu_b = (3.998e15)*((1+beta)**(-1.0/2.0))*(alpha**(-1.0/2.0))*(c1**(-1.0/2.0))*(c3**(1.0/2.0))*(m**(-1.0/2.0))*(mdot**(1.0/2.0))*(rmin**((-5.0/4.0) + (s/2.0)))

    # dimensionless frequency
    xM = 2.0*nu_arr/(3.0*nu_b*(theta_e**2.0))

    # find critical frequency at this radius
    volume = (1.896e8)*(relMax(xM)/kn(2,1.0/theta_e))*(alpha**(-1.0))*(c1**(-1.0))*(m**2.0)*mdot*nu_arr*(rmin**((3.0/2.0) + s))
    surface = (1.058e-24)*(nu_arr**2.0)*Te0*(m**2.0)*(rmin**(1.0+t))

    # critical frequency
    nu_p, Lnu_p = find_critical_freq(nu_arr,volume,surface)

    # save the spectrum at rmin
    Lnu_rmin = np.zeros_like(nu_arr)
    ind_lo = (nu_arr <= nu_p)
    Lnu_rmin[ind_lo] = surface[ind_lo]
    ind_hi = (nu_arr > nu_p)
    Lnu_rmin[ind_hi] = volume[ind_hi]

    ##############################################
    # get critical frequency at which synchrotron becomes optically thin at rmax

    # Te
    Te = Te0 / (rmax**(1.0-t))

    # dimensionless temperature
    theta_e = (1.68637e-10)*Te

    # gyro frequency
    nu_b = (3.998e15)*((1+beta)**(-1.0/2.0))*(alpha**(-1.0/2.0))*(c1**(-1.0/2.0))*(c3**(1.0/2.0))*(m**(-1.0/2.0))*(mdot**(1.0/2.0))*(rmax**((-5.0/4.0) + (s/2.0)))

    # dimensionless frequency
    xM = 2.0*nu_arr/(3.0*nu_b*(theta_e**2.0))

    # find critical frequency at this radius
    volume = (1.896e8)*(relMax(xM)/kn(2,1.0/theta_e))*(alpha**(-1.0))*(c1**(-1.0))*(m**2.0)*mdot*nu_arr*(rmax**((3.0/2.0) + s))
    surface = (1.058e-24)*(nu_arr**2.0)*Te0*(m**2.0)*(rmax**(1.0+t))
    
    # critical frequency
    nu_min, Lnu_min = find_critical_freq(nu_arr,volume,surface)

    # construct synchrotron spetrum
    Lnu_synch = np.zeros_like(nu_arr)

    # synchrotron emission below nu_min is blackbody
    index = (nu_arr <= nu_min)
    Lnu_synch[index] = Lnu_min*((nu_arr[index]/nu_min)**2.0)
    nu_lo = np.max(nu_arr[index])
    L_lo = np.copy(Lnu_synch[index])[np.argmax(nu_arr[index])]

    # synchrotron emission above nu_p is Maxwellian
    index = (nu_arr >= nu_p)
    Lnu_synch[index] = np.copy(Lnu_rmin[index])
    nu_hi = np.min(nu_arr[index])
    L_hi = np.copy(Lnu_synch[index])[np.argmin(nu_arr[index])]

    # synchrotron emission bewteen nu_min and nu_p is a power-law
    pl_exp = np.log(L_hi/L_lo) / np.log(nu_hi/nu_lo)
    index = ((nu_arr > nu_min) & (nu_arr < nu_p))
    Lnu_synch[index] = L_lo*((nu_arr[index]/nu_lo)**pl_exp)

    # total synchrotron power is integral over frequency
    P_synch = np.sum(0.5*(Lnu_synch[1:] + Lnu_synch[0:-1])*(nu_arr[1:] - nu_arr[0:-1]))

    return P_synch

def compute_synch_spectrum(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3,nu_p,L_p,nu_min,Lnu_min):

    # construct synchrotron spetrum
    Lnu_synch = np.zeros_like(nu_arr)

    # synchrotron emission below nu_min is blackbody
    index = (nu_arr <= nu_min)
    Lnu_synch[index] = Lnu_min*((nu_arr[index]/nu_min)**2.0)

    # synchrotron emission above nu_p is Maxwellian
    index = (nu_arr >= nu_p)

    Te = Te0 / (rmin**(1.0-t))
    theta_e = (1.68637e-10)*Te
    nu_b = (3.998e15)*((1+beta)**(-1.0/2.0))*(alpha**(-1.0/2.0))*(c1**(-1.0/2.0))*(c3**(1.0/2.0))*(m**(-1.0/2.0))*(mdot**(1.0/2.0))*(rmin**((-5.0/4.0) + (s/2.0)))
    xM = 2.0*nu_arr[index]/(3.0*nu_b*(theta_e**2.0))
    Lnu_synch[index] = (1.896e8)*(relMax(xM)/kn(2,1.0/theta_e))*(alpha**(-1.0))*(c1**(-1.0))*(m**2.0)*mdot*nu_arr[index]*(rmin**((3.0/2.0) + s))

    nu_hi = np.min(nu_arr[index])
    L_hi = np.copy(Lnu_synch[index])[np.argmin(nu_arr[index])]

    # synchrotron emission bewteen nu_min and nu_p is a power-law
    pl_exp = np.log(L_hi/Lnu_min) / np.log(nu_hi/nu_min)
    index = ((nu_arr > nu_min) & (nu_arr < nu_p))
    Lnu_synch[index] = Lnu_min*((nu_arr[index]/nu_min)**pl_exp)

    return Lnu_synch

##############################################
# inverse Compton functions

def compute_compt_spectrum(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3,nu_p,L_p):

    # optical depth to electron scattering
    tau_es = 6.205*(alpha**(-1.0))*(c1**(-1.0))*mdot*rmin**(-(1.0/2.0) + s)

    # dimensionless temperature
    theta_e0 = (1.68637e-10)*Te0

    # amplification factor
    A = 1.0 + (4.0*theta_e0) + (16.0*(theta_e0**2.0))

    # power-law index
    alpha_c = -np.log(tau_es) / np.log(A)

    # peak frequency
    nu_f = (6.251e10)*Te0

    # construct Compton spectrum
    Lnu_compt = L_p*((nu_arr/nu_p)**(-alpha_c))

    # adding in exponential cutoffs at nu_p and nu_f
    Lnu_compt *= np.exp(-((nu_arr/(0.5*nu_f))**2.0))
    Lnu_compt *= np.exp(-((nu_arr/(1.0*nu_p))**-4.0))

    # # original, sharp cutoffs
    # ind_lo = (nu_arr < nu_p)
    # Lnu_compt[ind_lo] = 0.0

    # ind_hi = (nu_arr > nu_f)
    # Lnu_compt[ind_hi] = 0.0

    return Lnu_compt

def compute_compt_power(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3):

    # optical depth to electron scattering
    tau_es = 6.205*(alpha**(-1.0))*(c1**(-1.0))*mdot*rmin**(-(1.0/2.0) + s)

    # dimensionless temperature
    theta_e0 = (1.68637e-10)*Te0

    # amplification factor
    A = 1.0 + (4.0*theta_e0) + (16.0*(theta_e0**2.0))

    # power-law index
    alpha_c = -np.log(tau_es) / np.log(A)

    # peak frequency
    nu_f = (6.251e10)*Te0

    # get peak synchrotron frequency
    nu_p, L_p = compute_peak_freq(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3)

    # compute total power
    P_compt = (nu_p*L_p/(1.0 - alpha_c))*(((nu_f/nu_p)**(1.0-alpha_c)) - 1.0)

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

def compute_brems_power(r,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3):

    # solve for Te power-law index
    t = (1.0 / np.log(rmax))*np.log((6.66e12)*beta*c3/(2.08*Te0*(1.0+beta)))

    # Te radial profile
    Te = Te0 / (r**(1.0-t))

    # dimensionless temperature
    theta_e = (1.68637e-10)*Te

    # integrate over volume
    integrand = (4.776e34)*(alpha**(-2.0))*(c1**(-2.0))*m*(mdot**2.0)*bremsF(theta_e)*(r**(-1.0 + (2.0*s)))
    P_brems = np.sum(0.5*(integrand[1:] + integrand[0:-1])*(r[1:] - r[0:-1]))

    return P_brems

def compute_brems_spectrum(nu_arr,r,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3):

    # solve for Te power-law index
    t = (1.0 / np.log(rmax))*np.log((6.66e12)*beta*c3/(2.08*Te0*(1.0+beta)))

    # Te radial profile
    Te = Te0 / (r**(1.0-t))

    # dimensionless temperature
    theta_e = (1.68637e-10)*Te

    # intialize array
    Lnu_brems = np.zeros_like(nu_arr)
    
    # integrate over radius at each frequency
    if len(nu_arr) != 0:
        for inu, nu in enumerate(nu_arr):
            integrand = (2.292e24)*(alpha**(-2.0))*(c1**(-2.0))*m*(mdot**2.0)*(Te0**(-1.0))*bremsF(theta_e)*np.exp(-(4.799e-11)*(nu/Te))*(r**((2.0*s) - t))
            Lnu_brems[inu] = np.sum(0.5*(integrand[1:] + integrand[0:-1])*(r[1:] - r[0:-1]))

    return Lnu_brems

##############################################
# SED main function

def SED(nu,m,mdot,verbose_return=False,s=0.5,alpha=0.2,beta=10.0,f=1.0,delta=0.3,rmin=3.0,
    rmax=1000.0,numin=1.0e2,numax=1.0e22,N_Te=100,N_r=30,N_nu=20000,rthresh=50.0,
    logTe0_lo=8.0,logTe0_hi=12.0,tol_logTe0=1.0e-6):
    """
    Inputs:
    nu: array of frequencies at which to compute the SED
    m: mass of BH, in solar masses
    mdot: Eddington ratio = L/L_Edd = eta*(c**2)*Mdot/L_Edd
    
    Returns:
    if verbose_return is set to False:
        Lnu: total luminosity density as a function of frequency
        nu_p: peak synchrotron frequency
    if verbose_return is set to True:
        Lnu: total luminosity density as a function of frequency
        nu_p: peak synchrotron frequency
        Te0: electron temperature at small radii
        Lnu_synch: synchrotron luminosity density as a function of frequency
        Lnu_compt: inverse Compton luminosity density as a function of frequency
        Lnu_brems: bremsstrahlung luminosity density as a function of frequency
    """

    ##############################################
    # warnings

    if (np.log10(mdot) >= -1.7):
        print('WARNING: the input accretion rate is larger than the maximum log(mdot) = -1.7, which will yield unphysical results')
    if ((nu < numin).sum() > 0):
        print('WARNING: the minimum input frequency is smaller than the minimum internally-computed frequency of '+str(numin)+' Hz')
    if ((nu > numax).sum() > 0):
        print('WARNING: the maximum input frequency is larger than the maximum internally-computed frequency of '+str(numax)+' Hz')

    ##############################################
    # derived quantities

    gamma = (8.0 + (5.0*beta)) / (6.0 + (3.0*beta))
    epsilon_prime = (1.0/f)*(((5.0/3.0) - gamma) / (gamma - 1.0))

    c1 = ((5.0 + (2.0*epsilon_prime)) / (3.0*(alpha**2.0)))*(np.sqrt(1.0 + ((18.0*(alpha**2.0))/((5.0 + (2.0*epsilon_prime))**2.0))) - 1.0)
    c3 = (2.0*(5.0 + (2.0*epsilon_prime)) / (9.0*(alpha**2.0)))*(np.sqrt(1.0 + ((18.0*(alpha**2.0))/((5.0 + (2.0*epsilon_prime))**2.0))) - 1.0)

    ##############################################
    # required arrays

    r = 10.0**np.linspace(np.log10(rmin),np.log10(rmax),N_r)
    nu_arr = 10.0**np.linspace(np.log10(numin),np.log10(numax),N_nu)

    ##############################################
    # determine the electron temperature

    def electron_energy_balance(Te0):

        # solve for Te power-law index
        t = (1.0 / np.log(rmax))*np.log((6.66e12)*beta*c3/(2.08*Te0*(1.0+beta)))

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
            Qplus = (9.430e38)*(f**(-1.0))*((1.0+beta)**(-1.0))*(c3**(1.0/2.0))*m*mdot*np.log(rmax/rmin)
        else:
            Qplus = (9.430e38)*(f**(-1.0))*((1.0+beta)**(-1.0))*(c3**(1.0/2.0))*m*mdot*((1.0-s)**(-1.0))*((rmin**(-1.0+s)) - (rmax**(-1.0+s)))

        # electron-ion heating rate
        qie = qie_wrapper(Te,Ti,ne,r,rthresh=rthresh)

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
        P_synch = compute_synch_power(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3)

        # inverse Compton emission
        P_compt = compute_compt_power(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3)

        # bremsstrahlung emission
        P_brems = compute_brems_power(r,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3)

        # total electron heating and cooling
        heating = Qie + (delta*Qplus)
        cooling = Qadve + P_synch + P_compt + P_brems

        return heating - cooling

    # physically allowed Te0 interval
    Te0_t0 = (6.66e12)*beta*c3/(2.08*(1.0+beta))
    Te0_t1 = Te0_t0/rmax

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
        print('WARNING: heating-cooling does not change sign between 1e'+str(logTe0_lo)+' K and 1e'+str(logTe0_hi)+' K. Returning the boundary with the smaller residual.')
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

    # print it out
    print('Electron temperature is '+str(np.round(Te0/(1.0e9),2))+' GK.')

    # check for extreme temperature values
    if on_boundary:
        print('WARNING: the self-consistently identified temperature is '+str(np.round((Te0/(1.0e9)),2))+' GK, which is on the boundary of the tested temperature range.')

    ##############################################
    # construct the spectrum

    # solve for Te power-law index
    t = (1.0 / np.log(rmax))*np.log((6.66e12)*beta*c3/(2.08*Te0*(1.0+beta)))

    # determine critical synchrotron frequencies and luminosities
    nu_p, L_p = compute_peak_freq(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3)
    nu_min, Lnu_min = compute_min_freq(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3)

    # synchrotron emission
    Lnu_synch_full = compute_synch_spectrum(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3,nu_p,L_p,nu_min,Lnu_min)
    synch_interpolator = interpolate.interp1d(np.log10(nu_arr), np.log10(Lnu_synch_full),kind='linear',fill_value=0.0)
    Lnu_synch = 10.0**synch_interpolator(np.log10(nu))

    # inverse Compton emission
    Lnu_compt_full = compute_compt_spectrum(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3,nu_p,L_p)
    compt_interpolator = interpolate.interp1d(np.log10(nu_arr), np.log10(Lnu_compt_full),kind='linear',fill_value=0.0)
    Lnu_compt = 10.0**compt_interpolator(np.log10(nu))

    # bremsstrahlung emission
    Lnu_brems_full = compute_brems_spectrum(nu_arr,r,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3)
    brems_interpolator = interpolate.interp1d(np.log10(nu_arr), np.log10(Lnu_brems_full),kind='linear',fill_value=0.0)
    Lnu_brems = 10.0**brems_interpolator(np.log10(nu))
    
    # combine
    Lnu = Lnu_synch + Lnu_compt + Lnu_brems

    ##############################################
    
    if verbose_return:
        return Lnu, nu_p, Te0, Lnu_synch, Lnu_compt, Lnu_brems
    else:
        return Lnu, nu_p
