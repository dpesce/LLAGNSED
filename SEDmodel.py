import numpy as np
from scipy.special import kn

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
    nu_p = nu_arr[np.argmin(np.abs(volume - surface))]
    L_p = 0.5*(surface[np.argmin(np.abs(volume - surface))] + volume[np.argmin(np.abs(volume - surface))])

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
    nu_min = nu_arr[np.argmin(np.abs(volume - surface))]
    Lnu_min = 0.5*(surface[np.argmin(np.abs(volume - surface))] + volume[np.argmin(np.abs(volume - surface))])

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
    nu_p = nu_arr[np.argmin(np.abs(volume - surface))]
    Lnu_p = 0.5*(surface[np.argmin(np.abs(volume - surface))] + volume[np.argmin(np.abs(volume - surface))])

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
    nu_min = nu_arr[np.argmin(np.abs(volume - surface))]
    Lnu_min = 0.5*(surface[np.argmin(np.abs(volume - surface))] + volume[np.argmin(np.abs(volume - surface))])

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
            integrand = (2.292e24)*(alpha**(-2.0))*(c1**(-2.0))*m*(mdot**2.0)*(Te0**(-1.0))*bremsF(theta_e)*np.exp(-(4.799e-11)*(nu/Te))*(r**(-2.0 + (2.0*s) + t))
            Lnu_brems[inu] = np.sum(0.5*(integrand[1:] + integrand[0:-1])*(r[1:] - r[0:-1]))

    return Lnu_brems

##############################################
# SED main function

def SED(nu,m,mdot,verbose_return=False,s=0.5,alpha=0.2,beta=10.0,f=1.0,delta=0.3,rmin=3.0,rmax=1000.0,N_Te=200,N_r=30,N_nu=20000,rthresh=50.0):
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
    # derived quantities

    gamma = (8.0 + (5.0*beta)) / (6.0 + (3.0*beta))
    epsilon_prime = (1.0/f)*(((5.0/3.0) - gamma) / (gamma - 1.0))

    c1 = ((5.0 + (2.0*epsilon_prime)) / (3.0*(alpha**2.0)))*(np.sqrt(1.0 + ((18.0*(alpha**2.0))/((5.0 + (2.0*epsilon_prime))**2.0))) - 1.0)
    c3 = (2.0*(5.0 + (2.0*epsilon_prime)) / (9.0*(alpha**2.0)))*(np.sqrt(1.0 + ((18.0*(alpha**2.0))/((5.0 + (2.0*epsilon_prime))**2.0))) - 1.0)

    ##############################################
    # required arrays

    r = 10.0**np.linspace(np.log10(rmin),np.log10(rmax),N_r)
    nu_arr = 10.0**np.linspace(2.0,22.0,N_nu)

    ##############################################
    # determining the electron temperature

    Te0_testarr = 10.0**np.linspace(8.0,12.0,N_Te)

    heating = np.zeros_like(Te0_testarr)
    cooling = np.zeros_like(Te0_testarr)

    Qplus_arr = np.zeros_like(Te0_testarr)
    Qie_arr = np.zeros_like(Te0_testarr)
    Qadve_arr = np.zeros_like(Te0_testarr)
    Ps_arr = np.zeros_like(Te0_testarr)
    Pc_arr = np.zeros_like(Te0_testarr)
    Pb_arr = np.zeros_like(Te0_testarr)
    for iT, Te0 in enumerate(Te0_testarr):

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

        Qplus_arr[iT] = Qplus

        # electron-ion heating rate
        qie = qie_wrapper(Te,Ti,ne,r,rthresh=rthresh)

        # integrate over volume
        integrand = (3.236e17)*(m**3.0)*qie*(r**2.0)
        Qie = np.sum(0.5*(integrand[1:] + integrand[0:-1])*(r[1:] - r[0:-1]))

        Qie_arr[iT] = Qie

        ##############################################
        # cooling ####################################
        ##############################################

        # electron advection
        integrand = (1.013e26)*m*mdot*Te0*(((1.0 - t)/(gammaCV(theta_e) - 1.0)) - (3.0/2.0) + s)*(r**(s+t-2.0))
        Qadve = np.sum(0.5*(integrand[1:] + integrand[0:-1])*(r[1:] - r[0:-1]))
        Qadve_arr[iT] = Qadve

        # synchrotron emission
        P_synch = compute_synch_power(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3)
        Ps_arr[iT] = P_synch

        # inverse Compton emission
        P_compt = compute_compt_power(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3)
        Pc_arr[iT] = P_compt

        # bremsstrahlung emission
        P_brems = compute_brems_power(r,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3)
        Pb_arr[iT] = P_brems
    
    # eliminate negative advected electron energy
    Qadve_arr[Qadve_arr <= 0.0] = 0.0

    # total electron heating and cooling
    heating = Qie_arr + (delta*Qplus_arr)
    cooling = Qadve_arr + Ps_arr + Pc_arr + Pb_arr

    # find the temperature for which heating balances cooling
    Te0 = Te0_testarr[np.argmin(np.abs(heating - cooling))]

    # print it out
    print('Electron temperature is '+str(np.round(Te0/(1.0e9),2))+' GK.')

    # check for extreme temperature values
    if ((np.argmin(np.abs(heating - cooling)) == 0) | (np.argmin(np.abs(heating - cooling)) == (len(Te0_testarr)-1))):
        print('WARNING: the self-consistently identified temperature is '+str(np.round((Te0/(1.0e9)),2))+' GK, which is on the boundary of the tested temperature range.')

    ##############################################
    # construct the spectrum

    # solve for Te power-law index
    t = (1.0 / np.log(rmax))*np.log((6.66e12)*beta*c3/(2.08*Te0*(1.0+beta)))

    # determine critical synchrotron frequencies and luminosities
    nu_p, L_p = compute_peak_freq(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3)
    nu_min, Lnu_min = compute_min_freq(nu_arr,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3)

    # synchrotron emission
    Lnu_synch = compute_synch_spectrum(nu,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3,nu_p,L_p,nu_min,Lnu_min)

    # inverse Compton emission
    Lnu_compt = compute_compt_spectrum(nu,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3,nu_p,L_p)

    # bremsstrahlung emission
    Lnu_brems = compute_brems_spectrum(nu,r,Te0,t,rmin,rmax,m,mdot,s,alpha,beta,c1,c3)

    # combine
    Lnu = Lnu_synch + Lnu_compt + Lnu_brems

    ##############################################
    
    if verbose_return:
        return Lnu, nu_p, Te0, Lnu_synch, Lnu_compt, Lnu_brems
    else:
        return Lnu, nu_p
