#####################################
# inputs

m = 6.5e9               # black hole mass, in solar masses
mdot = 1.0e-5           # Eddington rate

nu_min = 1.0e8          # minimum frequency, in Hz
nu_max = 1.0e22         # maximum frequency, in Hz

#####################################
# imports

import numpy as np
import matplotlib.pyplot as plt
from SEDmodel import SED

#####################################
# generate SED

nu = 10.0**np.linspace(np.log10(nu_min),np.log10(nu_max),1000)
Lnu, nu_p, Te0, Lnu_synch, Lnu_compt, Lnu_brems = SED(nu,m,mdot,verbose_return=True)

#####################################
# plot SED

fig = plt.figure(figsize=(4.25,4.25))
ax = fig.add_axes([0.1,0.1,0.8,0.8])

ax.plot(nu,nu*Lnu_synch,linestyle='-',color='C3',linewidth=3,alpha=0.5,label=r'$L_{\nu,\rm{synch}}$')
ax.plot(nu,nu*Lnu_compt,linestyle='-',color='C4',linewidth=3,alpha=0.5,label=r'$L_{\nu,\rm{compt}}$')
ax.plot(nu,nu*Lnu_brems,linestyle='-',color='C5',linewidth=3,alpha=0.5,label=r'$L_{\nu,\rm{brems}}$')

ax.plot(nu,nu*Lnu,'k-',linewidth=1.5)

ax.loglog()
ax.legend()

ax.set_xlabel(r'$\nu$ (Hz)')
ax.set_ylabel(r'$\nu L_{\nu}$ (erg s$^{-1}$)')

ax.set_xlim(1.0e8,1.0e22)
ax.set_ylim(1.0e35,1.0e43)

plt.savefig('example_SED.png',bbox_inches='tight',dpi=300)
plt.close()