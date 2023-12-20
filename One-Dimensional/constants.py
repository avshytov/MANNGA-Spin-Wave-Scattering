import numpy as np

pi = np.pi

meter   = 1e+6  # distance in um
Ampere  = 1.0   # current in amps
sec = 1e+9  # time in ns
kg  = 1e+18 # mass in attograms

Joule  = kg * meter**2 / sec**2
Newton = kg * meter / sec**2
Hz = 1.0 / sec
kHz = 1e+3 * Hz
MHz = 1e+6 * Hz
GHz = 1e+9 * Hz
THz = 1e+12 * Hz

GHz_2pi = GHz * 2.0 * pi

Coulomb = Ampere * sec
Volt    = Joule / Coulomb
V_m     = Volt  / meter
Farad   = Coulomb / Volt

A_m   = Ampere / meter
Tesla = Newton / Ampere / meter
Weber = Tesla * meter**2
Henry = Weber / Ampere

m_s  = meter / sec
m_s2 = meter / sec**2


mm = 1e-3 * meter
um = 1e-6 * meter
nm = 1e-9 * meter
pm = 1e-12 * meter
fm = 1e-12 * meter

rad_um = 1.0 / um

kA = 1e+3 * Ampere
MA = 1e+6 * Ampere
mA = 1e-3 * Ampere

kA_m = 1e+3 * A_m
MA_m = 1e+3 * A_m
Oe = kA_m / 4.0 / pi
mT = 1e-3 * Tesla

mC = 1e-3 * Coulomb
uC = 1e-6 * Coulomb
nC = 1e-9 * Coulomb
pC = 1e-12 * Coulomb

mF = 1e-3  * Farad
uF = 1e-6  * Farad
nF = 1e-9  * Farad
pF = 1e-12 * Farad

mH = 1e-3  * Henry
uH = 1e-6  * Henry
nH = 1e-9  * Henry
pH = 1e-12 * Henry

ms = 1e-3 * sec
us = 1e-6 * sec
ns = 1e-9 * sec
ps = 1e-12 * sec

mJ = 1e-3  * Joule
uJ = 1e-6  * Joule
nJ = 1e-9  * Joule
pJ = 1e-12 * Joule
fJ = 1e-15 * Joule
aJ = 1e-18 * Joule

pJ_m = pJ / meter # unit for exchange coupling

c_light = 299792458 * m_s
h_no_bar = 6.62607015e-34 * Joule * sec
hbar = h_no_bar / (2.0 * pi)

q_e   = 1.602176634e-19 * Coulomb
eps_0 = 8.85418782e-12 * Farad / meter
mu_0  = 4.0 * np.pi * 1e-7 * Henry / meter
m_e = 0.911e-30 * kg 


gamma_L = q_e  / 2.0 / m_e   # Larmor's ratio 
mu_B = gamma_L * hbar        # Bohr's magneton
g_s = 2.0             
gamma_s = gamma_L * g_s      # gyromagnetic ratio for spin

gamma_mu = gamma_s * mu_0

print ("eps_0 = ", eps_0, "mu_0 = ", mu_0)
print ("light:    ", 1.0 / np.sqrt(eps_0 * mu_0) / m_s)
print ("expected: ", c_light / m_s)
print ("gamma_s = ", gamma_s / GHz_2pi)

print ("1000 kA/m = ", 1000 * kA_m)
print ("mT = ", mT)

