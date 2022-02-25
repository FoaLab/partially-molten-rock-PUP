#!/usr/bin/env python
# coding: utf-8

# # Chapter 10 - Petrological thermodynamics of liquid and solid phases

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import numpy.matlib
from scipy.optimize import fsolve
from scipy.interpolate import interp1d

import warnings
warnings.filterwarnings('ignore')


# ## The equilibrium state
# 
# The partition coefficient $\equi{\parcod}_j$ is given by
# 
# $$
# \begin{equation*}
#   \label{eq:equilibrium_partitioning_nonlinear}
#   \parco_j = \left[\frac{\sum_k\left(\obar{\con}_k/\molarmass_k\right)\parco_k}
#              {\sum_k\left(\obar{\con}_k/\molarmass_k\right)}\right]
#              \exp\left[\frac{\Delta\enthalpy_j^{m}}{R/\molarmass_j}
#              \left(\frac{1}{\temp} -
#              \frac{1}{\melttemp_j}\right)\right].\nonumber
# \end{equation*}
# $$
# 
# In cases where, for example, we use a small set of fictitious *effective* components to approximate the full thermochemical system, the error that we make in taking $\phasemolarmass\liq/\phasemolarmass\sol \approx 1$ is acceptably small. With this approach, the partition coefficient $\equi{\parcod}_j$ becomes
# 
# $$
# \begin{equation*}
#   \label{eq:equilibrium_partitioning_simple}
#   \parco_j(P,T) = \exp\left[\frac{\latent_j}{R_j}\left(\frac{1}{\temp} - \frac{1}{\melttemp_j(P)}\right)\right],
# \end{equation*}
# $$
# 
# where $\latent_j\equiv-\Delta\enthalpy_j^m$ is the latent heat of melting for a solid composed of pure component $j$ and $R_j \equiv R/M_j$ is the modified gas constant.
# 
# Partition coefficients $\parco_j$ give an implicit equation for the equilibrium melt fraction $\equi\mf$ as
# 
# $$
# \begin{equation*}
#   \label{eq:equilibrium-melt-frac-implicit}
#   \sum_{j=1}^N\frac{\obar{\con}_j \left[1-\parco_j(\temp,\pres)\right]}{\equi\mf + (1-\equi\mf)\parco_j(\temp,\pres)} = 0,
# \end{equation*}
# $$
# 
# Once a value of $\equi\mf$ has been numerically obtained, it can be used to determine the phase composition
# 
# $$
# \begin{align}
#     \label{eq:equilib-phase-comp-liquid}
#     \equi\con\liq_j &= \frac{\obar{\con}_j}{\equi\mf +
#     (1-\equi\mf)\parco_j},\\
#     \label{eq:equilib-phase-comp-solid}
#     \equi\con\sol_j &= \frac{\obar{\con}_j}{\equi\mf/\parco_j +
#                       (1-\equi\mf)}.
# \end{align}
# $$
# 
# Table below shows parameter values for the two- and three-component ideal-solution systems. These values are useful for the demonstration of ideal-solution phase diagrams but should not be taken as an optimal calibration for the mantle.
# 
# | Name | $j$ | &nbsp; &nbsp; &nbsp; &nbsp; $\melttemp_j(P=0)$ [$^\circ$C] &nbsp; &nbsp; &nbsp; &nbsp; | $\clapeyron_j$ [GPa/K] | $\latent_j$ [kJ/kg] | $R_j$ [J/kg/K] |
# |:---:|:---:|:---:|:---:|:---:|:---:|
# | olivine | 1 | &nbsp; &nbsp; &nbsp; &nbsp; 1780 &nbsp; &nbsp; &nbsp; &nbsp; | 1/50 | 600 | 70 |
# | basalt | 2 | &nbsp; &nbsp; &nbsp; &nbsp; 950 &nbsp; &nbsp; &nbsp; &nbsp; | 1/100 | 450 | 30 |
# | hydrous basalt | 3 | &nbsp; &nbsp; &nbsp; &nbsp; 410 &nbsp; &nbsp; &nbsp; &nbsp; | 1/50 | 330 | 30 |

# ### Application to a two-pseudo-component system
# 
# The change of this melting temperature with pressure is given by the Clausius-Clapeyron equation. For simplicity, 
# we assume a constant value of $\clapeyron = \Delta\entropy/\Delta(1/\density)$, which gives
# 
# $$
# \begin{equation}
#   \label{eq:linear-solidus-relation}
#   \temp^m_j(P) = \left.\temp^m_j\right\vert_{\pres=0} +
#   \pres\clapeyron.
# \end{equation}
# $$
# 
# This produces the straight lines in Figure below (a) under the assumption that the thermodynamic pressure mis lithostatic, i.e., $\Grad\pres=\density\gravity$.

# In[2]:


class IdealSolutionParameters:
    
    def __init__(self, L, R, clap, Tm0):

        self.L = L        # J/kg
        self.R = R        # J/kg/K
        self.clap = clap  # GPa/K
        self.Tm0 = Tm0    # K


# In[3]:


class ThetaStructure:
    
    def __init__(self, f=0, cl=None, cs=None):

        self.Ts = 0.0
        self.Tf = 0.0
        self.f = f  # f the melt fraction
        self.cl = cl  # liquid concentration vector (length n)
        self.cs = cs  # solid concentration vector (length n)


# In[4]:


def ParCoef_IdealSolution(T, P, par):
    
    Tm = par.Tm0 + P/par.clap

    return np.exp(par.L / par.R * (1./T - 1./Tm))


# In[5]:


def EquilibriumResidual(TT, ff, PP, cc, par, parco_func):

    K = np.array(
        [parco_func(TT, PP, par_j) for par_j in par]
    ).flatten()

    R = np.array(
        [cc_j * (1. - K_) / (ff + (1. - ff) * K_) for cc_j, K_ in zip(cc, K)]
    )

    return R.sum()


# In[6]:


def EquilibriumState(n, cbar, T, P, par, parco_func):

    if n == 1:
        raise Exception('n must be greater than 1')

    if len(par) < n:
        if len(par) == 1:
            par = [par] * n
        else:
            raise Exception('par array of structures must be length 1 or n')

    if isinstance(T, np.ndarray) and isinstance(P, float):
        P = P * np.ones_like(T)
    if isinstance(P, np.ndarray) and isinstance(T, float):
        T = T * np.ones_like(P)
    if isinstance(T, float) and isinstance(P, float):
        r = cbar.shape[0]
        T = np.ones((r,)) * T
        P = np.ones((r,)) * P
    if len(T) != len(P):
        raise Exception(
            f'length(T)={len(T)} and length(P)={len(P)}'
            ' must be equal if both are greater than 1'
        )

    r = len(T)
    if cbar.shape != (r, n) and cbar.shape != (1, n) and cbar.shape != (n,):
        raise Exception('one dimension of the cbar array must have length n')
    if cbar.shape == (n, r):
        cbar = cbar.transpose()
    if cbar.shape == (1, r) or cbar.shape == (n,):
        cbar = np.matlib.repmat(cbar, r, 1)
    if cbar.shape != (r, n):
        raise Exception('cbar must have shape ({}, {})'.format(r, n))

    Theta = []

    for T_i, P_i, cbar_i in zip(T, P, cbar):
        Theta_i = ThetaStructure()
        Theta_i.Ts = fsolve(
            lambda T_: EquilibriumResidual(
                T_, 0.0, P_i, cbar_i, par, ParCoef_IdealSolution
            ), 
            T_i
        )[0]
        Theta_i.Tl = fsolve(
            lambda T_: EquilibriumResidual(
                T_, 1.0, P_i, cbar_i, par, ParCoef_IdealSolution
                ), 
            0.5*(T_i+Theta_i.Ts)
        )[0]

        if T_i < Theta_i.Ts:
            Theta_i.f = 0.0
        elif Theta_i.Tl < T_i:
            Theta_i.f = 1.0
        else:
            Theta_i.f = fsolve(
                lambda f_: EquilibriumResidual(
                    T_i, f_, P_i, cbar_i, par, ParCoef_IdealSolution
                ), 
                0.0
            )[0]

        K = np.array([parco_func(T_i, P_i, par_j) for par_j in par])

        Theta_i.cl = cbar_i / (Theta_i.f + (1. - Theta_i.f)*K)
        Theta_i.cs = K * Theta_i.cl
        Theta.append(Theta_i)

    if len(Theta) != len(T):
        raise Exception('inconsistent number of ThetaStructure')

    return Theta


# Figure 10.1 plots the ideal-solution model of a two-component system of _olivine_ ($j=1$) and _basalt_ ($j=2$). Parameter values are given in table above. __(a)__ Pure-component melting temperatures $\melttemp_j$ as a linear function of pressure. __(b)__ Partition coefficients $\parco_j$ as a function of temperature at a constant pressure of 1 GPa. __(c)__ Grey lines are the solidus and liquidus temperature as a function of the basalt fraction at a pressure of 1 GPa. The composition along the solidus curve refers to the solid phase; that along the liquidus curve refers to the liquid phase. Black lines indicate the compositional evolution of a closed system with bulk composition of 25% basalt component. __(d)__ Equilibrium melt fraction as a function of temperature for a bulk composition of 25% basalt component at 1 GPa.

# In[7]:


par = [
    IdealSolutionParameters(600e3, 70., 1./50., 1780. + 273.), 
    IdealSolutionParameters(450e3, 30., 1./100., 950. + 273.)
]

# thermodynamic state
Pref = 1.
Pvar = np.linspace(0., 7., 20)
Tref = 1350. + 273.
Tvar = np.linspace(1050., 1950., 200) + 273
cref = np.array([0.75, 0.25])
cvar = np.zeros((200, 2), dtype=float)
cvar[:, 0] = np.linspace(0., 1., 200)
cvar[:, 1] = 1.0 - np.linspace(0., 1., 200)

fig, ((axA, axB), (axC, axD)) = plt.subplots(
    nrows=2, ncols=2, figsize=(15.0, 18.0)
)

P  =  Pvar
T  =  Tref
c  =  cref
Tm = np.array(
    [par_i.Tm0 + P/par_i.clap for par_i in par]
).transpose()

axA.plot(
    P, Tm[:, 0] - 273., '-k', 
    linewidth=2, label=r'$T^\mathcal{S}_1$'
)
axA.plot(
    P, Tm[:, 1] - 273., '--k', 
    linewidth=2, label=r'$T^\mathcal{S}_2$'
)
axA.set_xlim(0.0, 7.0)
axA.set_xlabel('$P$, GPa', fontsize=20)
axA.set_ylim(900, 2000)
axA.set_ylabel(r'$T$, $^\circ$C', fontsize=20)
axA.text(
    0.01, 1990, r'(a)', fontsize=20, 
    verticalalignment='top', horizontalalignment='left'
)
axA.tick_params(axis='both', which='major', labelsize=15)
axA.legend(loc='lower right', fontsize=15)

P  =  Pref
T  =  Tvar
c  =  cref
K = np.array(
    [
        [
            ParCoef_IdealSolution(T_i, P, par_j) for par_j in par
        ] 
        for T_i in T
    ]
)

axB.plot(
    T - 273., K[:, 0], '-k', 
    linewidth=2, label=r'$\check{K}_1$'
)
axB.plot(
    T - 273., K[:, 1], '--k', 
    linewidth=2, label=r'$\check{K}_2$'
)
axB.set_yscale('log')
axB.set_xlim(1050, 1950)
axB.set_xticks((1200., 1400., 1600., 1800.))
axB.set_xticklabels((1200, 1400, 1600, 1800))
axB.set_xlabel(r'$T$, $^\circ$C', fontsize=20)
axB.set_ylim(0.01, 100)
axB.set_ylabel(r'${c}^s_j/{c}^\ell_j$', fontsize=20)
axB.text(
    1050, 95.0, r'(b)', fontsize=20, 
    verticalalignment='top', horizontalalignment='left'
)
axB.tick_params(axis='both', which='major', labelsize=15)
axB.legend(fontsize=15)

T = Tvar
P = Pref
c = cref
Theta = EquilibriumState(
    len(par), c, T, P, par, ParCoef_IdealSolution
)

axD.plot(
    Tvar - 273., 
    [t.f for t in Theta],'-k','linewidth',2
)
axD.set_xlim(1200., 1750.)
axD.set_xticks(
    (1200., 1300., 1400., 1500., 1600., 1700.)
)
axD.set_xticklabels(
    (1200, 1300, 1400, 1500, 1600, 1700)
)
axD.set_ylim(-0.02, 1.02)
axD.set_xlabel(r'$T$, $^\circ$C', fontsize=20)
axD.set_ylabel(r'$\check{f}$, wt. frac.', fontsize=20)
axD.tick_params(axis='both', which='major', labelsize=15)
axD.text(
    1200.0, 1.01, r'(d)', fontsize=20, 
    verticalalignment='top', horizontalalignment='left'
)

cl_hold = np.array([t.cl[1] for t in Theta])
cs_hold = np.array([t.cs[1] for t in Theta])
f_hold = np.array([t.f for t in Theta])
cl_hold[f_hold < 1e-6] = np.nan
cs_hold[1.-f_hold < 1e-6] = np.nan

c  = cvar
P  = Pref
T  = Tref
Theta2 = EquilibriumState(
    len(par), c, T, P, par, ParCoef_IdealSolution
)

axC.plot(
    c[:, 1], [t.Ts - 273. for t in Theta2], '-', 
    linewidth=4, color=[0.7, 0.7, 0.7]
)
axC.plot(
    c[:, 1], [t.Tl - 273. for t in Theta2], '-', 
    label='phase boundary', linewidth=4, color=[0.7, 0.7, 0.7]
)

axC.plot(
    cl_hold, Tvar - 273., '--k', 
    label='$\check{c}^\ell_2$', linewidth=2
)
axC.plot(
    cs_hold, Tvar - 273., '-k', 
    linewidth=2, label='$\check{c}^\ell_2$'
)
axC.set_xlim(0.0, 1.0)
axC.set_xlabel('${c}^i_2$, wt. frac.', fontsize=20)
axC.set_ylabel('$T$, $^\circ$C', fontsize=20)
axC.text(
    0.005, 1990, '$(c)$', fontsize=20, 
    verticalalignment='top', horizontalalignment='left'
)
axC.text(0.22, 1400, 'solid$+$liquid', fontsize=20)
axC.text(0.72, 1500,'liquid', fontsize=20)
axC.text(0.05, 1150,'solid', fontsize=20)
axC.tick_params(axis='both', which='major', labelsize=15)
axC.legend(fontsize=15)

fig.supxlabel("Figure 10.1", fontsize=20)

plt.show()


# ### Application to a three-pseudo-component system

# In[8]:


par3 = [
    IdealSolutionParameters(600e3, 70., 1./50., 1780. + 273.), 
    IdealSolutionParameters(450e3, 30., 1./100., 950. + 273.), 
    IdealSolutionParameters(330e3, 30., 1./50., 410. + 273.)
]


# In[9]:


Pref = 1.
Pvar = np.linspace(0.0, 7.0, 20)
Tref = 1350. + 273.
Tvar = np.linspace(1000., 1700. , 200) + 273.
cref = np.array([0.75, 0.248, 0.002])


# In[10]:


npts = 100
conc_water_in_pure_component = 1e-4/cref[2]


# Figures 10.2 below plot the ideal-solution model of a three-component system of "olivine" ($j=1$), "basalt" ($j=2$) and "hydrous basalt" ($j=3$). Parameter values are given in table above. __(a)__ Pure-component melting temperatures $\melttemp_j$ as a linear function of pressure. __(b)__ Partition coefficients $\parco_j$ as a function of temperature at a constant pressure of 1 GPa. __(d)__ Equilibrium melt fraction as a function of temperature for a bulk composition of 75 wt\% olivine component with and without 0.2 wt\% of hydrated basalt component at 1 GPa.

# In[11]:


fig, (axA, axB, axD) = plt.subplots(
    nrows=3, ncols=1, figsize=(6., 27.)
)

P  =  Pvar
T  =  Tref
c  =  cref
Tm = np.array(
    [par_i.Tm0 + P/par_i.clap for par_i in par3]
).transpose()

axA.plot(
    P, Tm[:, 0] - 273., '-k', 
    linewidth=2, label=r'$T^\mathcal{S}_1$'
)
axA.plot(
    P, Tm[:, 1] - 273., '--k', 
    linewidth=2, label=r'$T^\mathcal{S}_2$'
)
axA.plot(
    P, Tm[:, 2] - 273., '-.k', 
    linewidth=2, label=r'$T^\mathcal{S}_3$'
)
axA.set_xlim(0.0, 7.0)
axA.set_xlabel('$P$, GPa', fontsize=20)
axA.set_ylim(100., 2100.)
axA.set_ylabel(r'$T$, $^\circ$C', fontsize=20)
axA.set_yticks((500, 1000, 1500, 2000))
axA.text(
    0.01, 2080., r'(a)', fontsize=20, 
    verticalalignment='top', horizontalalignment='left'
)
axA.tick_params(axis='both', which='major', labelsize=13)
axA.legend(loc='lower right', fontsize=15, ncol=3)

P  =  Pref
T  =  Tvar
c  =  cref
K = np.array(
    [
        [
            ParCoef_IdealSolution(T_i, P, par_j) 
            for par_j in par3
        ] 
        for T_i in T
    ]
)

axB.plot(
    T - 273., K[:, 0], '-k', 
    linewidth=2, label=r'$\check{K}_1$'
)
axB.plot(
    T - 273., K[:, 1], '--k', 
    linewidth=2, label=r'$\check{K}_2$'
)
axB.plot(
    T - 273., K[:, 2], '-.k', 
    linewidth=2, label=r'$\check{K}_3$'
)
axB.set_yscale('log')
axB.set_xlim(1000., 1700.)
axB.set_xlabel(r'$T$, $^\circ$C', fontsize=20)
axB.set_ylim(1e-5, 100.)
axB.set_ylabel(r'${c}^s_j/{c}^\ell_j$', fontsize=20)
axB.set_yticks((1e-4, 1e-2, 1e0))
axB.text(
    1005., 90., r'(b)', fontsize=20, 
    verticalalignment='top', horizontalalignment='left'
)
axB.tick_params(axis='both', which='major', labelsize=13)
axB.legend(fontsize=15)

T = Tvar
P = Pref
cwet = cref
Theta_wet = EquilibriumState(
    len(par3), cwet, T, P, par3, ParCoef_IdealSolution
)

cdry = np.array([0.75, 0.25, 0.])
Theta_dry = EquilibriumState(
    len(par3), cdry, T, P, par3, ParCoef_IdealSolution
)

axD.plot(
    Tvar - 273., [t_wet.f for t_wet in Theta_wet], 
    '-k', linewidth=2, label='wet'
)
axD.plot(
    Tvar - 273., [t_dry.f for t_dry in Theta_dry], 
    '--k', linewidth=2, label='dry'
)
axD.set_xlabel('$T$, $^\circ$C', fontsize=20)
axD.set_ylabel('$\check{f}$, wt. frac.', fontsize=20)
axD.set_xlim(1000., 1500.)
axD.set_ylim(0., 0.4)
axD.set_yticks((0., 0.1, 0.2, 0.3, 0.4))
axD.text(
    1000., 0.396,'(d)', fontsize=20, 
    verticalalignment='top', horizontalalignment='left'
)
axD.tick_params(axis='both', which='major', labelsize=13)
axD.legend(fontsize=15, loc='lower right')

cl_hold = np.array([th.cl for th in Theta_wet])
cs_hold = np.array([th.cs for th in Theta_wet])
f_hold  = np.array([th.f for th in Theta_wet])
cl_hold[f_hold<1e-6] = np.nan
cs_hold[1-f_hold<1e-6] = np.nan

fig.supxlabel("Figure 10.2", fontsize=20)

plt.show()


# Figure 10.2c below plot the ideal-solution model of a three-component system of "olivine" ($j=1$), "basalt" ($j=2$) and "hydrous basalt" ($j=3$). Parameter values are given in table above. __(c)__ Surfaces are the solidus and liquidus temperature through the full, 3-component space at a pressure of 1 GPa. The composition along the solidus surface refers to the solid phase; that along the liquidus surface refers to the liquid phase. The compositional evolution of the melt for a closed system with 75 wt\% olivine and 0.2 wt\% hydrous basalt is shown by the solid line.  It starts at the triangle and progresses toward the circle with increasing $\temp$ and $\equi\mf$.

# In[12]:


C3 = np.linspace(0., 1., npts)
C2 = np.linspace(0., 1., npts)
[C2,C3] = np.meshgrid(C2,C3)
C2 = np.reshape(C2, npts*npts)
C3 = np.reshape(C3, npts*npts)
C1 = 1-C2-C3
C = np.zeros((npts*npts, 3), dtype=float)
C[:, 0] = C1
C[:, 1] = C2
C[:, 2] = C3
P = Pref
T = Tref
Theta = EquilibriumState(
    len(par3), C, T, P, par3, ParCoef_IdealSolution
);
Tsol = np.array([th.Ts for th in Theta])
Tliq = np.array([th.Tl for th in Theta])
Tsol[C[:,1]+C[:,2]>=1] = np.nan
Tliq[C[:,1]+C[:,2]>=1] = np.nan


# In[13]:


def terncoords(c1,c2,c3):

    csum = c1 + c2 + c3
    c1  = c1/csum
    c2 = c2/csum
    c3 = c3/csum
    y = c2 * np.sin(np.pi/3.)
    x = c1 + y / np.tan(np.pi/3.)

    return x, y


# In[14]:


X, Y = terncoords(C2, C3, C1)
triangXY = mtri.Triangulation(X, Y)


# In[15]:


fig, axC = plt.subplots(
    figsize=(18., 9.), 
    subplot_kw={"projection": "3d"}
)

axC.plot_trisurf(triangXY, Tsol)
surf = axC.plot_trisurf(triangXY, Tliq)
axC.plot(
    [0., 1., 0.5, 0.], 
    [0., 0., np.sin(np.pi/3.), 0.], color=[0,0,0]
)
x, y = terncoords(cl_hold[:,1], cl_hold[:,2], cl_hold[:,0])
axC.plot(x, y, zs=0, zdir='z', linewidth=4)

m = 5
grids = np.linspace(0, 100, m+1)
grids = grids[:-1]
x3, y3 = terncoords(100-grids, grids, np.zeros_like(grids))
x2, y2 = terncoords(grids, np.zeros_like(grids), 100-grids)
x1, y1 = terncoords(np.zeros_like(grids), 100-grids, grids)
n = m-1
for i in np.arange(1, m):
    axC.plot(
        [x1[i], x2[n-i+1]], [y1[i], y2[n-i+1]], 
        'k:', zs=0, zdir='z', linewidth=1
    )
    axC.plot(
        [x2[i], x3[n-i+1]], [y2[i], y3[n-i+1]], 
        'k:', zs=0, zdir='z', linewidth=1
    )
    axC.plot(
        [x3[i], x1[n-i+1]], [y3[i], y1[n-i+1]], 
        'k:', zs=0, zdir='z', linewidth=1
    )

axC.text(0.44, 0.9, z=0., s='${c}_3^i$', fontsize=20)
axC.text(1.15, 0.0, z=0., s='${c}_2^i$', fontsize=20)
axC.text(-0.13,-0.030, z=0., s='${c}_1^i$', fontsize=20)
    
labels = [str(int(g)) for g in grids[1:]]
for x3_, y3_, l_ in zip(x3[1:], y3[1:], labels):
    axC.text(x3_ + 0.09, y3_ - 0.04, z=0., s=l_, fontsize=20)
for x2_, y2_, l_ in zip(x2[1:], y2[1:], labels):
    axC.text(x2_ + 0.04, y2_ - 0.12, z=0., s=l_, fontsize=20)
for x1_, y1_, l_ in zip(x1[1:], y1[1:], labels):
    axC.text(x1_ - 0.09, y1_ + 0.02, z=0., s=l_, fontsize=20)
    
axC.set_xticks(())
axC.set_yticks(())
axC.set_zticks(())

fig.supxlabel("Figure 10.2c", fontsize=20)

ticks_ = (0., 0.2, 0.4, 0.6, 0.8, 1.)
cbar = fig.colorbar(surf, aspect=5, ticks=(ticks_))
cticks_ = (
    int(
        (np.nanmax(Tliq) - np.nanmin(Tliq))*s 
        + np.nanmin(Tliq)
    ) for s in ticks_
)
cbar.ax.set_yticklabels((cticks_));


# ### Approaching the eutectic phase diagram
# 
# Figure 10.3 plots a eutectic phase diagram compared with ideal-solution phase loops for different values of $R_2$. Other ideal-solution parameters are $R_1=60$ J/kg/K, $\latent_1=\latent_2=500$ kJ/kg. __(a)__ Solidus and liquidus curves computed with ideal solution theory compared with the eutectic solidus and liquidus. __(b)__ Isobaric melting curves computed based on the phase loops and the eutectic.

# In[16]:


def liquidusQuadratic(H, C, L, cp, T0, Te):

    a = L
    b = cp*(T0-Te)-H
    c = -C*cp*(T0-Te)

    return (-b + np.sqrt(b*b - 4*a*c))/2/a


# In[17]:


def equilibriumEutectic(H, C, T0, Te, L, cp):

    He_low = 0
    He_high = L*C[1]
    f = np.array([
        0 if Hj <= He_low else 
        Hj/L if Hj <= He_high else 
        np.minimum(
            liquidusQuadratic(
                Hj, C[1], L, cp, T0, Te
            ), 
            1
        ) for Hj in H
    ])
    T = Te + (H - L * f)/cp
    
    return T, f


# In[18]:


def pseudoEutecticSolidus(C, T0, Te, gamma):
    
    return T0 + (Te - T0) * (
        1. - np.exp(-C/gamma)
    )/(
        1.-np.exp(-1/gamma)
    )


# In[19]:


def pseudoEutecticLiquidus(C, T0, Te, gamma):

    return T0 + (Te - T0) * C


# In[20]:


def equilibriumPseudoEutectic(T, C, T0, Te, L, gamma):
    
    pES = pseudoEutecticSolidus(C, T0, Te, gamma)
    pEL = pseudoEutecticLiquidus(C, T0, Te, gamma)
    cl = np.array([
        np.nan if Tj < pES else
        C if Tj > pEL else
        (Tj-T0)/(Te-T0) for Tj in T
    ])
    cs = np.array([
        C if Tj < pES else
        np.nan if Tj > pEL else
        -gamma*np.log(
            1-cl_j*(1-np.exp(-1/gamma))
        ) for Tj, cl_j in zip(T, cl)
    ])
    f = np.array([
        0.0 if Tj < pES else
        1.0 if Tj > pEL else
        (
            C - cs_j
        )/(
            cl_j - cs_j
        ) for Tj, cl_j, cs_j in zip(T, cl, cs)
    ])

    return f


# In[21]:


fig, ((axA, axB)) = plt.subplots(
    nrows=1, ncols=2, figsize=(12., 9.)
)

npts =  500
Tref = 1350
Tvar = np.linspace(950, 1900, npts)
cref = np.array([0.75, 0.25])
Pref = 1
P    = np.linspace(0.0, 7.0, 300)
T0   =  2000
Te   =  1000
L    =  500e3

gamma_arr = np.flip(np.logspace(-2, -1, 3))
plotstyle = [':k', '-.k', '--k']
cp = 1200
H = np.linspace(-L, L + cp*(T0-Te), 1000)

T = np.linspace(T0, Te)
X = (T-Te)/(T0-Te)
C = 1 - X

axA.plot(C, T,'-k', linewidth=2)
axA.plot([0, 0], [Te, T0],'-k', linewidth=2)
for gamma, pltstyle in zip(gamma_arr, plotstyle):
    Ts = pseudoEutecticSolidus(C, T0, Te, gamma)
    axA.plot(C, Ts, pltstyle, linewidth=2)

axA.set_ylabel('$T$, $^\circ$C', fontsize=20)
axA.set_xlim((0, 1))
axA.set_xlabel('${c}^i_2/c^E_2$', fontsize=20)
axA.set_ylim((900, 2000))
axA.text(
    0.9, 1990.,'(a)', fontsize=20, 
    verticalalignment='top', horizontalalignment='left'
)
axA.tick_params(axis='both', which='major', labelsize=13)

TE, fE = equilibriumEutectic(H, cref, T0, Te, L, cp)
T = np.sort(np.hstack((Tvar, Te)))
plots = []
labels = []
for gamma, pltstyle in zip(gamma_arr, plotstyle):
    theta_f = equilibriumPseudoEutectic(
        T, cref[1], T0, Te, L, gamma
    )
    plots.append(
        axB.plot(
            T, theta_f, pltstyle, linewidth=2
        )
    )
    labels.append('$\gamma=${:.3f}'.format(gamma))
plots.append(axB.plot(TE, fE, '-k', linewidth=2))
labels.append('eutectic')

axB.set_xlim([950, 1900])
axB.set_ylim([0, 1])
axB.set_xlabel('$T$, $^\circ$C', fontsize=20)
axB.set_ylabel('$\check{f}$, wt. frac.', fontsize=20)
axB.text(
    1800, 0.99, '(b)', fontsize=20, 
    verticalalignment='top', horizontalalignment='left'
)
axB.tick_params(axis='both', which='major', labelsize=13)

handles = [p[0] for p in plots]
plt.legend(
    handles=handles, fontsize=15, labels=labels,
    bbox_to_anchor=(-1.0, 1.02, 2., .2),  loc='lower left',
    ncol=4, mode="expand", borderaxespad=0.
)

fig.supxlabel("Figure 10.3", fontsize=20)

plt.show()


# ### Linearising the two-component phase diagram
# 
# Phase diagrams constructed with ideal solution theory provide a useful basis for developing the melting relations required for magma/mantle dynamics. Their nonlinearity, however, precludes a use in analytical calculations; instead they must be handled numerically. It is therefore important to formulate an approximation in which the equilibrium state is computed analytically.
# 
# For linear variations in pressure and composition, we can write
# 
# $$
# \begin{align}
#   \label{eq:col-twocomp-pd-solidus}
#   \soltemp &= \soltemp_\text{ref} + (\pres-\pres_\text{ref})/\clapeyron  +
#               \solslope\left(\equi\con\sol - \equi\con\sol_{\text{ref}}\right),\\
#   \label{eq:col-twocomp-pd-deltac}
#   \equi\con\liq &= \equi\con\sol - \Delta\equi\con,
# \end{align}
# $$
# 
# where $\solslope$ is the constant slope of the solidus with concentration in the two component space and $\soltemp_\text{ref}$ is a reference temperature at $\pres=\pres_\text{ref}$ and $\equi\con\sol=\equi\con\sol_\text{ref}$.  Also, $\Delta\equi\con \equiv \equi\con\sol - \equi\con\liq$ is the concentration difference between the solidus and the liquidus.  If $\Delta\equi\con$ is taken to be a constant then the liquidus slope $\liqslope$ and the solidus slope $\solslope$ are equal.
# 
# Figure 10.4 below plots a comparison of a two-component, ideal-solution phase loop with a linearised solidus and liquidus at 1 GPa. The phase loop uses the parameters from the Table above. __(a)__ Solidus and liquidus curves.  Evolution of an initially unmolten rock with 25 wt% basaltic component for increasing temperature, along the linearised phase boundaries. __(b)__ Isobaric melting curves.

# In[22]:


def SolLiqTemperature(C, Cref, M, Tref):
    
    return Tref + M*(C-Cref)


# In[23]:


def SolLiqComposition(T, Tref, M, Cref):
    
    return Cref + (T - Tref)/M


# In[24]:


def MeltFraction(T, C, ML, MS, LCref, SCref, Tref):

    TS = SolLiqTemperature(C, SCref, MS, Tref)
    TL = SolLiqTemperature(C, LCref, ML, Tref)
    Cs = SolLiqComposition(T, Tref, MS, SCref)
    Cl = SolLiqComposition(T, Tref, ML, LCref)
    F = (C - Cs)/(Cl - Cs)
    F[T <= TS] = 0.
    F[T >= TL] = 1.
    Cs[T < TS] = C
    Cl[T < TS] = np.nan

    return F, Cs, Cl


# In[25]:


par_lin = [
    IdealSolutionParameters(600e3, 70., 1./45., 1780. + 273.), 
    IdealSolutionParameters(450e3, 30., 1./112., 950. + 273.)
]


# In[26]:


Pref = 1.
Pvar = np.linspace(0., 7., 20)
Tref = 1350. + 273.
Tvar = np.linspace(900., 1350., 300) + 273.
cref = np.array([0.75, 0.25])
cvar = np.zeros((300, 2), dtype=float)
cvar[:, 0] = np.linspace(0., 1., 300)
cvar[:, 1] = 1.0 - np.linspace(0., 1., 300)


# In[27]:


fig, ((axA, axB)) = plt.subplots(
    nrows=1, ncols=2, figsize=(12.0, 9.0)
)

c = cvar
T = Tref
P = Pref
Theta = EquilibriumState(
    len(par_lin), c, T, P, par_lin, ParCoef_IdealSolution
)

Cint = cref[1]*np.array([0.999, 1.001])
f_Tint = interp1d(
    c[:,1], np.array([th.Ts for th in Theta]) - 273.
)
Tint = f_Tint(Cint)
MS = (Tint[1]-Tint[0])/(Cint[1]-Cint[0])
Tref = np.mean(Tint)
f_Cint = interp1d(
    np.array([th.Tl for th in Theta]) - 273, c[:,1]
)
Cint = f_Cint(Tint)
ML = (Tint[1]-Tint[0])/(Cint[1]-Cint[0])
Cref = np.mean(Cint)
F, Cs, Cl = MeltFraction(
    Tvar - 273., cref[1], ML, MS, Cref, cref[1], Tref
)

plots = []
axA.plot(
    c[:,1], np.array([th.Ts for th in Theta]) - 273., 
    '-', linewidth=3, color=[0.6, 0.6, 0.6]
)
plots.append(
    axA.plot(
        c[:,1], np.array([th.Tl for th in Theta]) - 273., '-', 
        linewidth=3, color=[0.6, 0.6, 0.6]
    )
)
plots.append(
    axA.plot(
        c[:,1], SolLiqTemperature(c[:,1], Cref, ML, Tref), 
        '--', color=[0.6, 0.6, 0.6], linewidth=3
    )
)
plots.append(
    axA.plot(
        c[:,1], SolLiqTemperature(c[:,1], cref[1], MS, Tref),
        '--', color=[0.6, 0.6, 0.6], linewidth=3
    )
)
axA.plot(Cs, Tvar-273, '-k', linewidth=1.5)
axA.plot(Cl, Tvar-273, '-k', linewidth=1.5)

axA.set_ylabel('$T$, $^\circ$C', fontsize=20)
axA.set_xlabel('${c}^i_2$, wt. frac.', fontsize=20)
axA.set_xlim([0, 1])
axA.set_ylim([1000, 2000])
axA.tick_params(axis='both', which='major', labelsize=13)
axA.text(0.001, 1965., '(a)', fontsize=18)
labels=[
    '$T^\mathcal{S}, T^\mathcal{L}$ (ideal soln.)', 
    '$T^\mathcal{S},T^\mathcal{L}$ (linear)',
    '$\check{c}^s_2(T),\check{c}^\ell_2(T)$, (linear)'
]
axA.legend(
    handles=([p[0] for p in plots]), 
    fontsize=15, labels=labels
)

T = Tvar
P = Pref
c = cref
Theta = EquilibriumState(
    len(par_lin), c, T, P, par_lin, ParCoef_IdealSolution
)
axB.plot(
    T - 273., np.array([th.f for th in Theta]), 
    '-k', linewidth=2, label='ideal soln.'
)
axB.plot(T - 273., F,'--k', linewidth=2, label='linear')
axB.set_xlabel('$T$, $^\circ$C', fontsize=20)
axB.set_xlim([1200, 1350])
axB.set_xticks((1200, 1250, 1300, 1350))
axB.set_ylabel('$\check{f}$, wt. frac.', fontsize=20)
axB.set_yticks((0., 0.1, 0.2, 0.3, 0.4))
axB.text(1200.01, 0.385, '(b)', fontsize=20)
axB.tick_params(axis='both', which='major', labelsize=13)
axB.legend(fontsize=15)

fig.supxlabel("Figure 10.4", fontsize=20)

plt.show()

