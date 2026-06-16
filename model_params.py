"""Reference parameters and fixed forward-model settings for the shell-based
tidal-heating model.
"""

# Fixed forward-model settings (not calibrated).
# chi_v is the Benson & Du 2022 second-order velocity-anisotropy factor; its
# uncertainty is absorbed into f_2 (following Du+24), so it is held fixed here.
CHI_V = -0.333
# Strip on the subhalo's dynamical time at the tidal radius; accumulate the tidal
# tensor on the host orbital time (Du+24 eqs. 35, 38-39).
T_DYN_MODE = 'sub_lt'

# N-body reference parameter values: Du+24 Table IV, gamma=1 column
DU24_TABLE_IV = dict(alpha_s=3.93, eps_h=0.0741, beta_h=0.278, f_2=0.547, gamma_h=0.)
