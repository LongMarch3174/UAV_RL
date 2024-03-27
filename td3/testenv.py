import numpy as np
from scipy.spatial import distance


P_S = 10
P_E = 10
P_U = 1000
SIGMA2 = DELTA2 = 1e-26
OMEGA = 1e-3
EPSILON = 1
AIOT = 1

k = 1e14
rho = 1e-7

pu = (3927.15313621832, 1150.2640803086933, 73.44118335601918)
ps = (1500, 0, 0)
pd = (2000, 0, 0)
pe = (3000, 1000, 100)

dsu = distance.euclidean(pu, ps)
dsd = distance.euclidean(pd, ps)
dud = distance.euclidean(pu, pd)
dis = distance.euclidean(pu, pe)

hsu = OMEGA * (dsu ** -2)
hud = OMEGA * (dud ** -2)
hsd = OMEGA * (dsd ** -EPSILON) * AIOT
print(hsu, hsd, hud)

numerator = (np.abs(hsd + k * np.sqrt(rho) * hsu * hud) ** 2) * P_S
denominator = (1 + k ** 2 * hud ** 2) * SIGMA2
Gamma_SD = numerator / denominator

Gamma_SU = ((1 - rho) * hsu ** 2 * P_S) / SIGMA2

R_D = np.log2(1 + Gamma_SD)
R_U = np.log2(1 + Gamma_SU)

constraint = (abs(k) ** 2) * (rho * abs(hsu) ** 2 * P_S + SIGMA2)
if constraint <= P_U:
    print(True)
else:
    print(False)

print(Gamma_SD, Gamma_SU, R_D, R_U, dis, constraint, 2621293067932129, 3e15)

