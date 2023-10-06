"""Test program"""
import matplotlib.pyplot as plt
from tqdm.auto import trange

from chain import (
    random_distribution,
    hamiltonian,
    np,
    transmittance,
    sigma_lead,
    green_function,
    CI,
    c_t,
    write_input,
)

N_A = 80
N_PH = 0
DIST_KIND = 1  # Gaussian
NE_POINTS = 200
NUM_DISORDER = 1000
T_CH = 1.0
GAMMA = 0.0
DELTA_U = 0.5
OMEGA = 2.0
T_CS = 1.0
T_CD = 1.0
T_LS = 1.0
T_LD = 1.0
MU_S = 0.0
MU_D = 0.0


ARRAY_U_ONSITE = random_distribution(
    n_a=N_A,
    delta_u=DELTA_U,
    num_disorder=NUM_DISORDER,
    dist_kind=DIST_KIND,
)


return_array = np.zeros(
    shape=(3, NE_POINTS),
    dtype=float,
)
return_array[0, :] = np.linspace(
    start=-2.0,
    stop=2.0,
    num=NE_POINTS + 2,
)[1:-1]

transmittances = np.zeros(
    shape=(NUM_DISORDER, NE_POINTS),
    # dtype=complex,
)

for i in trange(NUM_DISORDER, desc="Desordem"):
    H = hamiltonian(
        u_onsite=ARRAY_U_ONSITE[i],
        n_a=N_A,
        n_ph=N_PH,
        t_ch=T_CH,
        gamma=GAMMA,
        omega=OMEGA,
    )

    # for ne in trange(NE_POINTS, desc="Transmitância"):
    for ne in range(NE_POINTS):
        sigma_s = sigma_lead(
            return_array[0, ne],
            "s",
            N_A,
            N_PH,
            T_CS,
            T_CD,
            T_LS,
            T_LD,
            MU_S,
            MU_D,
        )
        sigma_d = sigma_lead(
            return_array[0, ne],
            "d",
            N_A,
            N_PH,
            T_CS,
            T_CD,
            T_LS,
            T_LD,
            MU_S,
            MU_D,
        )
        sigma_l = sigma_s + sigma_d

        green_f = green_function(return_array[0, ne], H, sigma_l)

        gamma_s = CI * (sigma_s - c_t(sigma_s))
        gamma_d = CI * (sigma_d - c_t(sigma_d))

        transmittances[i, ne] = transmittance(green_f, gamma_s, gamma_d)

return_array[1, :] = np.mean(
    a=transmittances,
    axis=0,
)
return_array[2, :] = np.std(
    a=transmittances,
    axis=0,
    # ddof=1,
)

write_input(
    return_array,
    N_A,
    N_PH,
    DIST_KIND,
    NE_POINTS,
    NUM_DISORDER,
    T_CH,
    GAMMA,
    DELTA_U,
    OMEGA,
    T_CS,
    T_CD,
    T_LS,
    T_LD,
    MU_S,
    MU_D,
)

fig_1, ax_1 = plt.subplots()

ax_1.grid(visible=True)

ax_1.set_xlabel("E")
ax_1.set_xlim(-2, 2)

ax_1.set_ylabel("Transmitância")
ax_1.set_ylim(1e-12, 1.0)
ax_1.set_yscale("log")

ax_1.plot(
    return_array[0, :],
    return_array[1, :],
)

fig_1.savefig("transmitancia.png")

# plt.show()
