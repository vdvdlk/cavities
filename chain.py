"""Build the hamiltonian matrix and calculate the transmittance
of a 'single electron + cavity' system.
"""
import sys
import numpy as np
from scipy.special import factorial, perm
from tqdm.auto import trange

# np.set_printoptions(
#     precision=1,
# )

SEED = 1892
RNG = np.random.default_rng(
    seed=1892,
)
CI = complex(
    real=0,
    imag=1,
)
# PI = np.pi

N_A = 80  # Number of sites
N_PH = 0  # Number of photons
DIST_KIND = 1  # = 1 -> Gaussian, =/= 1 -> Uniform
NE_POINTS = 200  # Energy grid
NUM_DISORDER = 100  # Number of disorder configurations
T_CH = 1.0  # Hopping inside chain
GAMMA = 0.0  # Coupling constant
DELTA_U = 0.5
OMEGA = 2.0  # Photon frequency
T_CS = 1.0  # Hopping between source and channel
T_CD = 1.0  # Hopping between drain and channel
T_LS = 1.0  # Hopping inside source
T_LD = 1.0  # Hopping inside drain
MU_S = 0.0  # Source chemical potential
MU_D = 0.0  # Drain chemical potential


def random_distribution(
    n_a: int = N_A,
    delta_u: float = DELTA_U,
    num_disorder: int = NUM_DISORDER,
    dist_kind: int = DIST_KIND,
):
    """Random distribution of the on-site energies."""
    if dist_kind == 1:
        array_u_onsite = RNG.normal(
            scale=delta_u,
            size=(num_disorder, n_a),
        )
    else:
        array_u_onsite = RNG.uniform(
            low=-0.5 * delta_u,
            high=0.5 * delta_u,
            size=(num_disorder, n_a),
        )
    return array_u_onsite


def p_func(
    j: int,
    m_ph: int,
) -> float:
    """Auxiliary function P(j, M)."""
    p_squared = perm(
        N=m_ph,
        k=j,
        # exact=True,
    )
    return np.sqrt(p_squared)


def h_nm(
    n_ph: int,
    n_row: int,
    m_col: int,
    g_param: float,
) -> complex:
    """Auxiliary function h_NM(g)."""
    delta = np.eye(n_ph + 1)
    accum_sum = complex(0)
    for i, j in np.ndindex(n_row + 1, m_col + 1):
        accum_prod = complex(1)

        accum_prod *= delta[n_row - i, m_col - j]
        accum_prod *= (CI * g_param) ** i / factorial(i)
        accum_prod *= (CI * g_param) ** j / factorial(j)
        accum_prod *= p_func(j, m_col) * p_func(i, n_row)

        accum_sum += accum_prod

    return np.exp(-(g_param**2) / 2) * accum_sum


def field_block(
    n_a: int,
    n_diag: int,
    omega: float,
):
    """Build the field block hamiltonian."""
    id_matrix = np.identity(
        n=n_a,
        dtype=complex,
    )
    h_field = id_matrix * n_diag * omega
    return h_field


def chain_diag_block(
    u_onsite: np.ndarray,
):
    """Build the chain diagonal block hamiltonian."""
    h_chain = np.array(
        object=np.diagflat(u_onsite),
        dtype=complex,
    )
    return h_chain


def chain_offdiag_block(
    n_a: int,
    n_ph: int,
    n_row: int,
    m_col: int,
    t_ch: float,
    gamma: float,
):
    """Build the chain off-diagonal block hamiltonian."""
    g_param = gamma / t_ch

    array_h_nm = np.ones(n_a - 1, dtype=complex) * h_nm(
        n_ph,
        n_row,
        m_col,
        g_param,
    )

    h_chain = t_ch * np.diagflat(
        v=array_h_nm,
        k=1,
    )

    h_chain += t_ch * np.diagflat(
        v=np.conj(array_h_nm),
        k=-1,
    )

    return h_chain


def block_over(
    block_matrix: np.ndarray,
    n_ph: int,
    ph_row: int,
    ph_col: int,
) -> np.ndarray:
    """Build the blocked matrix"""
    delta = np.identity(
        n=n_ph + 1,
        dtype=complex,
    )

    full_matrix = np.outer(a=delta[ph_row, :], b=delta[ph_col, :])

    return np.kron(full_matrix, block_matrix)


def hamiltonian(
    u_onsite: np.ndarray,  # On-site energies
    n_a: int = N_A,
    n_ph: int = N_PH,
    t_ch: float = T_CH,
    gamma: float = GAMMA,
    omega: float = OMEGA,
):
    """Build the chain hamiltonian."""
    n_n = n_a * (n_ph + 1)
    h_matrix = np.zeros(shape=(n_n, n_n), dtype=complex)

    # Diagonal subblocks
    for n_diag in range(n_ph + 1):
        h_block = np.zeros(shape=(n_a, n_a), dtype=complex)
        h_block += field_block(n_a, n_diag, omega)
        h_block += chain_diag_block(u_onsite)

        h_matrix += block_over(h_block, n_ph, n_diag, n_diag)

    # Off-diagonal subblocks
    for n_row, m_col in np.ndindex(n_ph + 1, n_ph + 1):
        h_block = np.zeros(shape=(n_a, n_a), dtype=complex)
        h_block += chain_offdiag_block(
            n_a,
            n_ph,
            n_row,
            m_col,
            t_ch,
            gamma,
        )

        h_matrix += block_over(h_block, n_ph, n_row, m_col)

    return h_matrix


def f_func(arg: float) -> complex:
    """Auxiliary function f(x)"""
    return 0.5 * (arg - CI * np.sqrt(4 - arg**2))


def sigma_lead(
    energy: float,
    lead: str,
    n_a: int = N_A,
    n_ph: int = N_PH,
    t_cs: float = T_CS,
    t_cd: float = T_CD,
    t_ls: float = T_LS,
    t_ld: float = T_LD,
    mu_s: float = MU_S,
    mu_d: float = MU_D,
):
    """Build the retarded drain self-energy matrix."""
    block_matrix = np.zeros(
        shape=(n_a, n_a),
        dtype=complex,
    )

    if lead == "s":
        block_matrix[0, 0] = t_cs**2 / t_ls * f_func((energy - mu_s) / t_ls)
    elif lead == "d":
        block_matrix[-1, -1] = t_cd**2 / t_ld * f_func((energy - mu_d) / t_ld)

    n_n = n_a * (n_ph + 1)
    matrix = np.zeros(shape=(n_n, n_n), dtype=complex)
    for n_row, m_col in [(0, 0)]:
        matrix += block_over(block_matrix, n_ph, n_row, m_col)

    return matrix


def green_function(
    energy: float,
    h_matrix: np.ndarray,  # Hamiltonian matrix
    sigma_l: np.ndarray,  # Self-energy matrix
) -> np.ndarray:
    """Build the retarded Green's function of the chain."""
    identity = np.identity(
        n=h_matrix.shape[0],
        dtype=complex,
    )

    return np.linalg.inv(energy * identity - h_matrix - sigma_l)


def c_t(matrix: np.ndarray) -> np.ndarray:
    """Returns the conjugate transpose of a matrix."""
    return np.conj(np.transpose(matrix))


def transmittance(
    green_ret: np.ndarray,
    gamma_s: np.ndarray,
    gamma_d: np.ndarray,
) -> float:
    """Calculate the transmittance for the given self-energies and Green's function."""
    green_adv = c_t(green_ret)
    return np.real(np.trace(gamma_s @ green_ret @ gamma_d @ green_adv))


def write_input(
    return_array: np.ndarray,
    n_a: int = N_A,
    n_ph: int = N_PH,
    dist_kind: int = DIST_KIND,
    ne_points: int = NE_POINTS,
    num_disorder: int = NUM_DISORDER,
    t_ch: float = T_CH,
    gamma: float = GAMMA,
    delta_u: float = DELTA_U,
    omega: float = OMEGA,
    t_cs: float = T_CS,
    t_cd: float = T_CD,
    t_ls: float = T_LS,
    t_ld: float = T_LD,
    mu_s: float = MU_S,
    mu_d: float = MU_D,
):
    """Write input."""

    filename = "L" + str(n_a) + "Nph" + str(n_ph) + ".txt"
    # original_stdout = sys.stdout
    with open(file=filename, mode="w", encoding="utf-8") as f:
        sys.stdout = f

        print("Input data")
        print("Lx=", n_a, "Nph=", n_ph, "seed=", SEED)
        print("Distribution (1 Gaussian, any other retangular)=", dist_kind)
        print("Energy grid=", ne_points,
              "Number of disorder conf.=", num_disorder)
        print("t=", t_ch, "gam=", gamma,
              "sigma (dist)=", delta_u, "Omega=", omega)
        print("tcS=", t_cs, "tcD=", t_cd, "tlS=", t_ls, "tlD=", t_ld)
        print("muD=", mu_s, "muD=", mu_d)

    np.savetxt(
        fname="dados" + filename,
        X=return_array.T,
        # delimiter="        ",
    )


def main():
    """Main function."""
    # On-site energies for multiple disorder configs
    array_u_onsite = random_distribution()

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
        h_matrix = hamiltonian(array_u_onsite[i])

        # for ne in trange(NE_POINTS, desc="Transmitância"):
        for ne in range(NE_POINTS):
            sigma_s = sigma_lead(return_array[0, ne], "s")
            sigma_d = sigma_lead(return_array[0, ne], "d")
            sigma_l = sigma_s + sigma_d

            green_f = green_function(return_array[0, ne], h_matrix, sigma_l)

            gamma_s = CI * (sigma_s - c_t(sigma_s))
            gamma_d = CI * (sigma_d - c_t(sigma_d))

            transmittances[i, ne] = transmittance(green_f, gamma_s, gamma_d)

    return_array[1, :] = np.mean(
        a=transmittances,
        axis=0,
    )
    return_array[2, :] = np.nanstd(
        a=transmittances,
        axis=0,
        ddof=1,
    )

    write_input(return_array)


if __name__ == "__main__":
    main()
