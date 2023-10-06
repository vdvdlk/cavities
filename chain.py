"""Build and diagonalize the hamiltonian matrix
of a 'single electron + cavity' system.
"""
import sys
import numpy as np
from scipy.special import factorial, perm

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
PI = np.pi


def random_distribution(
    n_a: int,
    delta_u: float,
    num_disorder: int,
    dist_kind: int,
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
):
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
        accum_prod *= (CI * g_param) ** i / factorial(i)  # Meu cálculo deu diferente
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
    # aux_h_pg = h_nm(
    #     n_ph,
    #     n_row,
    #     m_col,
    #     g_param,
    # )
    h_chain = t_ch * np.diagflat(
        # v=aux_h_pg * np.ones(n_a - 1, dtype=complex),
        v=array_h_nm,
        k=1,
    )

    # aux_h_mg = h_nm(
    #     n_ph,
    #     n_row,
    #     m_col,
    #     -g_param,
    # )
    h_chain += t_ch * np.diagflat(
        # v=aux_h_mg * np.ones(n_a - 1, dtype=complex),
        v=np.conj(array_h_nm),
        k=-1,
    )

    return h_chain


def block_over(
    block_matrix: np.ndarray,
    # n_a: int,
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
    n_a: int = 80,  # Number of sites
    n_ph: int = 0,  # Number of photons
    t_ch: float = 1.0,  # Hopping inside chain
    gamma: float = 0.0,  # Coupling constant
    # delta_u: float = 0.0,  # Disorder standard deviation
    omega: float = 2.0,  # Photon frequency
    # dist_kind: int = 1,  # Distribution kind
):
    """Build the chain hamiltonian."""
    # delta = np.identity(
    #     n=n_ph + 1,
    #     dtype=complex,
    # )

    # u_onsite = random_distribution(
    #     n_a,
    #     delta_u,
    #     dist_kind,
    # )

    n_n = n_a * (n_ph + 1)
    h_matrix = np.zeros(shape=(n_n, n_n), dtype=complex)

    # Diagonal subblocks
    for n_diag in range(n_ph + 1):
        h_block = np.zeros(shape=(n_a, n_a), dtype=complex)
        h_block += field_block(n_a, n_diag, omega)
        h_block += chain_diag_block(u_onsite)

        # h_matrix += np.kron(
        #     a=np.outer(delta[:, n_diag], delta[:, n_diag]),
        #     b=h_block,
        # )

        h_matrix += block_over(h_block, n_ph, n_diag, n_diag)

    # Off-diagonal subblocks
    for n_row, m_col in np.ndindex(n_ph + 1, n_ph + 1):
        h_block = np.zeros(shape=(n_a, n_a), dtype=complex)
        # h_block += field_block_hamiltonian(l_x, n_ph, n_row, m_col, omega)
        # h_block += chain_diag_block_hamiltonian(n_ph, n_row, m_col, u_onsite)
        h_block += chain_offdiag_block(
            n_a,
            n_ph,
            n_row,
            m_col,
            t_ch,
            gamma,
        )

        # h_matrix += np.kron(
        #     a=np.outer(delta[:, n_row], delta[:, m_col]),
        #     b=h_block,
        # )

        h_matrix += block_over(h_block, n_ph, n_row, m_col)

    return h_matrix


def f_func(arg: float) -> complex:
    """Auxiliary function f(x)"""
    # return 2 / (arg + CI * np.sqrt(4 - arg**2))
    return 0.5 * (arg - CI * np.sqrt(4 - arg**2))


def sigma_lead(
    energy: float,
    lead: str,
    n_a: int = 80,
    n_ph: int = 0,
    t_cs: float = 1.0,  # Hopping between source and channel
    t_cd: float = 1.0,  # Hopping between drain and channel
    t_ls: float = 1.0,  # Hopping inside source
    t_ld: float = 1.0,  # Hopping inside drain
    mu_s: float = 0.0,
    mu_d: float = 0.0,
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

    # print(block_matrix)

    # delta = np.identity(
    #     n=n_ph + 1,
    #     dtype=complex,
    # )
    n_n = n_a * (n_ph + 1)
    matrix = np.zeros(shape=(n_n, n_n), dtype=complex)
    # for n_row, m_col in np.ndindex(n_ph + 1, n_ph + 1):
    for n_row, m_col in [(0, 0)]:
        # matrix += np.kron(
        #     a=np.outer(delta[:, n_row], delta[:, m_col]),
        #     b=block_matrix,
        # )

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
):
    """Calculate the transmittance for the given self-energies and Green's function."""
    green_adv = c_t(green_ret)
    return np.real(np.trace(gamma_s @ green_ret @ gamma_d @ green_adv))


def write_input(
    return_array: np.ndarray,
    n_a: int,
    n_ph: int,
    dist_kind: int,
    ne_points: int,
    num_disorder: int,
    t_ch: float,
    gamma: float,
    delta_u: float,
    omega: float,
    t_cs: float,
    t_cd: float,
    t_ls: float,
    t_ld: float,
    mu_s: float,
    mu_d: float,
):
    """Write input."""

    filename = "L" + str(n_a) + "Nph" + str(n_ph) + ".txt"
    # original_stdout = sys.stdout
    with open(file=filename, mode="w", encoding="utf-8") as f:
        sys.stdout = f

        print("Input data")
        print("Lx=", n_a, "Nph=", n_ph, "seed=", SEED)
        print("Distribution (1 Gaussian, any other retangular)=", dist_kind)
        print("Energy grid=", ne_points, "Number of disorder conf.=", num_disorder)
        print("t=", t_ch, "gam=", gamma, "sigma (dist)=", delta_u, "Omega=", omega)
        print("tcS=", t_cs, "tcD=", t_cd, "tlS=", t_ls, "tlD=", t_ld)
        print("muD=", mu_s, "muD=", mu_d)

    np.savetxt(
        fname="dados" + filename,
        X=return_array.T,
    )


# def main():
#     """Main function."""


# if __name__ == "__main__":
#     main()
