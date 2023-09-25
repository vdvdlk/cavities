"""Build and diagonalize the hamiltonian matrix
of a 'single electron + cavity' system.
"""
import numpy as np
from scipy.special import factorial, perm

RNG = np.random.default_rng(seed=1892)
CI = complex(0, 1)
PI = np.pi


def c_t(matrix: np.ndarray) -> np.ndarray:
    """Returns the conjugate transpose of a matrix."""
    return np.conj(matrix.T)


def random_distribution(
    n_a: int,
    sigma_u: float,
    dist_kind: int,
):
    """Random distribution of the on-site energies."""
    if dist_kind == 1:
        u_onsite = RNG.normal(
            scale=sigma_u,
            size=n_a,
        )
    else:
        u_onsite = RNG.uniform(
            low=-0.5 * sigma_u,
            high=0.5 * sigma_u,
            size=n_a,
        )
    return u_onsite


def p_func(
    j: int,
    m_ph: int,
):
    """Auxiliary function P(j, M)."""
    p_squared = perm(
        N=m_ph,
        k=j,
        exact=True,
    )
    return np.sqrt(p_squared)


def h_nm(
    n_ph: int,
    n_row: int,
    m_col: int,
    g_param: float,
):
    """Auxiliary function h_NM(g)."""
    delta = np.eye(n_ph + 1)
    accum_sum = complex(0)
    for i, j in np.ndindex(n_row + 1, m_col + 1):
        accum_prod = complex(1)

        accum_prod *= delta[n_row - i, m_col - j]
        accum_prod *= (CI * g_param) ** i / factorial(i, exact=False)
        accum_prod *= (CI * g_param) ** j / factorial(j, exact=False)
        accum_prod *= p_func(j, m_col) * p_func(i, n_row)

        accum_sum += accum_prod

    return np.exp(-(g_param**2) / 2) * accum_sum


def field_block_hamiltonian(
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


def chain_diag_block_hamiltonian(
    u_onsite: np.ndarray,
):
    """Build the chain diagonal block hamiltonian."""
    h_chain = np.array(
        object=np.diagflat(u_onsite),
        dtype=complex,
    )
    return h_chain


def chain_offdiag_block_hamiltonian(
    n_a: int,
    n_ph: int,
    n_row: int,
    m_col: int,
    t_ch: float,
    gamma: float,
):
    """Build the chain off-diagonal block hamiltonian."""
    g_param = gamma / t_ch

    aux_h_pg = h_nm(
        n_ph,
        n_row,
        m_col,
        g_param,
    )
    h_chain = t_ch * np.diagflat(
        v=aux_h_pg * np.ones(n_a - 1, dtype=complex),
        k=1,
    )

    aux_h_mg = h_nm(
        n_ph,
        n_row,
        m_col,
        -g_param,
    )
    h_chain += t_ch * np.diagflat(
        v=aux_h_mg * np.ones(n_a - 1, dtype=complex),
        k=-1,
    )

    return h_chain


def hamiltonian(
    n_a: int = 4,
    n_ph: int = 0,
    t_ch: float = 1.0,
    gamma: float = 0.0,
    sigma_u: float = 0.0,
    omega: float = 0.0,
    dist_kind: int = 1,
):
    """Build the chain hamiltonian."""
    # delta = np.eye(n_ph + 1)
    delta = np.identity(
        n=n_ph + 1,
        dtype=complex,
    )
    # g = gamma / t_hop

    u_onsite = random_distribution(
        n_a,
        sigma_u,
        dist_kind,
    )

    n_n = n_a * (n_ph + 1)
    h_matrix = np.zeros(shape=(n_n, n_n), dtype=complex)

    # Diagonal subblocks
    for n_diag in range(n_ph + 1):
        h_block = np.zeros(shape=(n_a, n_a), dtype=complex)
        h_block += field_block_hamiltonian(n_a, n_diag, omega)
        h_block += chain_diag_block_hamiltonian(u_onsite)

        h_matrix += np.kron(
            a=np.outer(delta[:, n_diag], delta[:, n_diag]),
            b=h_block,
        )

    # Off-diagonal elements
    for n_row, m_col in np.ndindex(n_ph + 1, n_ph + 1):
        h_block = np.zeros(shape=(n_a, n_a), dtype=complex)
        # h_block += field_block_hamiltonian(l_x, n_ph, n_row, m_col, omega)
        # h_block += chain_diag_block_hamiltonian(n_ph, n_row, m_col, u_onsite)
        h_block += chain_offdiag_block_hamiltonian(
            n_a,
            n_ph,
            n_row,
            m_col,
            t_ch,
            gamma,
        )

        h_matrix += np.kron(
            a=np.outer(delta[:, n_row], delta[:, m_col]),
            b=h_block,
        )

    return h_matrix


def f_func(arg):
    """Auxiliary function f(x)"""
    return 2 / (arg + CI * np.sqrt(4 - arg**2))


# def sigma(
#     energy: float,
#     lead: str,
#     n_a: int = 4,
#     n_ph: int = 0,
#     t_c: float = 1,
#     t_s: float = 1,
#     t_d: float = 1,
#     # epsilon_s: float,
#     # epsilon
# ):
#     """Build the retarded drain self-energy matrix."""
#     block_matrix = np.zeros(shape=(n_a, n_a), dtype=complex)

#     if lead == "s":
#         epsilon_s = -1.0
#         k_1 = np.arccos((epsilon_s - energy) / (2 * t_s))
#         block_matrix[0, 0] = -t_c * np.exp(CI * k_1)

#     elif lead == "d":
#         epsilon_d = 1.0
#         k_2 = np.arccos((epsilon_d - energy) / (2 * t_d))
#         block_matrix[-1, -1] = -t_c * np.exp(CI * k_2)

#     delta = np.eye(n_ph + 1)
#     n_n = n_a * (n_ph + 1)
#     matrix = np.zeros(shape=(n_n, n_n), dtype=complex)
#     for n_row, m_col in np.ndindex(n_ph + 1, n_ph + 1):
#         matrix += np.kron(
#             a=np.outer(delta[:, n_row], delta[:, m_col]),
#             b=block_matrix,
#         )

#     return matrix


def sigma_lead(
    energy: float,
    lead: str,
    n_a: int = 4,
    # n_ph: int = 0,
    t_c: float = 1.0,
    t_s: float = 1.0,
    t_d: float = 1.0,
    # epsilon_s: float,
    # epsilon
):
    """Build the retarded drain self-energy matrix."""
    block_matrix = np.zeros(shape=(n_a, n_a), dtype=complex)

    if lead == "s":
        epsilon_s = -1.0
        k_1 = np.arccos((epsilon_s - energy) / (2 * t_s))
        block_matrix[0, 0] = -t_c * np.exp(CI * k_1)

    elif lead == "d":
        epsilon_d = 1.0
        k_2 = np.arccos((epsilon_d - energy) / (2 * t_d))
        block_matrix[-1, -1] = -t_c * np.exp(CI * k_2)

    # delta = np.identity(
    #     n=n_ph + 1,
    #     dtype=complex,
    # )
    # n_n = n_a * (n_ph + 1)
    # matrix = np.zeros(shape=(n_n, n_n), dtype=complex)
    # for n_row, m_col in np.ndindex(n_ph + 1, n_ph + 1):
    #     matrix += np.kron(
    #         a=np.outer(delta[:, n_row], delta[:, m_col]),
    #         b=block_matrix,
    #     )

    return block_matrix


def green_function(
    energy: float,
    h_matrix: np.ndarray,
    sigma_matrix: np.ndarray,
) -> np.ndarray:
    """Build the retarded Green's function of the chain."""
    identity = np.identity(
        h_matrix.shape[0],
        dtype=complex,
    )

    return np.linalg.inv(energy * identity - h_matrix - sigma_matrix)


def transmittance(
    green_ret: np.ndarray,
    gamma_s: np.ndarray,
    gamma_d: np.ndarray,
):
    """Calculate the transmittance for the given self-energies and Green's function."""
    green_adv = c_t(green_ret)
    return np.trace(gamma_s @ green_ret @ gamma_d @ green_adv)


# def main():
#     """Main function."""


# if __name__ == "__main__":
#     main()
