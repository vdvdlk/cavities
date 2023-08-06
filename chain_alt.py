"""Build and diagonalize the hamiltonian matrix
of a 'single electron + cavity' system.
"""
import numpy as np
from scipy.special import factorial, perm

RNG = np.random.default_rng(seed=1892)
CI = complex(0, 1)


def random_distribution(
    l_x: int,
    sigma_u: float,
    dist_kind: int,
):
    """Random distribution of the on-site energies."""
    if dist_kind == 1:
        u_onsite = RNG.normal(
            scale=sigma_u,
            size=l_x,
        )
    else:
        u_onsite = RNG.uniform(
            low=-0.5 * sigma_u,
            high=0.5 * sigma_u,
            size=l_x,
        )
    return u_onsite


def p_func(
    j: int,
    m_ph: int,
):
    """Auxiliary function P(j, M)."""
    p_sq = perm(
        N=m_ph,
        k=j,
        exact=True,
    )
    return np.sqrt(p_sq)


def h_nm(
    n_ph: int,
    n_row: int,
    m_col: int,
    g_param: float,
):
    """Auxiliary function h_NM(g)"""
    delta = np.eye(n_ph + 1)
    accum_sum = complex(0)
    for i, j in np.ndindex(n_row + 1, m_col + 1):
        accum_prod = complex(1)
        fat_i = factorial(n=i, exact=True)
        fat_j = factorial(n=j, exact=True)

        accum_prod *= delta[n_row - i, m_col - j]
        accum_prod *= (CI * g_param) ** i / fat_i
        accum_prod *= (CI * g_param) ** j / fat_j
        accum_prod *= p_func(j, m_col) * p_func(i, n_row)

        accum_sum += accum_prod

    return np.exp(-(g_param**2) / 2) * accum_sum


def field_block_hamiltonian(
    l_x: int,
    n_ph: int,
    n_row: int,
    m_col: int,
    omega: float,
):
    """Build the field block hamiltonian"""
    delta = np.eye(n_ph + 1)
    id_matrix = np.identity(
        n=l_x,
        dtype=complex,
    )
    h_field = id_matrix * delta[n_row, m_col] * n_row * omega
    return h_field


def chain_diag_block_hamiltonian(
    n_ph: int,
    n_row: int,
    m_col: int,
    u_onsite: np.ndarray,
):
    """Build the chain diagonal block hamiltonian"""
    delta = np.eye(n_ph + 1)
    h_chain = np.array(
        object=delta[n_row, m_col] * np.diagflat(u_onsite),
        dtype=complex,
    )
    return h_chain


def chain_offdiag_block_hamiltonian(
    l_x: int,
    n_ph: int,
    n_row: int,
    m_col: int,
    t_hop: float,
    gamma: float,
):
    """Build the chain off-diagonal block hamiltonian"""
    g_param = gamma / t_hop

    aux_h_pg = h_nm(
        n_ph,
        n_row,
        m_col,
        g_param,
    )
    h_chain = t_hop * np.diagflat(
        v=aux_h_pg * np.ones(l_x - 1, dtype=complex),
        k=1,
    )

    aux_h_mg = h_nm(
        n_ph,
        n_row,
        m_col,
        -g_param,
    )
    h_chain += t_hop * np.diagflat(
        v=aux_h_mg * np.ones(l_x - 1, dtype=complex),
        k=-1,
    )

    return h_chain


def hamiltonian(
    l_x: int,
    n_ph: int,
    t_hop: float,
    gamma: float,
    sigma_u: float,
    omega: float,
    dist_kind: int = 0,
):
    """Build the system hamiltonian"""
    delta = np.eye(n_ph + 1)
    # g = gamma / t_hop

    u_onsite = random_distribution(
        l_x,
        sigma_u,
        dist_kind,
    )

    n_n = l_x * (n_ph + 1)
    h_matrix = np.zeros(shape=(n_n, n_n), dtype=complex)
    for n_row, m_col in np.ndindex(n_ph + 1, n_ph + 1):
        h_block = np.zeros(shape=(l_x, l_x), dtype=complex)

        h_block += field_block_hamiltonian(l_x, n_ph, n_row, m_col, omega)
        h_block += chain_diag_block_hamiltonian(n_ph, n_row, m_col, u_onsite)
        h_block += chain_offdiag_block_hamiltonian(
            l_x, n_ph, n_row, m_col, t_hop, gamma
        )

        h_matrix += np.kron(
            a=np.outer(delta[:, n_row], delta[:, m_col]),
            b=h_block,
        )

    return h_matrix


def main():
    """Main function."""
    # l_x, n_ph, seed, distkind = 20, 2, 1892, 1
    # t, gam, sigma_u = 1.000, 0.0001, 0.200
    # TEMP, OMEGA = 0.0001, 0.500

    # NN = l_x * (n_ph + 1)

    matrix = hamiltonian(
        l_x=4,
        n_ph=2,
        t_hop=1.0,
        gamma=0.1,
        sigma_u=1.0,
        omega=0.0,
    )
    print(matrix, end="\n\n")

    eigvals, eigvecs = np.linalg.eigh(a=matrix, UPLO="U")
    print(eigvals, end="\n\n")
    print(eigvecs, end="\n\n")

    np.savetxt(
        fname="pytest_alt.txt",
        X=hamiltonian(l_x=4, n_ph=2, t_hop=1.0, gamma=0.1, sigma_u=0.0, omega=1.0),
        fmt="%.2e",
    )


if __name__ == "__main__":
    main()
