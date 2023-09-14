"""Build and diagonalize the hamiltonian matrix
of a 'single electron + cavity' system.
"""
import numpy as np
from scipy.special import factorial, perm

RNG = np.random.default_rng(seed=1892)
CI = complex(0, 1)


def c_t(matrix: np.ndarray):
    """Returns the conjugate transpose of a matrix."""
    return np.transpose(matrix.T)


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
        accum_prod *= (-CI * g_param) ** i / factorial(i, exact=False)
        accum_prod *= (-CI * g_param) ** j / factorial(j, exact=False)
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
    n_a: int,
    n_ph: int,
    t_ch: float,
    gamma: float,
    sigma_u: float,
    omega: float,
    dist_kind: int = 1,
):
    """Build the chain hamiltonian."""
    delta = np.eye(n_ph + 1)
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


# def f_func(x):
#     """Auxiliary function f(x)"""
#     return 2 / (x + CI * np.sqrt(4 - x**2))


def sigma(n_a: int, n_ph: int, lead: str):
    """Build the retarded drain self-energy matrix."""
    k = 2 * np.pi / n_a
    block_matrix = np.zeros(shape=(n_a, n_a), dtype=complex)
    if lead == 's':
        block_matrix[0, 0] = np.exp(2 * CI * k)  # MUST CHANGE
    elif lead == 'd':
        block_matrix[-1, -1] = np.exp(2 * CI * k)  # MUST CHANGE

    delta = np.eye(n_ph + 1)
    n_n = n_a * (n_ph + 1)
    matrix = np.zeros(shape=(n_n, n_n), dtype=complex)
    for n_row, m_col in np.ndindex(n_ph + 1, n_ph + 1):
        matrix += np.kron(
            a=np.outer(delta[:, n_row], delta[:, m_col]),
            b=block_matrix,
        )

    return matrix


def transmitance(
    n_a: int,
    n_ph: int,
    energy: float,
    h_matrix: np.ndarray,
):
    """Calculate the transmittance for a given energy value."""
    n_n = n_a * (n_ph + 1)
    identity = np.identity(n=n_n, dtype=complex)

    sigma_l = sigma(n_a, n_ph, 's') + sigma(n_a, n_ph, 'd')

    gamma_s = CI * (sigma(n_a, n_ph, 's') - c_t(sigma(n_a, n_ph, 's')))
    gamma_d = CI * (sigma(n_a, n_ph, 'd') - c_t(sigma(n_a, n_ph, 'd')))

    green_ret = np.linalg.inv(energy * identity - h_matrix - sigma_l)
    green_adv = c_t(green_ret)

    return np.trace(gamma_s @ green_ret @ gamma_d @ green_adv)



def main():
    """Main function."""
    # l_x, n_ph, seed, distkind = 20, 2, 1892, 1
    # t, gam, sigma_u = 1.000, 0.0001, 0.200
    # TEMP, OMEGA = 0.0001, 0.500

    # NN = l_x * (n_ph + 1)

    H = hamiltonian(
        n_a=4,
        n_ph=2,
        t_ch=1.0,
        gamma=0.1,
        sigma_u=0.0,
        omega=1.0,
    )
    print(H, end="\n\n")

    # eigvals, eigvecs = np.linalg.eigh(a=matrix, UPLO="U")
    # print(eigvals, end="\n\n")
    # print(eigvecs, end="\n\n")

    np.savetxt(
        fname="H_test_alt.txt",
        X=H,
        fmt="%.2e",
    )


if __name__ == "__main__":
    main()
