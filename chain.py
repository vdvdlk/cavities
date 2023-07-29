"""chain.f90 in Python."""
import numpy as np


def random_distribution(
    l_x: int,
    sigma_u: float,
    dist_kind: int,
    rng: np.random.Generator,
):
    """Random distribution of the on-site energies."""
    if dist_kind == 1:
        u_i = rng.normal(
            scale=sigma_u,
            size=l_x,
        )
    else:
        u_i = rng.random(size=l_x)
        u_i = sigma_u * (u_i - 0.5)
    return u_i


def hamiltonian(
    l_x: int,
    n_ph: int,
    sigma_u: float,
    omega: float,
    dist_kind: int,
    rng: np.random.Generator,
):
    """Build the system hamiltonian."""
    n_sites = l_x * (n_ph + 1)
    h_matrix = np.zeros(
        shape=(n_sites, n_sites),
        dtype=complex,
    )

    u_i = random_distribution(l_x, sigma_u, dist_kind, rng)

    # Diagonal
    for j in range(0, n_ph + 1):
        for i in range(1, l_x + 1):
            h_matrix[j * l_x + i, j * l_x + i] += j * omega + u_i[i]

    # Off-diagonal
    for j_1 in range(0, n_ph + 1):
        for j_2 in range(j_1, n_ph + 1):
            for i in range(1):
                h_matrix += 0.0

    return h_matrix


def main():
    """Main function."""
    l_x, N_PH, SEED, DISTKIND = 20, 2, 1892, 1
    T, GAM, SIGMA_U = 1.000, 0.0001, 0.200
    TEMP, OMEGA = 0.0001, 0.500

    SEED = 1892
    RNG = np.random.default_rng(seed=SEED)

    NN = l_x * (N_PH + 1)
    return None


if __name__ == "__main__":
    main()
