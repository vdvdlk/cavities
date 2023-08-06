import random as rng
import numpy as np


rng.seed(1892)
PI = np.pi
CI = complex(0, 1)


def randomdistribution(
    Lx: int,
    sigmaU: float,
    distkind: int,
):
    U = np.zeros(Lx)
    if distkind == 1:  #Gaussian
        for i in range(Lx):
            r1 = rng.random()
            r2 = rng.random()
            U[i] = sigmaU * np.cos(2 * PI * r1) * np.sqrt(- 2.0 * np.log(1.0 - r2))
    else:  # uniform
        for i in range(Lx):
            U[i] = sigmaU * (rng.random() - 0.5)
    return U


def funcoesP(Nph: int):
    P = np.zeros(shape=(Nph + 1, Nph + 1))
    P[0, :] = np.ones(Nph + 1)

    delta = np.zeros(shape=(Nph + 1, Nph + 1))
    delta[0, 0] = 1.0

    for M in range(1, Nph + 1):
        delta[M, M] = 1.0
        for j in range(1, M + 1):
            P[j, M] = np.sqrt(M - (j - 1)) * P[j - 1, M]

    return P, delta


def Hamiltonian(
    Lx: int,
    Nph: int,
    t: float,
    gam: float,
    sigmaU: float,
    omega: float,
    distkind: int,
):
    """Build the system hamiltonian."""
    g = gam / t
    NN = Lx * (Nph + 1)
    H = np.zeros(
        shape=(NN, NN),
        dtype=complex,
    )

    U = randomdistribution(Lx, sigmaU, distkind)
    P, delta = funcoesP(Nph)

    # diagonal
    print("escrevendo a matrix")

    for j in range(0, Nph + 1):
        for i in range(Lx):
            H[j * Lx + i, j * Lx + i] = j * omega + U[i]

    # off-diagonal
    print("escrevendo a matrix")

    for j1 in range(0, Nph + 1):  #j1 = N
        for j2 in range(j1, Nph + 1):  #j2 = M
            hNM = 0.0
            fats = 1
            for s in range(0, j1 + 1):
                fatj = 1
                for j in range(0, j2 + 1):
                    hNM += np.exp(-0.5 * g**2) * ((CI*g)**s) * ((CI*g)**j) * \
                        delta[j1 - s, j2 - j] * P[s, j1] * P[j, j2] / (fats * fatj)
                    fatj *= j + 1
                fats *= s + 1
            
            for i in range(0, Lx - 1):
                H[j2 * Lx + i, j1 * Lx + i + 1] = hNM * t
                H[j2 * Lx + i + 1, j1 * Lx + i] = np.conj(hNM) * t

                H[j1 * Lx + i + 1, j2 * Lx + i] = np.conj(hNM) * t
                H[j1 * Lx + i, j2 * Lx + i + 1] = hNM * t

    return H


def main():
    """Main function."""
    h_mat = Hamiltonian(
        Lx=4,
        Nph=2,
        t=1.0,
        gam=0.1,
        sigmaU=0.0,
        omega=1.0,
        distkind=0,
    )
    # np.savetxt(
    #     fname="pytest.txt",
    #     X=h_mat,
    #     fmt="%.2e",
    # )
    return None


if __name__ == "__main__":
    main()
