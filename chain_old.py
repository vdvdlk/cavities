"""Build and diagonalize the hamiltonian matrix
of a 'single electron + cavity' system.
"""
import sys
import numpy as np
from scipy.special import factorial, perm

np.set_printoptions(
    precision=2,
)

SEED = 1892
RNG = np.random.default_rng(
    seed=SEED,
)
CI = complex(
    real=0,
    imag=1,
)
PI = np.pi
DISTKIND = 1


def randomdistribution(
    distkind: int,
    Lx: int,
    sigmaU: float,
):
    """Random distribution of the on-site energies."""
    if distkind == 1:  # Gaussian
        U = RNG.normal(
            scale=sigmaU,
            size=Lx,
        )
    else:
        U = RNG.uniform(
            low=-0.5 * sigmaU,
            high=0.5 * sigmaU,
            size=Lx,
        )
    return U


def funcoesP(
    Nph: int,
):
    """Auxiliary function P(j, M)."""
    P = np.zeros(
        shape=(Nph + 1, Nph + 1),
        dtype=float,
    )
    P[0, :] = 1.0

    delta = np.identity(
        n=Nph + 1,
        dtype=float,
    )

    for M in range(1, Nph + 1):
        for j in range(1, M + 1):
            P[j, M] = np.sqrt(
                perm(
                    N=M,
                    k=j,
                    # exact=True,
                )
            )

    return P, delta


def Hamiltonian(
    Lx: int,
    Nph: int,
    t: float,
    gam: float,
    sigmaU: float,
    omega: float,
    distkind: int = 1,
):
    """Build the chain hamiltonian."""
    g = gam / t
    NN = Lx * (Nph + 1)
    H = np.zeros(
        shape=(NN, NN),
        dtype=complex,
    )

    U = randomdistribution(
        distkind,
        Lx,
        sigmaU,
    )
    P, delta = funcoesP(Nph)

    # diagonal
    # print("escrevendo a matrix")

    for j in range(0, Nph + 1):
        for i in range(Lx):
            H[j * Lx + i, j * Lx + i] = j * omega + U[i]

    # off-diagonal
    # print("escrevendo a matrix")

    for j1 in range(0, Nph + 1):  # j1 = N
        for j2 in range(j1, Nph + 1):  # j2 = M
            hNM = 0.0

            for s in range(0, j1 + 1):
                fats = factorial(
                    n=s,
                    # exact=True,
                )

                for j in range(0, j2 + 1):
                    fatj = factorial(
                        n=j,
                        # exact=True,
                    )
                    hNM += (
                        np.exp(-0.5 * g**2)
                        * ((CI * g) ** s)
                        * ((CI * g) ** j)
                        * delta[j1 - s, j2 - j]
                        * P[s, j1]
                        * P[j, j2]
                        / (fats * fatj)
                    )

            for i in range(0, Lx - 1):
                H[j2 * Lx + i, j1 * Lx + i + 1] = hNM * t
                H[j2 * Lx + i + 1, j1 * Lx + i] = np.conj(hNM) * t

                H[j1 * Lx + i + 1, j2 * Lx + i] = np.conj(hNM) * t
                H[j1 * Lx + i, j2 * Lx + i + 1] = hNM * t

    return H


def writeInput(
    Lx: int,
    Nph: int,
    seed: int,
    distkind: int,
    NEpoints: int,
    Ndisorder: int,
    t: float,
    gam: float,
    sigmaU: float,
    omega: float,
    tcS: float,
    tcD: float,
    tlS: float,
    tlD: float,
    muS: float,
    muD: float,
):
    """Write input."""

    filename = "L" + str(Lx) + "Nph" + str(Nph) + ".dat"
    # original_stdout = sys.stdout
    with open(file=filename, mode="w") as f:
        sys.stdout = f

        print("Input data")
        print("Lx=", Lx, "Nph=", Nph, "seed=", seed)
        print("Distribution (1 Gaussian, any other retangular)=", distkind)
        print("Energy grid=", NEpoints, "Number of disorder conf.=", Ndisorder)
        print("t=", t, "gam=", gam, "sigma (dist)=", sigmaU, "Omega=", omega)
        print("tcS=", tcS, "tcD=", tcD, "tlS=", tlS, "tlD=", tlD)
        print("muD=", muS, "muD=", muD)


def values1(
    vm: np.ndarray,
    bins: int,
):
    """Values."""
    aval = np.sum(vm) / bins

    vr = (vm - aval) ** 2

    erval = np.sqrt(np.sum(vr) / (bins - 1))

    return erval


def Transmissivity(
    Lx: int,
    H,
    GgammaS,
    GgammaD,
    GF,
    NN,
    bTt,
    ii,
    NEPoints,
    muS: float = 0.0,
    muD: float = 0.0,
    tlS: float = 1.0,
    tlD: float = 1.0,
):
    """Transmissivity"""
    Id = np.identity(
        n=NN,
        dtype=complex,
    )

    SigmaS = np.zeros(
        shape=(NN, NN),
        dtype=complex,
    )

    SigmaD = np.zeros(
        shape=(NN, NN),
        dtype=complex,
    )

    for ne in range(1, (2 * NEPoints - 1) + 1):
        E = -2 + 2 * ne / NEPoints

        xS = (E - muS) / tlS
        xD = (E - muD) / tlD

        j = 0
        i = 1
        SigmaS[j * Lx + i, j * Lx + i] = 1.0


# def main():
#     """Main function."""


# if __name__ == "__main__":
#     main()
