import numpy as np
from numba.experimental import jitclass
from numba.types import float64, int64


def set_IC(
    L0: float = 100.0,  # vent radius [m]
    U0: float = 100.0,  # exit velocity [m/s]
    n0: float = 0.05,  # gas mass fraction [1]
    T0: float = 1200.0,  # temperature at the vent [K]
    z0: float = 0.0,  # vent altitude [m]
    Q0: float = 0.0,
    M0: float = 0.0,
    E0: float = 0.0,
    phia0: float = 0.0,
    phiv0: float = 0.0,
    phis0: float = 0.0,
    phiw0: float = 0.0,
    phii0: float = 0.0,
    Rg0: float = 0.0,
    Cp0: float = 0.0,
    rho0: float = 0.0,
    Pa0: float = 0.0,
    FoundL0: int = 0,
    FoundU0: int = 0,
    FoundQ0: int = 0,
    FoundM0: int = 0,
):

    IC_dtype = np.dtype(
        [
            ("L0", np.float64),  # vent radius [m]
            ("U0", np.float64),  # exit velocity [m/s]
            ("n0", np.float64),  # gas mass fraction [1]
            ("T0", np.float64),  # temperature at the vent [K]
            ("z0", np.float64),  # vent altitude [m]'Q0': np.float64(0.0)
            ("Q0", np.float64),
            ("M0", np.float64),
            ("E0", np.float64),
            ("phia0", np.float64),
            ("phiv0", np.float64),
            ("phis0", np.float64),
            ("phiw0", np.float64),
            ("phii0", np.float64),
            ("Rg0", np.float64),
            ("Cp0", np.float64),
            ("rho0", np.float64),
            ("Pa0", np.float64),
            ("FoundL0", np.int64),
            ("FoundU0", np.int64),
            ("FoundQ0", np.int64),
            ("FoundM0", np.int64),
        ]
    )

    IC = np.zeros((1), dtype=IC_dtype)

    IC["L0"][0] = np.float64(L0)
    IC["U0"][0] = np.float64(U0)
    IC["n0"][0] = np.float64(n0)
    IC["T0"][0] = np.float64(T0)
    IC["z0"][0] = np.float64(z0)
    IC["Q0"][0] = np.float64(Q0)
    IC["M0"][0] = np.float64(M0)
    IC["E0"][0] = np.float64(E0)
    IC["phia0"][0] = np.float64(phia0)
    IC["phiv0"][0] = np.float64(phiv0)
    IC["phis0"][0] = np.float64(phis0)
    IC["phiw0"][0] = np.float64(phiw0)
    IC["phii0"][0] = np.float64(phii0)
    IC["Rg0"][0] = np.float64(Rg0)
    IC["Cp0"][0] = np.float64(Cp0)
    IC["rho0"][0] = np.float64(rho0)
    IC["Pa0"][0] = np.float64(Pa0)
    IC["FoundL0"][0] = FoundL0
    IC["FoundU0"][0] = FoundU0
    IC["FoundQ0"][0] = FoundQ0
    IC["FoundM0"][0] = FoundM0

    return IC


# ICSetSpec = [
#     ("L0", float64),  # vent radius [m]
#     ("U0", float64),  # exit velocity [m/s]
#     ("n0", float64),  # gas mass fraction [1]
#     ("T0", float64),  # temperature at the vent [K]
#     ("z0", float64),  # vent altitude [m]'Q0': np.float64(0.0)
#     ("Q0", float64),
#     ("M0", float64),
#     ("E0", float64),
#     ("phia0", float64),
#     ("phiv0", float64),
#     ("phis0", float64),
#     ("phiw0", float64),
#     ("phii0", float64),
#     ("Rg0", float64),
#     ("Cp0", float64),
#     ("rho0", float64),
#     ("Pa0", float64),
#     ("FoundL0", int64),
#     ("FoundU0", int64),
#     ("FoundQ0", int64),
#     ("FoundM0", int64),
# ]


# @jitclass(ICSetSpec)
# class ICSet(object):
#     L0: float64  # vent radius [m]
#     U0: np.float64  # exit velocity [m/s]
#     n0: np.float64  # gas mass fraction [1]
#     T0: np.float64  # temperature at the vent [K]
#     z0: np.float64  # vent altitude [m]'Q0': np.float64(0.0)
#     Q0: np.float64
#     M0: np.float64
#     E0: np.float64
#     phia0: np.float64
#     phiv0: np.float64
#     phis0: np.float64
#     phiw0: np.float64
#     phii0: np.float64
#     Rg0: np.float64
#     Cp0: np.float64
#     rho0: np.float64
#     Pa0: np.float64
#     FoundL0: int
#     FoundU0: int
#     FoundQ0: int
#     FoundM0: int

#     def __init__(
#         self,
#         L0: float = 100.0,  # vent radius [m]
#         U0: float = 100.0,  # exit velocity [m/s]
#         n0: float = 0.05,  # gas mass fraction [1]
#         T0: float = 1200.0,  # temperature at the vent [K]
#         z0: float = 0.0,  # vent altitude [m]'Q0': np.float64(0.0)
#         Q0: float = 0.0,
#         M0: float = 0.0,
#         E0: float = 0.0,
#         phia0: float = 0.0,
#         phiv0: float = 0.0,
#         phis0: float = 0.0,
#         phiw0: float = 0.0,
#         phii0: float = 0.0,
#         Rg0: float = 0.0,
#         Cp0: float = 0.0,
#         rho0: float = 0.0,
#         Pa0: float = 0.0,
#         FoundL0: int = 0,
#         FoundU0: int = 0,
#         FoundQ0: int = 0,
#         FoundM0: int = 0,
#     ):

#         self.L0 = np.float64(L0)
#         self.U0 = np.float64(U0)
#         self.n0 = np.float64(n0)
#         self.T0 = np.float64(T0)
#         self.z0 = np.float64(z0)
#         self.Q0 = np.float64(Q0)
#         self.M0 = np.float64(M0)
#         self.E0 = np.float64(E0)
#         self.phia0 = np.float64(phia0)
#         self.phiv0 = np.float64(phiv0)
#         self.phis0 = np.float64(phis0)
#         self.phiw0 = np.float64(phiw0)
#         self.phii0 = np.float64(phii0)
#         self.Rg0 = np.float64(Rg0)
#         self.Cp0 = np.float64(Cp0)
#         self.rho0 = np.float64(rho0)
#         self.Pa0 = np.float64(Pa0)
#         self.FoundL0 = FoundL0
#         self.FoundU0 = FoundU0
#         self.FoundQ0 = FoundQ0
#         self.FoundM0 = FoundM0

#         return
