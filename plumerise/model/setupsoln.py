import numpy as np
from numba import jit, njit
from numba.experimental import jitclass
from numba.types import float64, int64, string

from .MetDataSet import MetDataArrays, SpecHumidity, esfunc, eswfunc
from .ParamSet import ParamsSet
from .SolverParams import SolverSet

solnArraySpec = [
    # solnArray stores the solution data.
    # Each variable has values as function of height
    ("s", float64[:]),
    ("Q", float64[:]),
    ("M", float64[:]),
    ("phia", float64[:]),
    ("phiv", float64[:]),
    ("phis", float64[:]),
    ("phiw", float64[:]),
    ("phii", float64[:]),
    ("Cp", float64[:]),
    ("rho", float64[:]),
    ("T", float64[:]),
    ("theta", float64[:]),
    ("psi", float64[:]),
    ("x", float64[:]),  # E-W coordinate,
    ("y", float64[:]),  # N-S coordinate,
    ("z", float64[:]),
    ("Pa", float64[:]),
    ("Ta", float64[:]),
    ("RH", float64[:]),
    ("phivA", float64[:]),
    ("rhoAtm", float64[:]),
    ("V", float64[:]),
    ("psia", float64[:]),
    ("L", float64[:]),
    ("Lgaussian", float64[:]),
    ("U", float64[:]),
    ("Ugaussian", float64[:]),
    ("Uz", float64[:]),
    ("Uzgaussian", float64[:]),
    ("E", float64[:]),
    ("n", float64[:]),
    ("Q0", float64),
    ("Hn", float64),
    ("Htop", float64),
    ("HtopL", float64),
    ("HtopLg", float64),
    ("HCondense", float64),
    ("Collapse", int64),
]


@jitclass(solnArraySpec)
class solnArray(object):
    s: np.ndarray
    Q: np.ndarray
    M: np.ndarray
    phia: np.ndarray
    phiv: np.ndarray
    phis: np.ndarray
    phiw: np.ndarray
    phii: np.ndarray
    Cp: np.ndarray
    rho: np.ndarray
    T: np.ndarray
    theta: np.ndarray
    psi: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    Pa: np.ndarray
    Ta: np.ndarray
    RH: np.ndarray
    phivA: np.ndarray
    rhoAtm: np.ndarray
    V: np.ndarray
    psia: np.ndarray
    L: np.ndarray
    Lgaussian: np.ndarray
    U: np.ndarray
    Ugaussian: np.ndarray
    Uz: np.ndarray
    Uzgaussian: np.ndarray
    E: np.ndarray
    n: np.ndarray
    Q0: float64
    Hn: float64
    Htop: float64
    HtopL: float64
    HtopLg: float64
    HCondense: float64
    Collapse: int64

    def __init__(self, N):
        self.s = np.zeros((N), dtype=np.float64)
        self.Q = np.zeros((N), dtype=np.float64)
        self.M = np.zeros((N), dtype=np.float64)
        self.phia = np.zeros((N), dtype=np.float64)
        self.phiv = np.zeros((N), dtype=np.float64)
        self.phis = np.zeros((N), dtype=np.float64)
        self.phiw = np.zeros((N), dtype=np.float64)
        self.phii = np.zeros((N), dtype=np.float64)
        self.Cp = np.zeros((N), dtype=np.float64)
        self.rho = np.zeros((N), dtype=np.float64)
        self.T = np.zeros((N), dtype=np.float64)
        self.theta = np.zeros((N), dtype=np.float64)
        self.psi = np.zeros((N), dtype=np.float64)
        self.x = np.zeros((N), dtype=np.float64)
        self.y = np.zeros((N), dtype=np.float64)
        self.z = np.zeros((N), dtype=np.float64)
        self.Pa = np.zeros((N), dtype=np.float64)
        self.Ta = np.zeros((N), dtype=np.float64)
        self.RH = np.zeros((N), dtype=np.float64)
        self.phivA = np.zeros((N), dtype=np.float64)
        self.rhoAtm = np.zeros((N), dtype=np.float64)
        self.V = np.zeros((N), dtype=np.float64)
        self.psia = np.zeros((N), dtype=np.float64)
        self.L = np.zeros((N), dtype=np.float64)
        self.Lgaussian = np.zeros((N), dtype=np.float64)
        self.U = np.zeros((N), dtype=np.float64)
        self.Ugaussian = np.zeros((N), dtype=np.float64)
        self.Uz = np.zeros((N), dtype=np.float64)
        self.Uzgaussian = np.zeros((N), dtype=np.float64)
        self.E = np.zeros((N), dtype=np.float64)
        self.n = np.zeros((N), dtype=np.float64)

        return

    def update(
        self,
        s: float64,
        y: np.ndarray,
        Params: ParamsSet,
        Met: MetDataArrays,
        SolParams: SolverSet,
        z_pos: int64,
        j: int64,
    ):

        if j < SolParams.MaxSolnPnts:

            self.s[j] = s
            self.Q[j] = np.pi * y[0, 0]
            self.M[j] = np.pi * y[1, 0]
            self.phia[j] = y[2, 0]
            self.phiv[j] = y[3, 0]
            self.phis[j] = y[4, 0]
            self.phiw[j] = y[5, 0]
            self.phii[j] = y[6, 0]
            self.Cp[j] = y[7, 0]
            self.rho[j] = y[8, 0]
            self.T[j] = y[9, 0]
            self.theta[j] = y[10, 0]
            self.psi[j] = y[11, 0]
            self.x[j] = y[12, 0]
            self.y[j] = y[13, 0]
            self.z[j] = y[14, 0]

            z = y[14, 0]

            MetVals, z_pos = Met.interp(z, z_pos)
            Ta = MetVals["Ta"]
            esa, _ = eswfunc(Ta)  # saturation vapor pressure in atmosphere

            Pa = MetVals["Pa"]
            RH = MetVals["RH"]
            psia = MetVals["psia"]
            V = MetVals["V"]

            phivA = SpecHumidity(RH, esa, Pa, Params.Ra, Params.Rv)
            phiaA = 1.0 - phivA
            # atmospheric density
            Rga = phivA * Params.Rv + phiaA * Params.Ra
            rhoAtm = Pa / Rga / Ta

            self.Pa[j] = Pa
            self.Ta[j] = Ta
            self.RH[j] = RH
            self.phivA[j] = phivA
            self.rhoAtm[j] = rhoAtm
            self.V[j] = V
            self.psia[j] = psia

            L = y[0, 0] / np.sqrt(y[8, 0] * y[1, 0])  # L = Q/sqrt(rho*M)

            U = y[1, 0] / y[0, 0]
            E = y[7, 0] * y[9, 0] * y[0, 0]  # E = Cp*T*Q

            self.L[j] = L
            self.Lgaussian[j] = L / np.sqrt(2.0)
            self.U[j] = U
            self.Ugaussian[j] = 2.0 * U
            self.E[j] = np.pi * E
            self.n[j] = self.phia[j] + self.phiv[j]

            self.Uz[j] = U * np.sin(y[10, 0])
            self.Uzgaussian[j] = 2.0 * U * np.sin(y[10, 0])

            j += 1

        return z_pos, j

    def clip(self, n: int):

        self.s = self.s[:n]
        self.Q = self.Q[:n]
        self.M = self.M[:n]
        self.phia = self.phia[:n]
        self.phiv = self.phiv[:n]
        self.phis = self.phis[:n]
        self.phiw = self.phiw[:n]
        self.phii = self.phii[:n]
        self.Cp = self.Cp[:n]
        self.rho = self.rho[:n]
        self.T = self.T[:n]
        self.theta = self.theta[:n]
        self.psi = self.psi[:n]
        self.x = self.x[:n]
        self.y = self.y[:n]
        self.z = self.z[:n]
        self.Pa = self.Pa[:n]
        self.Ta = self.Ta[:n]
        self.RH = self.RH[:n]
        self.phivA = self.phivA[:n]
        self.rhoAtm = self.rhoAtm[:n]
        self.V = self.V[:n]
        self.psia = self.psia[:n]
        self.L = self.L[:n]
        self.Lgaussian = self.Lgaussian[:n]
        self.U = self.U[:n]
        self.Ugaussian = self.Ugaussian[:n]
        self.Uz = self.Uz[:n]
        self.Uzgaussian = self.Uzgaussian[:n]
        self.E = self.E[:n]
        self.n = self.n[:n]

        return

    def NeutralHeight(self):

        Hneutral = np.float64(-1.0)

        for j in range(len(self.z) - 1):
            if (
                np.abs(self.rho[j] - self.rhoAtm[j]) < 2.0 * np.finfo(np.float64).eps
            ) and ((self.rho[j + 1] - self.rhoAtm[j + 1]) > 0.0):
                Hneutral = self.z[j]
            elif ((self.rho[j] - self.rhoAtm[j]) < 0.0) and (
                (self.rho[j + 1] - self.rhoAtm[j + 1]) > 0.0
            ):
                Hneutral = self.z[j] - (self.rho[j] - self.rhoAtm[j]) * (
                    self.z[j + 1] - self.z[j]
                ) / (
                    (self.rho[j + 1] - self.rhoAtm[j + 1])
                    - (self.rho[j] - self.rhoAtm[j])
                )

        self.Hn = Hneutral
        return
