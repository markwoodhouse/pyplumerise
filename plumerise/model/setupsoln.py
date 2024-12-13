from functools import cache

import numpy as np
from numba import jit, njit
# from numba.experimental import jitclass
from numba.types import float64, int64, string

from .MetDataSet import Met_interp, SpecHumidity, esfunc, eswfunc

# from .ParamSet import ParamsSet
# from .SolverParams import SolverSet


@njit(cache=True)
def init_soln(N):

    soln = np.nan*np.ones((N, 32), dtype=np.float64)
    # s -> 0
    # logQ -> 1
    # logM -> 2
    # phia -> 3
    # phiv -> 4
    # phis -> 5
    # phiw -> 6
    # phii -> 7
    # Cp -> 8
    # rho -> 9
    # T -> 10
    # theta -> 11
    # z -> 12
    # psi -> 13
    # x -> 14
    # y -> 15
    # Pa -> 16
    # Ta -> 17
    # RH -> 18
    # phivA -> 19
    # rhoAtm -> 20
    # V -> 21
    # psia -> 22
    # L -> 23
    # Lgaussian -> 24
    # U -> 25
    # 26 -> Ugaussian
    # 27 -> Q
    # 28 -> M
    # 29 -> E
    # 30 -> n = phia + phiv
    # 31 -> Uz
    # 32 -> Uzgaussian

    return soln


@njit(cache=True)
def update_soln(
    soln: np.ndarray,
    s: np.float64,
    y_vec: np.ndarray,
    Params: np.ndarray,
    Met: np.ndarray,
    SolParams: np.ndarray,
    z_pos: np.int64,
    j: np.int64,
):
    
    logQ = y_vec[0, 0]  # logQ
    logM = y_vec[1, 0]  # logM
    phia = y_vec[2, 0]  # phia
    phiv = y_vec[3, 0]  # phiv
    phis = y_vec[4, 0]  # phis
    if SolParams["eqn"][0]==2:
        phiw = y_vec[5, 0]  # phiw
        phii = y_vec[6, 0]  # phii
        Cp = y_vec[7, 0]  # Cp
        rho = y_vec[8, 0]  # rho
        T = y_vec[9, 0]  # T
        theta = y_vec[10, 0]  # theta
        z = y_vec[11, 0]  # z
        if SolParams["threeD"][0]:
            psi = y_vec[12, 0]  # psi
            x = y_vec[13, 0]  # x
            y = y_vec[14, 0]  # y
        else:
            psi = np.float64(0.0)
            x = np.float64(0.0)
            y = np.float64(0.0)
    elif SolParams["eqn"][0]==1:
        phiw = y_vec[5, 0]  # phiw
        phii = np.float64(0.0)  # phii
        Cp = y_vec[6, 0]  # Cp
        rho = y_vec[7, 0]  # rho
        T = y_vec[8, 0]  # T
        theta = y_vec[9, 0]  # theta
        z = y_vec[10, 0]  # z
        if SolParams["threeD"][0]:
            psi = y_vec[11, 0]  # psi
            x = y_vec[12, 0]  # x
            y = y_vec[13, 0]  # y
        else:
            psi = np.float64(0.0)
            x = np.float64(0.0)
            y = np.float64(0.0)
    else: # dry
        phiw = np.float64(0.0)  # phiw
        phii = np.float64(0.0)  # phii
        Cp = y_vec[5, 0]  # Cp
        rho = y_vec[6, 0]  # rho
        T = y_vec[7, 0]  # T
        theta = y_vec[8, 0]  # theta
        z = y_vec[9, 0]  # z
        if SolParams["threeD"][0]:
            psi = y_vec[10, 0]  # psi
            x = y_vec[11, 0]  # x
            y = y_vec[12, 0]  # y
        else:
            psi = np.float64(0.0)
            x = np.float64(0.0)
            y = np.float64(0.0)


    if j < SolParams["MaxSolnPnts"]:

        Ra = Params["Ra"][0]
        Rv = Params["Rv"][0]

        soln[j, 0] = s  # s
        soln[j, 1] = logQ
        soln[j, 2] = logM
        soln[j, 3] = phia
        soln[j, 4] = phiv
        soln[j, 5] = phis
        soln[j, 6] = phiw
        soln[j, 7] = phii
        soln[j, 8] = Cp
        soln[j, 9] = rho
        soln[j, 10] = T
        soln[j, 11] = theta
        soln[j, 12] = z
        soln[j, 13] = psi
        soln[j, 14] = x
        soln[j, 15] = y
        
        MetVals, z_pos = Met_interp(Met, z, z_pos)
        Ta = MetVals["Ta"]
        esa, _ = eswfunc(Ta)  # saturation vapor pressure in atmosphere
        Pa = MetVals["Pa"]
        RH = MetVals["RH"]
        psia = MetVals["psia"]
        V = MetVals["V"]
        phivA = SpecHumidity(RH, esa, Pa, Ra, Rv)
        phiaA = 1.0 - phivA
        # atmospheric density
        Rga = phivA * Rv + phiaA * Ra
        rhoAtm = Pa / Rga / Ta
        soln[j, 16] = Pa  # Pa
        soln[j, 17] = Ta  # Ta
        soln[j, 18] = RH  # RH
        soln[j, 19] = phivA  # phivA
        soln[j, 20] = rhoAtm  # rhoAtm
        soln[j, 21] = V  # V
        soln[j, 22] = psia  # psia
        Q = np.exp(logQ)
        M = np.exp(logM)
        L = Q / np.sqrt(rho * M)  # L = Q/sqrt(rho*M)
        U = M / Q  # U = M/Q
        E = Cp * T * Q
        soln[j, 23] = L  # L
        soln[j, 24] = L / np.sqrt(2.0)  # Lgaussian
        soln[j, 25] = U  # U
        soln[j, 26] = 2.0 * U  # Ugaussian
        soln[j, 27] = Q
        soln[j, 28] = M
        soln[j, 29] = E  # E
        soln[j, 30] = phia + phiv
        soln[j, 31] = U * np.sin(theta)  # Uz
        soln[j, 32] = 2.0 * U * np.sin(theta)  # Uzgaussian
        j += 1
    return soln, z_pos, j


@njit(cache=True)
def clip_soln(soln: np.ndarray, n: np.int64):

    return soln[:n, :]


@njit(cache=True)
def get_soln(soln: np.ndarray, var: str, with_pi: bool=True):

    match var:
        case 's':
            return soln[:,0]
        case 'logQ':
            return soln[:,1]
        case 'logM':
            return soln[:,2]
        case 'phia':
            return soln[:,3]
        case 'phiv':
            return soln[:,4]
        case 'phis':
            return soln[:,5]
        case 'phiw':
            return soln[:,6]
        case 'phii':
            return soln[:,7]
        case 'Cp':
            return soln[:,8]
        case 'rho':
            return soln[:,9]
        case 'T':
            return soln[:,10]
        case 'theta':
            return soln[:,11]
        case 'z':
            return soln[:,12]
        case 'psi':
            return soln[:,13]
        case 'x':
            return soln[:,14]
        case 'y':
            return soln[:,15]
        case 'Pa':
            return soln[:,16]
        case 'Ta':
            return soln[:,17]
        case 'RH':
            return soln[:,18]
        case 'phivA':
            return soln[:,19]
        case 'rhoAtm':
            return soln[:,20]
        case 'V':
            return soln[:,21]
        case 'psia':
            return soln[:,22]
        case 'L':
            return soln[:,23]
        case 'Lgaussian':
            return soln[:,24]
        case 'U':
            return soln[:,25]
        case 'Ugaussian':
            return soln[:,26]
        case 'Q':
            if with_pi:
                return np.pi * soln[:,27]
            else:
                return soln[:,27]
        case 'M':
            if with_pi:
                return np.pi * soln[:,28]
            else:
                return soln[:,28]
        case 'E':
            if with_pi:
                return np.pi * soln[:,29]
            else:    
                return soln[:,29]
        case 'n':
            return soln[:,30]
        case 'Uz':
            return soln[:,31]
        case 'Uzgaussian':
            return soln[:,32]
        case _:
            raise ValueError("Requested variable not in soln array")
        

@njit(cache=True)
def NeutralHeight(soln):

    Hneutral = np.float64(-1.0)

    z = get_soln(soln, 'z')
    rho = get_soln(soln, 'rho')
    rhoAtm = get_soln(soln, 'rhoAtm')

    for j in range(len(z) - 1):
        if (np.abs(rho[j] - rhoAtm[j]) < 10.0 * np.finfo(np.float64).eps) and (
            (rho[j + 1] - rhoAtm[j + 1]) > 0.0
        ):
            Hneutral = z[j]
            break
        elif ((rho[j] - rhoAtm[j]) < 0.0) and (
            (rho[j + 1] - rhoAtm[j + 1]) > 0.0
        ):
            Hneutral = z[j] - (rho[j] - rhoAtm[j]) * (
                z[j + 1] - z[j]
            ) / (
                (rho[j + 1] - rhoAtm[j + 1]) - (rho[j] - rhoAtm[j])
            )
            break

    return Hneutral