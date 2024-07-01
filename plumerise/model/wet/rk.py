from functools import cache
from typing import Callable

import numpy as np
from numba import njit
from numba.types import float64, int64, string

# from .IC import ICSet
from ..MetDataSet import Met_interp
from .wet_eqns import fwet

# from .ParamSet import ParamsSet
# from .SolverParams import SolverSet


@njit(cache=True)
def rkCK_wet(
    y_in: np.ndarray,
    dy: np.ndarray,
    h: np.float64,
    Params: np.ndarray,
    Met: np.ndarray,
    z_pos: np.int64,
    GasThrust: np.int64,
    threeD: bool,
):

    A2 = np.float64(0.2)
    A3 = np.float64(0.3)
    A4 = np.float64(0.6)
    A5 = np.float64(1.0)
    A6 = np.float64(0.875)
    B21 = np.float64(0.2)
    B31 = np.float64(3.0) / np.float64(40.0)
    B32 = np.float64(9.0) / np.float64(40.0)
    B41 = np.float64(0.3)
    B42 = -np.float64(0.9)
    B43 = np.float64(1.2)
    B51 = -np.float64(11.0) / np.float64(54.0)
    B52 = np.float64(2.5)
    B53 = -np.float64(70.0) / np.float64(27.0)
    B54 = np.float64(35.0) / np.float64(27.0)
    B61 = np.float64(1631.0) / np.float64(55296.0)
    B62 = np.float64(175.0) / np.float64(512.0)
    B63 = np.float64(575.0) / np.float64(13824.0)
    B64 = np.float64(44275.0) / np.float64(110592.0)
    B65 = np.float64(253.0) / np.float64(4096.0)
    C1 = np.float64(37.0) / np.float64(378.0)
    C2 = np.float64(0.0)
    C3 = np.float64(250.0) / np.float64(621.0)
    C4 = np.float64(125.0) / np.float64(594.0)
    C5 = np.float64(0.0)
    C6 = np.float64(512.0) / np.float64(1771.0)
    DC1 = C1 - np.float64(2825.0) / np.float64(27648.0)
    DC2 = np.float64(0.0)
    DC3 = C3 - np.float64(18575.0) / np.float64(48384.0)
    DC4 = C4 - np.float64(13525.0) / np.float64(55296.0)
    DC5 = -np.float64(277.0) / np.float64(14336.0)
    DC6 = C6 - np.float64(0.25)

    yt = y_in + B21 * h * dy  # First step.
    NaNfound = np.any(np.isnan(yt))
    if NaNfound:
        Flag = -1
        return np.zeros_like(y_in), np.zeros_like(y_in), Flag

    try:
        ak2, z_pos, eqn_flag = fwet(yt, Params, Met, z_pos, GasThrust, threeD)  # Second step.
    except:
        eqn_flag = -1
        return np.zeros_like(y_in), np.zeros_like(y_in), eqn_flag

    yt = y_in + h * (B31 * dy + B32 * ak2)
    NaNfound = np.any(np.isnan(yt))
    if NaNfound:
        Flag = -1
        return np.zeros_like(y_in), np.zeros_like(y_in), Flag

    try:
        ak3, z_pos, eqn_flag = fwet(yt, Params, Met, z_pos, GasThrust, threeD)  # Third step.
    except:
        eqn_flag = -1
        return np.zeros_like(y_in), np.zeros_like(y_in), eqn_flag

    yt = y_in + h * (B41 * dy + B42 * ak2 + B43 * ak3)
    NaNfound = np.any(np.isnan(yt))
    if NaNfound:
        Flag = -1
        return np.zeros_like(y_in), np.zeros_like(y_in), Flag

    try:
        ak4, z_pos, eqn_flag = fwet(yt, Params, Met, z_pos, GasThrust, threeD)  # Fourth step.
    except:
        eqn_flag = -1
        return np.zeros_like(y_in), np.zeros_like(y_in), eqn_flag

    yt = y_in + h * (B51 * dy + B52 * ak2 + B53 * ak3 + B54 * ak4)
    NaNfound = np.any(np.isnan(yt))
    if NaNfound:
        Flag = -1
        return np.zeros_like(y_in), np.zeros_like(y_in), Flag

    try:
        ak5, z_pos, eqn_flag = fwet(yt, Params, Met, z_pos, GasThrust, threeD)  # Fifth step.
    except:
        eqn_flag = -1
        return np.zeros_like(y_in), np.zeros_like(y_in), eqn_flag

    yt = y_in + h * (B61 * dy + B62 * ak2 + B63 * ak3 + B64 * ak4 + B65 * ak5)
    NaNfound = np.any(np.isnan(yt))
    if NaNfound:
        Flag = -1
        return np.zeros_like(y_in), np.zeros_like(y_in), Flag

    try:
        ak6, z_pos, eqn_flag = fwet(yt, Params, Met, z_pos, GasThrust, threeD)  # Sixth step.
    except:
        eqn_flag = -1
        return np.zeros_like(y_in), np.zeros_like(y_in), eqn_flag

    y_out = y_in + h * (
        C1 * dy + C3 * ak3 + C4 * ak4 + C6 * ak6
    )  # Accumulate increments with proper weights.
    y_err = h * (DC1 * dy + DC3 * ak3 + DC4 * ak4 + DC5 * ak5 + DC6 * ak6)

    Flag = 1
    
    return y_out, y_err, Flag


@njit(cache=True)
def rkstep_wet(
    y: np.ndarray,
    dy: np.ndarray,
    h_in: np.float64,
    Params: np.ndarray,
    SolParams: np.ndarray,
    Met: np.ndarray,
    z_pos: np.int64,
    GasThrust: np.int64,
):

    threeD = SolParams["threeD"][0]

    solverFlag = 1
    h = np.float64(h_in)
    while True:
        yt, y_err, SolverFlag = rkCK_wet(y, dy, h, Params, Met, z_pos, GasThrust, threeD)
        if SolverFlag < 0:
            return z_pos, np.zeros_like(y), h, h, solverFlag

        NaNfound = np.any(np.isnan(yt))
        if NaNfound:
            solverFlag = 0
            return z_pos, np.zeros_like(y), h, h, solverFlag
        else:
            y_scal = SolParams['atol'][0] + SolParams['rtol'][0] * np.maximum(np.abs(y), np.abs(yt))
            errmax = np.sqrt(np.sum((y_err/y_scal)**2)/len(y_err))

            #y_scal = np.zeros_like(y)
            #errmax = np.float64(0.0)
            #for j in range(len(y)):
            #    y_scal[j,0] = SolParams['rtol'] * (np.float64(1.0) + np.max(np.abs(y[j,0]), np.abs(yt[j,0])))
            #    errmax = errmax + (y_err[j,0]/y_scal[j,0])**2
            #errmax = np.max(np.abs(y_err/y_scal))
            #errmax = np.sqrt(errmax/len(y_err))
            #errmax = np.max(np.abs(y_err / y_scal)) / SolParams["rtol"][0]

        if errmax <= 1:
            break

        htemp = np.float64(0.9) * h * (np.power(errmax, -0.25))
        htemp = np.maximum(np.abs(htemp), np.float64(0.1) * np.abs(h))
        h = htemp * np.sign(h)

    if np.abs(h) < SolParams["MinStep"][0]:
        solverFlag = -1

    h_out = np.float64(h)

    if errmax > np.float64(1.89e-4):
        h_next = np.float64(0.9) * h * (np.power(errmax, -0.2))
    else:
        h_next = np.float64(5.0) * h  # Numerical Recipes criteria

    y_out = yt
    
    h_next = np.minimum(h_next, SolParams["MaxStep"][0])

    return z_pos, y_out, np.float64(h_out), np.float64(h_next), solverFlag
