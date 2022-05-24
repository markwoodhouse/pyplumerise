from functools import cache

import numpy as np
from numba import njit
from numba.types import float64, int64, string

# from .IC import ICSet
from .ice_eqns import fice
from .MetDataSet import Met_interp

# from .ParamSet import ParamsSet
# from .SolverParams import SolverSet


@njit(cache=True)
def rkCK(
    y_in: np.ndarray,
    dy: np.ndarray,
    h: float64,
    IC: np.ndarray,
    Params: np.ndarray,
    Met: np.ndarray,
    z_pos: int64,
    GasThrust: int64,
):

    A2 = 0.2
    A3 = 0.3
    A4 = 0.6
    A5 = 1.0
    A6 = 0.875
    B21 = 0.2
    B31 = 3.0 / 40.0
    B32 = 9.0 / 40.0
    B41 = 0.3
    B42 = -0.9
    B43 = 1.2
    B51 = -11.0 / 54.0
    B52 = 2.5
    B53 = -70.0 / 27.0
    B54 = 35.0 / 27.0
    B61 = 1631.0 / 55296.0
    B62 = 175.0 / 512.0
    B63 = 575.0 / 13824.0
    B64 = 44275.0 / 110592.0
    B65 = 253.0 / 4096.0
    C1 = 37.0 / 378.0
    C2 = 0.0
    C3 = 250.0 / 621.0
    C4 = 125.0 / 594.0
    C5 = 0.0
    C6 = 512.0 / 1771.0
    DC1 = C1 - 2825.0 / 27648.0
    DC2 = 0.0
    DC3 = C3 - 18575.0 / 48384.0
    DC4 = C4 - 13525.0 / 55296.0
    DC5 = -277.0 / 14336.0
    DC6 = C6 - 0.25

    yt = y_in + B21 * h * dy  # First step.
    NaNfound = np.any(np.isnan(yt))
    if NaNfound:
        Flag = -1
        return np.zeros_like(y_in), np.zeros_like(y_in), Flag

    ak2, z_pos, eqn_flag = fice(yt, IC, Params, Met, z_pos, GasThrust)  # Second step.

    if eqn_flag:
        Flag = -1
        return np.zeros_like(y_in), np.zeros_like(y_in), Flag

    yt = y_in + h * (B31 * dy + B32 * ak2)
    NaNfound = np.any(np.isnan(yt))
    if NaNfound:
        Flag = -1
        return np.zeros_like(y_in), np.zeros_like(y_in), Flag

    ak3, z_pos, eqn_flag = fice(yt, IC, Params, Met, z_pos, GasThrust)  # Third step.

    if eqn_flag:
        Flag = -1
        return np.zeros_like(y_in), np.zeros_like(y_in), Flag

    yt = y_in + h * (B41 * dy + B42 * ak2 + B43 * ak3)
    NaNfound = np.any(np.isnan(yt))
    if NaNfound:
        Flag = -1
        return np.zeros_like(y_in), np.zeros_like(y_in), Flag

    ak4, z_pos, eqn_flag = fice(yt, IC, Params, Met, z_pos, GasThrust)  # Fourth step.

    if eqn_flag:
        Flag = -1
        return np.zeros_like(y_in), np.zeros_like(y_in), Flag

    yt = y_in + h * (B51 * dy + B52 * ak2 + B53 * ak3 + B54 * ak4)
    NaNfound = np.any(np.isnan(yt))
    if NaNfound:
        Flag = -1
        return np.zeros_like(y_in), np.zeros_like(y_in), Flag

    ak5, z_pos, eqn_flag = fice(yt, IC, Params, Met, z_pos, GasThrust)  # Fifth step.

    if eqn_flag:
        Flag = -1
        return np.zeros_like(y_in), np.zeros_like(y_in), Flag

    yt = y_in + h * (B61 * dy + B62 * ak2 + B63 * ak3 + B64 * ak4 + B65 * ak5)
    NaNfound = np.any(np.isnan(yt))
    if NaNfound:
        Flag = -1
        return np.zeros_like(y_in), np.zeros_like(y_in), Flag

    ak6, z_pos, eqn_flag = fice(yt, IC, Params, Met, z_pos, GasThrust)  # Sixth step.

    if eqn_flag:
        Flag = -1
        return np.zeros_like(y_in), np.zeros_like(y_in), Flag

    y_out = y_in + h * (
        C1 * dy + C3 * ak3 + C4 * ak4 + C6 * ak6
    )  # Accumulate increments with proper weights.
    y_err = h * (DC1 * dy + DC3 * ak3 + DC4 * ak4 + DC5 * ak5 + DC6 * ak6)

    Flag = 1

    return y_out, y_err, Flag


@njit(cache=True)
def rkstep(
    y: np.ndarray,
    dy: np.ndarray,
    h_in: float64,
    IC: np.ndarray,
    Params: np.ndarray,
    SolParams: np.ndarray,
    Met: np.ndarray,
    z_pos: int64,
    GasThrust: int64,
):

    solverFlag = 1
    h = h_in
    while True:
        yt, y_err, SolverFlag = rkCK(y, dy, h, IC, Params, Met, z_pos, GasThrust)
        if SolverFlag < 0:
            return z_pos, np.zeros_like(y), h, h, solverFlag

        NaNfound = np.any(np.isnan(yt))
        if NaNfound:
            solverFlag = 0
            return z_pos, np.zeros_like(y), h, h, solverFlag
        else:
            y_scal = np.abs(yt) + np.abs(h * dy) + np.finfo(np.float64).eps
            errmax = np.max(y_err / y_scal) / SolParams["rtol"][0]

        if errmax <= 1:
            break

        htemp = 0.9 * h * (np.power(errmax, -0.25))
        htemp = np.maximum(np.abs(htemp), 0.1 * np.abs(h))
        h = np.abs(htemp) * np.sign(h)

    if np.abs(h) < SolParams["MinStep"][0]:
        solverFlag = -1

    h_out = h

    if errmax > 1.89e-4:
        h_next = 0.9 * h / (np.power(errmax, 0.2))
    else:
        h_next = 5.0 * h  # Numerical Recipes criteria

    y_out = yt
    for j in range(10):
        if y_out[j] < 0.0:
            y_out[j] = 0.0

    h_next = np.minimum(h_next, SolParams["MaxStep"][0])

    return z_pos, y_out, h_out, h_next, solverFlag
