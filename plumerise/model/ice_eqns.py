from functools import cache

import numpy as np
from numba import njit
from numba.types import float64, int64, string

# from .IC import ICSet
from .MetDataSet import Met_interp, SpecHumidity, esfunc, eswfunc

# from .ParamSet import ParamsSet
# from .SolverParams import SolverSet


@njit(cache=True)
def fice(
    y: np.ndarray,
    IC: np.ndarray,
    Params: np.ndarray,
    Met: np.ndarray,
    z_pos: int64,
    GasThrust: int64,
):
    """fice -- the right hand side of the odes including water and ice formation

    Input:
        y [array] : the dependent variables
        IC [dict] : initial conditions
        Params [dict] : model parameters
        Met [float[:,:]] : met data
        z_pos [integer] : index of the z position in met data
        GasThrust [integer] : flag for the gas thrust region

    Returns:
        dy [array] : the derivatives of the dependent variables
        z_pos [integer] :: index of the z position in met data

    Raises:
        RuntimeError if there is an error

    """

    n0 = IC.n0
    Q0 = IC.Q0

    g = Params["g"][0]
    Ca = Params["Ca"][0]
    Cv = Params["Cv"][0]
    Cw = Params["Cw"][0]
    Cs = Params["Cs"][0]
    Ci = Params["Ci"][0]
    Ra = Params["Ra"][0]
    Rv = Params["Rv"][0]
    rhos = Params["rhos"][0]
    rhow = Params["rhow"][0]
    rhoi = Params["rhoi"][0]
    Lv0 = Params["Lv0"][0]
    Lm0 = Params["Lm0"][0]

    Ti0 = Params["Ti0"][0]
    Ti00 = Params["Ti00"][0]

    ks = Params["ks"][0]
    kw = Params["kw"][0]

    Nsize = 15

    A = np.zeros((Nsize, Nsize), dtype=np.float64)
    B = np.zeros((Nsize, 1), dtype=np.float64)

    Q = y[0, 0]
    M = y[1, 0]
    phia = y[2, 0]
    phiv = y[3, 0]
    phis = y[4, 0]
    phiw = y[5, 0]
    phii = y[6, 0]
    Cp = y[7, 0]
    rho = y[8, 0]
    T = y[9, 0]
    theta = y[10, 0]
    psi = y[11, 0]
    x_lon = y[12, 0]
    y_lat = y[13, 0]
    z = y[14, 0]

    MetVals, z_pos = Met_interp(Met, z, z_pos)

    Pa = MetVals["Pa"]
    dPadz = MetVals["dPadz"]

    Ta = MetVals["Ta"]
    dTadz = MetVals["dTadz"]

    esa, desadT = eswfunc(Ta)  # saturation vapor pressure in atmosphere
    RH = MetVals["RH"]  # relative humidity in the atmosphere

    phivA = SpecHumidity(RH, esa, Pa, Ra, Rv)
    phiaA = 1.0 - phivA

    Catm = phiaA * Ca + phivA * Cv

    # atmospheric density
    RgA = phivA * Rv + phiaA * Ra
    rhoAtm = Pa / RgA / Ta

    V = MetVals["V"]  # Wind speed
    psiA = MetVals["psia"]  # Wind direction

    GT = 0
    # Check if we're still in the gas thrust region
    if GasThrust == 1:
        GT = 1 * (rho > rhoAtm)

    # Change entrainment coefficient if in Gas thrust region
    if (GasThrust == 1) and (GT == 1):
        ks_var = np.sqrt(rho / rhoAtm) / 16.0
    else:
        ks_var = ks

    Ue = ks_var * np.abs(M / Q - V * np.cos(theta)) + kw * np.abs(
        V * np.sin(theta)
    )  # Entrainment velocity from Hewett, Fay & Hoult (1971)
    # match Params.Entrainment:
    #     case 'Hewett':
    #         Ue = ks_var*np.abs(M/Q - V*np.cos(theta)) + kw*np.abs(V*np.sin(theta)) # Entrainment velocity from Hewett, Fay & Hoult (1971)
    #     case 'Ooms':
    #         Ue = ks_var*np.abs(M/Q - V*np.cos(theta)) + kw*V*np.cos(theta)*np.abs(np.sin(theta)) # Entrainment velocity from Ooms & Mahieu (1981)
    #     case 'Contini':
    #         Ue = ks_var*np.abs(M/Q - V*np.cos(theta)) + kw*np.abs(M/Q*np.sin(theta))*np.cos(theta) # Entrainment velocity from Contini (2011)
    #     case 'Devenish':
    #         Ue = np.power( np.power(ks_var*np.abs(M/Q - V*np.cos(theta)),Params.Devenish_m) + np.power(kw*np.abs(V*np.sin(theta)),Params.Devenish_m), 1.0/Params.Devenish_m) # Entrainment velocity from Devenish
    #     case _:
    #         Ue = ks_var*np.abs(M/Q - V*np.cos(theta)) + kw*np.abs(V*np.sin(theta)) # Entrainment velocity defaults to Hewett

    es, desdT = esfunc(T, Ti0, Ti00)  # saturation vapour pressure

    Rg = (phiv * Rv + phia * Ra) / (phia + phiv)  # Plume gas constant

    rhoa = Pa / Ra / T
    rhov = Pa / Rv / T

    Pv = phiv * Rv / (phia * Ra + phiv * Rv) * Pa  # partial pressure of water vapour

    if Pv < es:
        Condense = 0  # unsaturated
    else:
        Condense = 1  # saturated

    if T > Ti0:
        IceType = 0
    elif T < Ti00:
        IceType = 2
    else:
        IceType = 1

    if Condense == 0:  # unsaturated
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 0] = -(phiaA - phia) / Q
        A[2, 2] = 1.0
        A[3, 0] = -(phivA - phiv) / Q
        A[3, 3] = 1.0
        A[4, 0] = phis / Q
        A[4, 4] = 1.0
        if IceType == 1:  # Mixed phase
            A[5, 0] = (Ti0 - Ti00) / (T - Ti00) * phiw / Q
            A[5, 5] = (Ti0 - Ti00) / (T - Ti00)
            A[5, 9] = -(Ti0 - Ti00) / (T - Ti00) / (T - Ti00) * phiw
            A[6, 0] = phii / Q - (Ti0 - T) / (T - Ti00) * phiw / Q
            A[6, 5] = -(Ti0 - T) / (T - Ti00)
            A[6, 6] = 1.0
            A[6, 9] = (Ti0 - Ti00) / (T - Ti00) / (T - Ti00) * phiw
        else:  # no ice, or all ice
            A[5, 0] = phiw / Q
            A[5, 5] = 1.0
            A[6, 0] = phii / Q
            A[6, 6] = 1.0
        A[7, 2] = -Ca
        A[7, 3] = -Cv
        A[7, 4] = -Cs
        A[7, 5] = -Cw
        A[7, 6] = -Ci
        A[7, 7] = 1.0
        A[8, 2] = rho * rho / rhoa
        A[8, 3] = rho * rho / rhov
        A[8, 4] = rho * rho / rhos
        A[8, 5] = rho * rho / rhow
        A[8, 6] = rho * rho / rhoi
        A[8, 8] = 1.0
        A[8, 9] = (
            rho
            * rho
            * (phia / rhoa / rhoa / Ra / T + phiv / rhov / rhov / Rv / T)
            * Pa
            / T
        )
        A[8, 14] = (
            -rho
            * rho
            * (phia / rhoa / rhoa / Ra / T + phiv / rhov / rhov / Rv / T)
            * dPadz
        )

        if IceType == 1:  # Mixed
            A[9, 0] = (
                (Cp * T - Catm * Ta) / Q / Cp
                - (Ue * Ue + M * M / Q / Q) / 2.0 / Q / Cp
                - Lm0 * (Ti0 - T) / (T - Ti00) * phiw / Q / Cp
            )
            A[9, 1] = M / Cp / Q / Q
            A[9, 5] = -Lm0 / Cp * (Ti0 - T) / (T - Ti00)
            A[9, 7] = T / Cp
            A[9, 9] = 1.0 + Lm0 * phiw / Cp * (Ti0 - Ti00) / (T - Ti00) / (T - Ti00)
            A[9, 14] = g / Cp
        else:  # no ice or all ice
            A[9, 0] = (Cp * T - Catm * Ta) / Q / Cp - (
                Ue * Ue + M * M / Q / Q
            ) / 2.0 / Q / Cp
            A[9, 1] = M / Cp / Q / Q
            A[9, 7] = T / Cp
            A[9, 9] = 1.0
            A[9, 14] = g / Cp

        A[10, 10] = 1.0
        A[11, 11] = 1.0
        A[12, 12] = 1.0
        A[13, 13] = 1.0
        A[14, 14] = 1.0

        B[0, 0] = 2.0 * Ue * rhoAtm * Q / np.sqrt(M * rho)
        B[1, 0] = g * (rhoAtm - rho) * Q * Q / rho / M * np.sin(
            theta
        ) + 2.0 * rhoAtm * Q / np.sqrt(rho * M) * Ue * V * (
            np.cos(psi) * np.cos(psiA) + np.sin(psi) * np.sin(psiA)
        ) * np.cos(
            theta
        )
        B[2, 0] = 0.0
        B[3, 0] = 0.0
        B[4, 0] = 0.0
        B[5, 0] = 0.0
        B[6, 0] = 0.0
        B[7, 0] = 0.0
        B[8, 0] = 0.0
        B[9, 0] = 0.0
        B[10, 0] = g * (rhoAtm - rho) * Q * Q * np.cos(
            theta
        ) / rho / M / M - 2.0 * rhoAtm * Q / np.sqrt(rho * M) / M * Ue * V * (
            np.cos(psi) * np.cos(psiA) + np.sin(psi) * np.sin(psiA)
        ) * np.sin(
            theta
        )
        B[11, 0] = (
            2.0
            * rhoAtm
            * Q
            / M
            / np.sqrt(rho * M)
            * Ue
            * V
            * (np.cos(psi) * np.sin(psiA) - np.sin(psi) * np.cos(psiA))
            / np.cos(theta)
        )
        B[12, 0] = np.cos(theta) * np.cos(psi)
        B[13, 0] = np.cos(theta) * np.sin(psi)
        B[14, 0] = np.sin(theta)

    else:  # saturated
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 0] = -(phiaA - phia) / Q
        A[2, 2] = 1.0
        A[3, 2] = -Ra * es / Rv / (Pa - es)
        A[3, 3] = 1.0
        A[3, 9] = -phia * Ra * Pa / Rv / (Pa - es) / (Pa - es) * desdT
        A[3, 14] = phia * Ra * es / Rv / (Pa - es) / (Pa - es) * dPadz

        A[4, 0] = phis / Q
        A[4, 4] = 1.0
        if IceType == 0:  # No ice
            A[5, 0] = (phiw - phivA + phiv) / Q
            A[5, 2] = Ra * es / Rv / (Pa - es)
            A[5, 5] = 1.0
            A[5, 9] = phia * Ra * Pa / Rv / (Pa - es) / (Pa - es) * desdT
            A[5, 14] = -phia * Ra * es / Rv / (Pa - es) / (Pa - es) * dPadz

            A[6, 0] = phii / Q
            A[6, 6] = 1.0
        elif IceType == 1:  # Mixed phase
            A[5, 0] = (phiw - phivA + phiv + (Ti0 - Ti00) / (T - Ti00) * phiw) / Q
            A[5, 2] = Ra * es / Rv / (Pa - es)
            A[5, 5] = (Ti0 - Ti00) / (T - Ti00)
            A[5, 9] = (
                phia * Ra * Pa / Rv / (Pa - es) / (Pa - es) * desdT
                - (Ti0 - Ti00) / (T - Ti00) / (T - Ti00) * phiw
            )
            A[5, 14] = -phia * Ra * es / Rv / (Pa - es) / (Pa - es) * dPadz

            A[6, 0] = (phii - (Ti0 - T) / (T - Ti00) * phiw) / Q
            A[6, 5] = -(Ti0 - T) / (T - Ti00)
            A[6, 6] = 1.0
            A[6, 9] = phiw * (Ti0 - Ti00) / (T - Ti00) / (T - Ti00)
        else:  # all ice
            A[5, 0] = phiw / Q
            A[5, 5] = 1.0
            A[6, 0] = phii / Q
            A[6, 2] = Ra * es / Rv / (Pa - es)
            A[6, 6] = 1.0
            A[6, 9] = phia * Ra * Pa / Rv / (Pa - es) / (Pa - es) * desdT
            A[6, 14] = -phia * Ra * es / Rv / (Pa - es) / (Pa - es) * dPadz
        A[7, 2] = -Ca
        A[7, 3] = -Cv
        A[7, 4] = -Cs
        A[7, 5] = -Cw
        A[7, 6] = -Ci
        A[7, 7] = 1.0
        A[8, 2] = rho * rho / rhoa
        A[8, 3] = rho * rho / rhov
        A[8, 4] = rho * rho / rhos
        A[8, 5] = rho * rho / rhow
        A[8, 6] = rho * rho / rhoi
        A[8, 8] = 1.0
        A[8, 9] = (
            rho
            * rho
            * (phia / rhoa / rhoa / Ra / T + phiv / rhov / rhov / Rv / T)
            * Pa
            / T
        )
        A[8, 14] = (
            -rho
            * rho
            * (phia / rhoa / rhoa / Ra / T + phiv / rhov / rhov / Rv / T)
            * dPadz
        )

        if IceType == 0:  # No ice
            A[9, 0] = (
                (Cp * T - Catm * Ta) / Q / Cp
                - (Ue * Ue + M * M / Q / Q) / 2.0 / Q / Cp
                - Lv0 / Cp * (phivA - phiv) / Q
            )
            A[9, 1] = M / Cp / Q / Q
            A[9, 2] = Lv0 / Cp * Ra * es / Rv / (Pa - es)
            A[9, 7] = T / Cp
            A[9, 9] = (
                1.0 + Lv0 / Cp * phia * Ra * Pa / Rv / (Pa - es) / (Pa - es) * desdT
            )
            A[9, 14] = (
                g / Cp - Lv0 / Cp * phia * Ra * es / Rv / (Pa - es) / (Pa - es) * dPadz
            )
        elif IceType == 1:  # Mixed
            A[9, 0] = (
                (Cp * T - Catm * Ta) / Q / Cp
                - (Ue * Ue + M * M / Q / Q) / 2.0 / Q / Cp
                - Lv0 / Cp * (phivA - phiv) / Q
                - Lm0 / Cp * (Ti0 - T) / (T - Ti00) * phiw / Q
            )
            A[9, 1] = M / Cp / Q / Q
            A[9, 2] = Lv0 / Cp * Ra * es / Rv / (Pa - es)
            A[9, 5] = -Lm0 / Cp * (Ti0 - T) / (T - Ti00)
            A[9, 7] = T / Cp
            A[9, 9] = (
                1.0
                + Lv0 / Cp * phia * Ra * Pa / Rv / (Pa - es) / (Pa - es) * desdT
                + Lm0 / Cp * (Ti0 - Ti00) / (T - Ti00) / (T - Ti00) * phiw
            )
            A[9, 14] = (
                g / Cp - Lv0 / Cp * phia * Ra * es / Rv / (Pa - es) / (Pa - es) * dPadz
            )
        else:  # all ice
            A[9, 0] = (
                (Cp * T - Catm * Ta) / Q / Cp
                - (Ue * Ue + M * M / Q / Q) / 2.0 / Q / Cp
                - (Lv0 + Lm0) / Cp * (phivA - phiv) / Q
            )
            A[9, 1] = M / Cp / Q / Q
            A[9, 2] = (Lv0 + Lm0) / Cp * Ra * es / Rv / (Pa - es)
            A[9, 7] = T / Cp
            A[9, 9] = (
                1.0
                + (Lv0 + Lm0) / Cp * phia * Ra * Pa / Rv / (Pa - es) / (Pa - es) * desdT
            )
            A[9, 14] = (
                g / Cp
                - (Lv0 + Lm0) / Cp * phia * Ra * es / Rv / (Pa - es) / (Pa - es) * dPadz
            )
        A[10, 10] = 1.0
        A[11, 11] = 1.0
        A[12, 12] = 1.0
        A[13, 13] = 1.0
        A[14, 14] = 1.0

        B[0, 0] = 2.0 * Ue * rhoAtm * Q / np.sqrt(M * rho)
        B[1, 0] = g * (rhoAtm - rho) * Q * Q / rho / M * np.sin(
            theta
        ) + 2.0 * rhoAtm * Q / np.sqrt(rho * M) * Ue * V * (
            np.cos(psi) * np.cos(psiA) + np.sin(psi) * np.sin(psiA)
        ) * np.cos(
            theta
        )
        B[2, 0] = 0.0
        B[3, 0] = 0.0
        B[4, 0] = 0.0
        B[5, 0] = 0.0
        B[6, 0] = 0.0
        B[7, 0] = 0.0
        B[8, 0] = 0.0
        B[9, 0] = 0.0
        B[10, 0] = g * (rhoAtm - rho) * Q * Q * np.cos(
            theta
        ) / rho / M / M - 2.0 * rhoAtm * Q / np.sqrt(rho * M) / M * Ue * V * (
            np.cos(psi) * np.cos(psiA) + np.sin(psi) * np.sin(psiA)
        ) * np.sin(
            theta
        )
        B[11, 0] = (
            2.0
            * rhoAtm
            * Q
            / M
            / np.sqrt(rho * M)
            * Ue
            * V
            * (np.cos(psi) * np.sin(psiA) - np.sin(psi) * np.cos(psiA))
            / np.cos(theta)
        )
        B[12, 0] = np.cos(theta) * np.cos(psi)
        B[13, 0] = np.cos(theta) * np.sin(psi)
        B[14, 0] = np.sin(theta)

    Flag = 0
    try:
        dy = np.linalg.solve(A, B)
    except:
        dy = np.zeros_like(B)
        Flag = 1

    return dy, z_pos, Flag


@njit(cache=True)
def CheckStep(
    s: float64,
    y: np.ndarray,
    Params: np.ndarray,
    Met: np.ndarray,
    SolParams: np.ndarray,
    z_pos: int64,
    GasThrust: int64,
):

    # Flag = 0 if step is ok, and can continue integration
    # Flag = 1 if step is ok but s => sMax
    # Flag = 2 if step is ok but stop condition has been satisfied
    # Flag < 0 if something is wrong :
    # Flag = -1 if Q<0.
    # Flag = -2 is E<0.
    # Flag = -3 something else went wrong

    Ca = Params["Ca"][0]
    Cv = Params["Cv"][0]
    Ra = Params["Ra"][0]
    Rv = Params["Rv"][0]

    Ti0 = Params["Ti0"][0]
    Ti00 = Params["Ti00"][0]

    Q = y[0, 0]
    M = y[1, 0]
    phia = y[2, 0]
    phiv = y[3, 0]
    Cp = y[7, 0]
    rho = y[8, 0]
    T = y[9, 0]
    theta = y[10, 0]
    x_lon = y[12, 0]
    y_lat = y[13, 0]
    z = y[14, 0]

    MetVals, z_pos = Met_interp(Met, z, z_pos)

    Pa = MetVals["Pa"]

    E = Cp * T * Q

    Ta = MetVals["Ta"]
    esa, _ = eswfunc(Ta)  # saturation vapor pressure in atmosphere
    RH = MetVals["RH"]

    phivA = SpecHumidity(RH, esa, Pa, Ra, Rv)
    Pv = phiv * Rv * Pa / (phia * Ra + phiv * Rv)
    es, _ = esfunc(T, Ti0, Ti00)
    if Pv > es:
        Condense = 1
    else:
        Condense = 0

    phiaA = 1.0 - phivA

    # atmospheric density
    RgA = phiaA * Ra + phivA * Rv
    rhoAtm = Pa / RgA / Ta

    # First check that Q>0 and E>0
    if Q < 0:
        Flag = -1
        return z_pos, GasThrust, Condense, Flag

    if E < 0:
        Flag = -2
        return z_pos, GasThrust, Condense, Flag

    # Are we still in Gas Thrust region?
    if GasThrust == 1:
        GasThrust = 1 * (rho > rhoAtm)

    # Now check if s<sMax
    if s >= SolParams["sMax"][0]:
        Flag = 1
        return z_pos, GasThrust, Condense, Flag

    # Check if U<Umin
    U = M / Q
    Uz = U * np.sin(theta)

    if np.abs(Uz - SolParams["Umin"][0]) <= 2.0 * np.finfo(np.float64).eps:
        Flag = 2
        return z_pos, GasThrust, Condense, Flag

    # match SolParams.StopCond:
    #     case 'Top':
    #         # Check if U<Umin
    #         U = M/Q
    #         Uz = U*np.sin(theta)

    #         if (Uz-SolParams.Umin <= 2.0*np.finfo(np.float64).eps):
    #             Flag = 2
    #             return z_pos, GasThrust, Condense, Flag

    #     case 'HorizDistance':

    #         HorizDistance = np.sqrt(x_lon*x_lon+y_lat*y_lat)

    #         if (HorizDistance>=SolParams.MaxHorizDistance):
    #             Flag=2
    #             return z_pos, GasThrust, Condense, Flag

    #     case 'Neutral':

    #         if (GasThrust==0):
    #             if (rho>=rhoAtm):
    #                 Flag=2
    #                 return z_pos, GasThrust, Condense, Flag

    Flag = 0

    return z_pos, GasThrust, Condense, Flag
