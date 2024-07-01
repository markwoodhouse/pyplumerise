from functools import cache

import numpy as np
from numba import njit
from numba.types import float64, int64, string

from ..MetDataSet import Met_interp, SpecHumidity, esfunc, eswfunc

# from .ParamSet import ParamsSet
# from .SolverParams import SolverSet


@njit(cache=True)
def fice(
    y: np.ndarray,
    Params: np.ndarray,
    Met: np.ndarray,
    z_pos: np.int64,
    GasThrust: np.int64,
    threeD: bool
):
    """fice -- the right hand side of the odes including water and ice formation

    Input:
        y [array] : the dependent variables
        Params [dict] : model parameters
        Met [float[:,:]] : met data
        z_pos [integer] : index of the z position in met data
        GasThrust [integer] : flag for the gas thrust region
        threeD [bool] : does the plume rotate with wind?

    Returns:
        dy [array] : the derivatives of the dependent variables
        z_pos [integer] :: index of the z position in met data

    Raises:
        RuntimeError if there is an error

    """

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

    if threeD:
        Nsize = 15
    else:
        Nsize = 12

    A = np.zeros((Nsize, Nsize), dtype=np.float64)
    B = np.zeros((Nsize, 1), dtype=np.float64)

    logQ = y[0, 0]
    logM = y[1, 0]
    phia = y[2, 0]
    phiv = y[3, 0]
    phis = y[4, 0]
    phiw = y[5, 0]
    phii = y[6, 0]
    Cp = y[7, 0]
    rho = y[8, 0]
    T = y[9, 0]
    theta = y[10, 0]
    z = y[11, 0]
    if threeD:
        psi = y[12, 0]
        #x_lon = y[13, 0]
        #y_lat = y[14, 0]
    else:
        psi = np.float64(0.0)

    Q = np.exp(logQ)
    M = np.exp(logM)

    sqrt_rhoM = np.sqrt(rho*M)
    U = np.exp(logM-logQ)
    U2 = U*U

    #if phiw<np.finfo(float64).eps:
    #    phiw = np.float64(0.0)
    #if phii<np.finfo(float64).eps:
    #    phii = np.float64(0.0)
    
    MetVals, z_pos = Met_interp(Met, z, z_pos)

    Pa = MetVals["Pa"]
    dPadz = MetVals["dPadz"]

    Ta = MetVals["Ta"]

    esa, _ = eswfunc(Ta)  # saturation vapor pressure in atmosphere
    RH = MetVals["RH"]  # relative humidity in the atmosphere

    phivA = SpecHumidity(RH, esa, Pa, Ra, Rv)
    phiaA = np.float64(1.0) - phivA

    Catm = phiaA * Ca + phivA * Cv

    # atmospheric density
    RgA = phivA * Rv + phiaA * Ra
    rhoAtm = Pa / RgA / Ta

    V = MetVals["V"]  # Wind speed
    if threeD:
        psiA = MetVals["psia"]  # Wind direction
    else:
        psiA = np.float64(0.0)

    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    cos_psiA = np.cos(psiA)
    sin_psiA = np.sin(psiA)
    cos_psiA_psi = np.cos(psiA - psi)
    sin_psiA_psi = np.sin(psiA - psi)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    GT = 0
    # Check if we're still in the gas thrust region
    if GasThrust == 1:
        GT = 1 * (rho > rhoAtm)

    # Change entrainment coefficient if in Gas thrust region
    if (GasThrust == 1) and (GT == 1):
        ks_var = np.sqrt(rho / rhoAtm) / np.float64(16.0)
    else:
        ks_var = ks

    Ue = ks_var * np.abs(U - V * cos_theta) + kw * np.abs(V * sin_theta)
    # Entrainment velocity from Hewett, Fay & Hoult (1971)
    
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

    rho2 = rho * rho

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

    # Bulk mass conservation
    # A[0,0]*d(logQ)/ds = B[0,0]
    A[0, 0] = np.float64(1.0)
    B[0, 0] = np.float64(2.0) * Ue * rhoAtm / sqrt_rhoM

    # Bulk momentum conservation
    # A[1,1]*d(logM)/ds + A[1,10]*d(theta)/ds = B[1,0]
    #A[1, 0] = -V * (cos_psi * cos_psiA + sin_psi * sin_psiA) * cos_theta / U
    #A[1, 1] = np.float64(1.0)
    #B[1, 0] = g * (rhoAtm - rho) * sin_theta / rho / U2
    A[1,  1] = sin_theta
    A[1, 10] = cos_theta
    B[1, 0] = g * (rhoAtm - rho) / (rho * U2)

    # Dry air mass conservation
    # A[2,0]*d(logQ)/ds + d(phia)/ds = 0
    A[2, 0] = -(phiaA - phia)
    A[2, 2] = np.float64(1.0)
    B[2, 0] = np.float64(0.0)

    # Solids mass conservation
    # A[4,0]*d(logQ)/ds + d(phis)/ds = 0
    A[4, 0] = phis
    A[4, 4] = np.float64(1.0)
    B[4, 0] = np.float64(0.0)

    # Bulk heat capacity
    # A[7,2]*d(phia)/ds + A[7,3]*d(phiv)/ds + A[7,4]*d(phis)/ds + A[7,5]*d(phiw)/ds 
    #   + A[7,6]*d(phii)/ds + A[6,7]*d(Cp)/ds = 0
    A[7, 2] = -Ca
    A[7, 3] = -Cv
    A[7, 4] = -Cs
    A[7, 5] = -Cw
    A[7, 6] = -Ci
    A[7, 7] = np.float64(1.0)
    B[7, 0] = np.float64(0.0)

    # Bulk density
    # A[8,2]*d(phia)/ds + A[8,3]*d(phiv)/ds + A[8,4]*d(phis)/ds + A[8,5]*d(phiw)/ds 
    #   + A[8,6]*d(phii)/ds + A[8,8]*d(rho)/ds + A[8,9]*dT/ds + A[8,14]*dz/dz = 0
    A[8, 2] = rho2 / rhoa
    A[8, 3] = rho2 / rhov
    A[8, 4] = rho2 / rhos
    A[8, 5] = rho2 / rhow
    A[8, 6] = rho2 / rhoi
    A[8, 8] = np.float64(1.0)
    A[8, 9] = (
        rho2
        * (phia / (rhoa * rhoa * Ra * T) + phiv / (rhov * rhov * Rv * T))
        * Pa / T
    )
    A[8, 11] = (
        -rho2
        * (phia / (rhoa * rhoa * Ra * T) + phiv / (rhov * rhov * Rv * T))
        * dPadz
    )
    B[8, 0] = np.float64(0.0)

    if Condense == 0:  # unsaturated
        
        # Dry air mass conservation
        # A[2,0]*d(logQ)/ds + d(phia)/ds = 0
        A[2, 0] = -(phiaA - phia)
        A[2, 2] = np.float64(1.0)
        B[2, 0] = np.float64(0.0)

        # Water vapour mass conservation
        # A[3,0]*d(logQ)/ds + d(phiv)/ds = 0
        A[3, 0] = -(phivA - phiv)
        A[3, 3] = np.float64(1.0)
        B[3, 0] = np.float64(0.0)

        # Liquid water mass conservation
        # A[5,0]*d(logQ/ds) + A[5,5]*d(phiw)/ds + A[5,9]*dT/ds = 0
        if IceType==1: # Mixed phase
            A[5, 0] = (T - Ti00) * phiw
            A[5, 5] = (T - Ti00)
            A[5, 9] = -phiw
        else: # no ice, or all ice
            A[5, 0] = phiw
            A[5, 5] = np.float64(1.0)
        B[5, 0] = np.float64(0.0)

        # Ice mass conservation
        # A[6,0]*d(logQ/ds) + A[6,5]*d(phiw)/ds + A[6,6]*d(phii)/ds + A[6,9]*dT/ds = 0
        if IceType==1: # Mixed phase
            A[6, 0] = phii * (T - Ti00) * (T - Ti00) - (Ti0 - T) * (T - Ti00) * phiw
            A[6, 5] = -(Ti0 - T) * (T - Ti00)
            A[6, 6] = (T - Ti00) * (T - Ti00)
            A[6, 9] = (Ti0 - Ti00) * phiw
        else:  # no ice, or all ice
            A[6, 0] = phii
            A[6, 6] = np.float64(1.0)
        B[6, 0] = np.float64(0.0)

        # Bulk energy conservation
        # A[9,0]*d(logQ)/ds + A[9,1]*d(logM)/ds + A[9,5]*d(phiw)/ds + A[9,7]*d(Cp)/ds
        #  + A[9,9]*dT/ds + A[9,14]*dz/ds = 0
        if IceType == 1:  # Mixed
            A[9, 0] = (
                (Cp * T - Catm * Ta) * (T - Ti00) * (T - Ti00)
                - np.float64(0.5) * (Ue * Ue + U2) * (T - Ti00) * (T - Ti00)
                - Lm0 * (Ti0 - T) * (T - Ti00) * phiw
            )
            A[9, 1] = U2  * (T - Ti00) * (T - Ti00)
            A[9, 5] = -Lm0 * (Ti0 - T) * (T - Ti00)
            A[9, 7] = T * (T - Ti00) * (T - Ti00)
            A[9, 9] = (T - Ti00) * (T - Ti00) * Cp + Lm0 * phiw * (Ti0 - Ti00)
            A[9, 11] = (T - Ti00) * (T - Ti00) * g
        else:  # no ice or all ice
            A[9, 0] = (Cp * T - Catm * Ta) - np.float64(0.5) * (Ue * Ue + U2)
            A[9, 1] = U2
            A[9, 7] = T
            A[9, 9] = Cp
            A[9, 11] = g
        B[9, 0] = np.float64(0.0)

    else:  # saturated

        # Water vapour mass conservation
        # A[3,2]*d(phia)/ds + A[3,3]*d(phiv)/ds + A[3,9]*dT/dz + A[3,14]*dz/ds = 0
        A[3, 2] = -Ra * es / Rv * (Pa - es)
        A[3, 3] = (Pa - es) * (Pa - es)
        A[3, 9] = -phia * Ra * Pa / Rv * desdT
        A[3, 11] = phia * Ra * es / Rv * dPadz
        B[3, 0] = np.float64(0.0)

        # Liquid water mass conservation
        # A[5,0]*d(logQ/ds) + A[5,2]*d(phia)/ds + A[5,5]*d(phiw)/ds
        #  + A[5,9]*dT/ds + A[5,14]*dz/ds = 0
        if IceType == 0:  # No ice
            A[5, 0] = (Pa - es) * (Pa - es) * (phiw - phivA + phiv)
            A[5, 2] = Ra * es / Rv * (Pa - es)
            A[5, 5] = (Pa - es) * (Pa - es)
            A[5, 9] = phia * Ra * Pa / Rv * desdT
            A[5, 11] = -phia * Ra * es / Rv * dPadz
        elif IceType == 1:  # Mixed phase
            A[5, 0] = (Pa - es) * (Pa - es) * (
                (T - Ti00) * (T - Ti00) * (phiw - phivA + phiv)
                + (Ti0 - Ti00) * (T - Ti00) * phiw
            )
            A[5, 2] = Ra * es / Rv * (Pa - es) * (T - Ti00) * (T - Ti00)
            A[5, 5] = (Pa - es) * (Pa - es) * (Ti0 - Ti00) * (T - Ti00)
            A[5, 9] = (
                phia * Ra * Pa / Rv * (T - Ti00) * (T - Ti00) * desdT
                - (Pa - es) * (Pa - es) * (Ti0 - Ti00) * phiw
            )
            A[5, 11] = -phia * Ra * es / Rv * (T - Ti00) * (T - Ti00) * dPadz
        else:  # all ice
            A[5, 0] = phiw
            A[5, 5] = np.float64(1.0)
        B[5 , 0] = np.float64(0.0)

        # Ice mass conservation
        # A[6,0]*d(logQ/ds) + A[6,5]*d(phiw)/ds + A[6,6]*d(phii)/ds + A[6,9]*dT/ds
        # + A[6, 14]*dz/ds = 0
        if IceType == 0:  # No ice
            A[6, 0] = phii
            A[6, 6] = np.float64(1.0)
        elif IceType == 1:  # Mixed phase
            A[6, 0] = (T - Ti00) * (T - Ti00) * phii - (Ti0 - T) * (T - Ti00) * phiw
            A[6, 5] = -(Ti0 - T) * (T - Ti00)
            A[6, 6] = (T - Ti00) * (T - Ti00)
            A[6, 9] = phiw * (Ti0 - Ti00)
        else:  # all ice
            A[6, 0] = phii * (Pa - es) * (Pa - es)
            A[6, 2] = Ra * es / Rv * (Pa - es)
            A[6, 6] = (Pa - es) * (Pa - es)
            A[6, 9] = phia * Ra * Pa / Rv * desdT
            A[6, 11] = -phia * Ra * es / Rv * dPadz
        B[6, 0] = np.float64(0.0)

        # Bulk energy conservation
        # A[9,0]*d(logQ)/ds + A[9,1]*d(logM)/ds + A[9,5]*d(phiw)/ds + A[9,7]*d(Cp)/ds
        #  + A[9,9]*dT/ds + A[9,14]*dz/ds = 0
        if IceType == 0:  # No ice
            A[9, 0] = (
                (Cp * T - Catm * Ta)
                - np.float64(0.5) * (Ue * Ue + U2)
                - Lv0 * (phivA - phiv)
            ) * (Pa - es) * (Pa - es)
            A[9, 1] = (Pa - es) * (Pa - es) * U2
            A[9, 2] = Lv0 * Ra * es / Rv * (Pa - es)
            A[9, 7] = (Pa - es) * (Pa - es) * T
            A[9, 9] = (
                (Pa - es) * (Pa - es) * Cp 
                + Lv0 * phia * Ra * Pa / Rv * desdT
            )
            A[9, 11] = (
                g * (Pa - es) * (Pa - es)
                 - Lv0 * phia * Ra * es / Rv * dPadz
            )
        elif IceType == 1:  # Mixed
            A[9, 0] = (
                ((Cp * T - Catm * Ta)
                - np.float64(0.5)*(Ue * Ue + U2)
                - Lv0 * (phivA - phiv)) * (T - Ti00) * (T - Ti00)
                - Lm0 * (Ti0 - T) * (T - Ti00) * phiw
            ) * (Pa - es) * (Pa - es)
            A[9, 1] = (Pa - es) * (Pa - es) * (T - Ti00) * (T - Ti00) * U2
            A[9, 2] = Lv0 * Ra * es / Rv * (Pa - es) * (T - Ti00) * (T - Ti00)
            A[9, 5] = -Lm0 * (Ti0 - T) * (T - Ti00) * (Pa - es) * (Pa - es)
            A[9, 7] = T * (T - Ti00) * (T - Ti00) * (Pa - es) * (Pa - es)
            A[9, 9] = (
                Cp * (T - Ti00) * (T - Ti00) * (Pa - es) * (Pa - es)
                + Lv0 * phia * Ra * Pa / Rv * (T - Ti00) * (T - Ti00) * desdT
                + Lm0 * (Ti0 - Ti00) * (Pa - es) * (Pa - es) * phiw
            )
            A[9, 11] = (
                g * (T - Ti00) * (T - Ti00) * (Pa - es) * (Pa - es)
                  - Lv0 * phia * Ra * es / Rv * (T - Ti00) * dPadz
            )
        else:  # all ice
            A[9, 0] = (
                (Cp * T - Catm * Ta)
                - np.float64(0.5) * (Ue * Ue + U2)
                - (Lv0 + Lm0) * (phivA - phiv)
            ) * (Pa - es) * (Pa - es)
            A[9, 1] = (Pa - es) * (Pa - es) * U2
            A[9, 2] = (Lv0 + Lm0) * Ra * es / Rv * (Pa - es)
            A[9, 7] = (Pa - es) * (Pa - es) * T
            A[9, 9] = (
                Cp * (Pa - es) * (Pa - es)
                + (Lv0 + Lm0) * phia * Ra * Pa / Rv * desdT
            )
            A[9, 11] = (
                g * (Pa - es) * (Pa - es)
                - (Lv0 + Lm0) * phia * Ra * es / Rv * dPadz
            )
        B[9, 0] = np.float64(0.0)

    # Plume inclination angle
    # A[10,0]*d(logQ)/ds + A[10,10]*d(theta)/ds = B[10,0]
    #A[10,  0] = (V / U) * (cos_psi * cos_psiA + sin_psi * sin_psiA) * sin_theta
    #A[10, 10] = np.float64(1.0)
    #B[10, 0] = g * ((rhoAtm - rho) / rho) * cos_theta / U2
    #A[10,  0] = (V / U) * sin_psiA
    #A[10,  1] = -cos_theta * sin_psi
    #A[10, 10] = sin_psi * sin_theta
    #A[10, 11] = -cos_psi * cos_theta
    #B[10, 0] = np.float64(0.0)
    A[10,  0] = (V / U) * cos_psiA_psi
    A[10,  1] = -cos_theta
    A[10, 10] = sin_theta
    B[10, 0] = np.float64(0.0)

    # Elevation
    # A[11,11]*dz/ds = B[11,0]
    A[11, 11] = np.float64(1.0)
    B[11, 0] = sin_theta

    if threeD:
        # Plume bearing angle
        # A[11,0]*d(logQ)/ds + A[11,11]*d(psi)/ds = 0
        #A[11, 0] = - (V / U) * (cos_psi * sin_psiA - sin_psi * cos_psiA)
        #A[11, 11] = cos_theta
        #B[11, 0] = np.float64(0.0)
        #A[11, 0] = (V / U) * cos_psiA
        #A[11, 1] = - cos_psi * cos_theta
        #A[11, 10] = cos_psi * sin_theta
        #A[11, 11] = sin_psi * cos_theta
        #B[11, 0] = np.float64(0.0)
        A[12, 0] = (V / U) * sin_psiA_psi
        A[12, 12] = -cos_theta
        B[12, 0] = np.float64(0.0)

        # Easting
        # A[13,13]*dx/ds = B[13,0]
        A[13, 13] = np.float64(1.0)
        B[13, 0] = cos_theta * cos_psi

        # Northing
        # A[14,14]*dy/ds = B[14,0]
        A[14, 14] = np.float64(1.0)
        B[14, 0] = cos_theta * sin_psi

    Flag = 0
    try:
        dy = np.linalg.solve(A, B)
    except:
        print ('bad matrices')
        dy = np.zeros_like(B)
        Flag = -1
        return dy, z_pos, Flag

    for j in range(Nsize):
        if np.abs(dy[j, 0]) < np.finfo(np.float64).eps:
            dy[j, 0] = np.float64(0.0)

    return dy, z_pos, Flag


@njit(cache=True)
def CheckStep_ice(
    s: np.float64,
    y: np.ndarray,
    Params: np.ndarray,
    Met: np.ndarray,
    SolParams: np.ndarray,
    z_pos: np.int64,
    GasThrust: np.int64,
):

    # Flag = 0 if step is ok, and can continue integration
    # Flag = 1 if step is ok but s => sMax
    # Flag = 2 if step is ok but stop condition has been satisfied
    # Flag < 0 if something is wrong :
    # Flag = -1 if Q<0.
    # Flag = -2 is E<0.
    # Flag = -3 something else went wrong

    Ra = Params["Ra"][0]
    Rv = Params["Rv"][0]

    Ti0 = Params["Ti0"][0]
    Ti00 = Params["Ti00"][0]

    logQ = y[0, 0]
    logM = y[1, 0]
    phia = y[2, 0]
    phiv = y[3, 0]
    Cp = y[7, 0]
    rho = y[8, 0]
    T = y[9, 0]
    theta = y[10, 0]
    z = y[11, 0]

    MetVals, z_pos = Met_interp(Met, z, z_pos)

    Pa = MetVals["Pa"]

    Q = np.exp(logQ)

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

    phiaA = np.float64(1.0) - phivA

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

    # Check if U<=Umin
    U = np.exp(logM - logQ)
    Uz = U * np.sin(theta)

    if Uz <= SolParams["Umin"][0]:
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
