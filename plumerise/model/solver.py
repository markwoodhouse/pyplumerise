import numpy as np
from numba import njit
from numba.types import float64, int64, string
from plumerise.model.setupsoln import init_soln

# from .IC import ICSet
from .ice_eqns import CheckStep, fice
from .MetDataSet import Met_interp, SpecHumidity, esfunc, eswfunc

# from .ParamSet import ParamsSet
from .rk import rkstep
from .setupsoln import NeutralHeight, clip_soln, get_soln, init_soln, update_soln

# from .SolverParams import SolverSet


@njit(cache=True)
def NextStep(
    h_next: np.float64,
    Params: np.ndarray,
    SolParams: np.ndarray,
    Met: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    z_pos: np.int64,
    GasThrust: np.int64,
):

    U = np.exp(y[1,0]-y[0,0]) # y[0,0] = logQ, y[1,0] = logM, M/Q = exp(logM - logQ)
    theta = y[10, 0]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    dU = U * (dy[1,0]-dy[0,0]) # dU/ds = U * (d(logM)/ds - d(logQ)/ds)
    dtheta = dy[10,0]

    h = h_next    

    Uz = U * sin_theta
    dUz = dU*sin_theta + U*cos_theta*dtheta # d(Uz)/ds = d/ds (U sin theta)

    Uz_next = Uz + dUz*h

    if (Uz_next <= 0):
        h = -Uz/dUz
        h = 0.9*h
    
    h = np.maximum(h, SolParams['MinStep'][0])
    h = np.minimum(h, SolParams['MaxStep'][0])

    return h


@njit(cache=True)
def integrate(
    IC: np.ndarray,
    Params: np.ndarray,
    SolParams: np.ndarray,
    Met: np.ndarray,
    GT: np.int64,
    Cond: np.int64,
    y_start: np.ndarray,
):

    z_pos = 0
    s0 = np.float64(0.0)
    sold = s0

    threeD = SolParams["threeD"][0]

    Nsize = len(y_start)

    soln = init_soln(SolParams["MaxSolnPnts"][0])

    plume_state = dict()

    plume_state["Q0"] = np.pi * y_start[0][0]
    plume_state["Qs0"] = (1.0 - IC["n0"][0]) * plume_state["Q0"]
    soln_pos = 0
    soln, z_pos, soln_pos = update_soln(
        soln, s0, y_start, Params, Met, SolParams, z_pos, soln_pos
    )

    s = s0
    y = y_start
    h = SolParams["InitStep"][0]
    h_max = SolParams["MaxStep"][0]

    GasThrust = GT

    GotCondHeight = -1

    if Cond == 0:
        GotCondHeight = 0
        plume_state["HCondense"] = np.float64(0.0)
    
    try:
        dy, z_pos, eqn_flag = fice(y, Params, Met, z_pos, GasThrust, threeD)
    except:
        msg = "Nans/Infs in equations at initial step"
        solverFlag = 0
        return plume_state, soln, solverFlag, msg

    h = SolParams["InitStep"][0]/np.linalg.norm(dy)
    if h<SolParams["MinStep"][0]: h = SolParams["MinStep"][0]

    while True:
        
        solverFlag = 0

        while h > SolParams['MinStep'][0]:
            
            z_pos, yt, h_done, h_next, solverFlag = rkstep(
                y, dy, h, Params, SolParams, Met, z_pos, GasThrust)
            
            if solverFlag<0:
                h = 0.9*h
                print('Failed rkstep: reducing step size')
                continue
            
            z_pos, GasThrust, Condense, StepFlag = CheckStep(
                s + h_done, yt, Params, Met, SolParams, z_pos, GasThrust
            )

            try:
                dy_next, z_pos_next, eqn_flag = fice(yt, Params, Met, z_pos, GasThrust, threeD)
            except:
                h = 0.9*h
                print('Failed rhs calculation: reducing step size')
                continue

            break
        
        s = s + h_done
        y = yt
        dy = dy_next
        z_pos = z_pos_next
        h = NextStep(h_next, Params, SolParams, Met, y, dy, z_pos, GasThrust)

        if (GotCondHeight == 0) and (Condense == 1):
            plume_state["HCondense"] = y[14, 0]
            GotCondHeight = 1

        if solverFlag == 0:
            soln = clip_soln(soln, soln_pos+1)
            msg = "Something went wrong"
            break
        
        if StepFlag < 0:
            soln = clip_soln(soln, soln_pos+1)
            msg = "Something went wrong"
            break
        elif StepFlag == 1:  # passed end point for integration
            soln, z_pos, soln_pos = update_soln(
                soln, s, y, Params, Met, SolParams, z_pos, soln_pos
            )
            soln = clip_soln(soln, soln_pos)
            msg = "Integrated past end point"
            break
        elif StepFlag == 2:  # stop condition has been satisfied
            soln, z_pos, soln_pos = update_soln(
                soln, s, y, Params, Met, SolParams, z_pos, soln_pos
            )
            soln = clip_soln(soln, soln_pos)
            msg = "Stop condition satisfied"
            break
        elif StepFlag == 0:
            if (s - sold) > SolParams["OutStep"][0]:
                soln, z_pos, soln_pos = update_soln(
                    soln, s, y, Params, Met, SolParams, z_pos, soln_pos
                )
                sold = s
            if solverFlag == -1:
                soln = clip_soln(soln, soln_pos)
                msg = "In solver, step size below minimum"
                break

    topPos = np.argmax(get_soln(soln, "z"))
    plume_state["Htop"] = get_soln(soln, "z")[topPos]
    plume_state["HtopL"] = get_soln(soln, "z")[topPos] + get_soln(soln, "L")[topPos]
    plume_state["HtopLg"] = (
        get_soln(soln, "z")[topPos] + get_soln(soln, "Lgaussian")[topPos]
    )
    plume_state["Hn"] = NeutralHeight(soln)

    # match SolParams.StopCond:
    #     case 'Top' | 'HorizDistance':
    #         topPos = np.argmax(soln.z)
    #         soln.Htop = soln.z[topPos]
    #         soln.HtopL = soln.z[topPos] + soln.L[topPos]
    #         soln.HtopLg = soln.z[topPos] + soln.Lgaussian[topPos]
    #     case 'Neutral':
    #         soln.Htop = np.float64(-1.0)

    if GasThrust == 1:
        plume_state["Collapse"] = 1.0
    else:
        plume_state["Collapse"] = 0.0

    return plume_state, soln, solverFlag, msg


def integrateICs(IC: np.ndarray, Params: np.ndarray, Met: np.ndarray, threeD: bool):

    # Calculate some starting values

    z_pos = 1

    MetVals, z_pos = Met_interp(Met, IC["z0"][0], z_pos)

    psi0 = MetVals["psia"]
    Ta0 = MetVals["Ta"]

    esa0, _ = eswfunc(Ta0)  # saturation vapor pressure in atmosphere

    Pa0 = MetVals["Pa"]

    IC["Pa0"] = Pa0

    RH0 = MetVals["RH"]

    Ra = Params["Ra"][0]
    Rv = Params["Rv"][0]
    Ca = Params["Ca"][0]
    Cv = Params["Cv"][0]
    Cw = Params["Cw"][0]
    Ci = Params["Ci"][0]
    Cs = Params["Cs"][0]
    rhos = Params["rhos"][0]
    Ti0 = Params["Ti0"][0]
    Ti00 = Params["Ti00"][0]

    phivA0 = SpecHumidity(RH0, esa0, Pa0, Ra, Rv)

    phiaA0 = np.float64(1.0) - phivA0

    Catm0 = phiaA0 * Ca + phivA0 * Cv
    # atmospheric density
    Rga0 = phiaA0 * Ra + phivA0 * Rv
    rhoAtm0 = Pa0 / Rga0 / Ta0

    # Set some related initial quantities
    IC["phia0"][0] = np.float64(0.0)  # no dry air at vent -- all gas is water vapour
    IC["phiv0"][0] = IC["n0"][0]  # all gas at vent is water vapour
    IC["phiw0"][0] = np.float64(0.0)  # no liquid water at vent
    IC["phii0"][0] = np.float64(0.0)  # no ice at vent
    IC["phis0"][0] = np.float64(1.0) - IC["n0"][0]  # solids mass fraction at vent
    IC["Rg0"][0] = (IC["phiv0"][0] * Rv + IC["phia0"][0] * Ra) / (
        IC["phia0"][0] + IC["phiv0"][0]
    )  # bulk gas constant of volcanic gas at vent
    IC["rho0"][0] = np.float64(1.0) / (
        IC["n0"][0] * IC["Rg0"][0] * IC["T0"][0] / IC["Pa0"][0]
        + (np.float64(1.0) - IC["n0"][0]) / rhos
    )  # bulk density of the plume

    Pv0 = IC["phiv0"][0] * Rv / (IC["phia0"][0] * Ra + IC["phiv0"][0] * Rv) * IC["Pa0"]
    es0, _ = esfunc(IC["T0"][0], Ti0, Ti00)

    if Pv0 > es0:
        Condense = 1
    else:
        Condense = 0

    # Do some checks on initial conditions
    if IC["rho0"][0] <= 0:
        raise ValueError("Inputs produce unphysical rho0 < 0.")

    IC["Cp0"][0] = (
        IC["phia0"][0] * Ca
        + IC["phis0"][0] * Cs
        + IC["phiv0"][0] * Cv
        + IC["phiw0"][0] * Cw
        + IC["phii0"][0] * Ci
    )  # specific heat capacity of bulk plume

    # Source fluxes
    if IC["L0"][0] > 0.0 and IC["U0"][0] > 0.0:
        IC["Q0"][0] = (
            IC["rho0"][0] * IC["U0"][0] * IC["L0"][0] * IC["L0"][0]
        )  # [kg/s] mass flux at vent
        IC["M0"][0] = (
            IC["rho0"][0] * IC["U0"][0] * IC["U0"][0] * IC["L0"][0] * IC["L0"][0]
        )  # [kg*m/s^2] momentum flux at vent
    elif IC["Q0"][0] > 0.0 and IC["M0"][0] > 0.0:
        IC["Q0"][0] = IC["Q0"][0] / np.pi
        IC["U0"][0] = IC["M0"][0] / IC["Q0"][0]
        IC["L0"][0] = IC["Q0"][0] / np.sqrt(IC["M0"][0] * IC["rho0"][0])
    elif IC["L0"][0] > 0.0 and IC["Q0"][0] > 0.0:
        IC["Q0"][0] = IC["Q0"][0] / np.pi
        IC["U0"][0] = IC["Q0"][0] / IC["rho0"][0] / IC["L0"][0] / IC["L0"][0]
        IC["M0"][0] = (
            IC["Q0"][0] * IC["Q0"][0] / IC["rho0"][0] / IC["L0"][0] / IC["L0"][0]
        )  # [kg*m/s^2] momentum flux at vent
    elif IC["L0"][0] > 0.0 and IC["M0"][0] > 0.0:
        IC["M0"][0] = IC["M0"][0] / np.pi
        IC["U0"][0] = np.sqrt(IC["M0"][0] / IC["rho0"][0]) / IC["L0"][0]
        IC["Q0"][0] = IC["L0"][0] * np.sqrt(
            IC["rho0"][0] * IC["M0"][0]
        )  # [kg/s] mass flux at vent
    elif IC["U0"][0] > 0.0 and IC["Q0"][0] > 0.0:
        IC["Q0"][0] = IC["Q0"][0] / np.pi
        IC["L0"][0] = np.sqrt(IC["Q0"][0] / IC["rho0"][0] / IC["U0"][0])
        IC["M0"][0] = IC["Q0"][0] * IC["U0"][0]  # [kg*m/s^2] momentum flux at vent
    elif IC["U0"][0] > 0.0 and IC["M0"][0] > 0.0:
        IC["M0"][0] = IC["M0"][0] / np.pi
        IC["L0"][0] = np.sqrt(IC["M0"][0] / IC["rho0"][0]) / IC["U0"][0]
        IC["Q0"][0] = IC["M0"][0] / IC["U0"][0]  # [kg/s] mass flux at vent
    else:
        raise ValueError(
            "Error in setting initial conditions.  Incomplete input data found."
        )

    IC["M0"][0] = (
        IC["rho0"][0] * IC["U0"][0] * IC["U0"][0] * IC["L0"][0] * IC["L0"][0]
    )  # [kg*m/s^2] momentum flux at vent
    IC["E0"][0] = IC["Q0"][0] * IC["Cp0"][0] * IC["T0"][0]

    if IC["rho0"][0] > rhoAtm0:
        GasThrust = 1  # density of plume at source is bigger than atmospheric
    else:
        GasThrust = 0

    # Set initial conditions
    if threeD:
        Nsize = 15
    else:
        Nsize = 12
    y_start = np.zeros((Nsize, 1), dtype=np.float64)
    y_start[0, 0] = np.log(IC["Q0"][0])
    y_start[1, 0] = np.log(IC["M0"][0])
    y_start[2, 0] = IC["phia0"][0]
    y_start[3, 0] = IC["phiv0"][0]
    y_start[4, 0] = IC["phis0"][0]
    y_start[5, 0] = IC["phiw0"][0]
    y_start[6, 0] = IC["phii0"][0]
    y_start[7, 0] = IC["Cp0"][0]
    y_start[8, 0] = IC["rho0"][0]
    y_start[9, 0] = IC["T0"][0]
    y_start[10, 0] = np.float64(0.5) * np.pi
    y_start[11, 0] = IC["z0"][0]
    if threeD:
        y_start[12, 0] = psi0
        y_start[13, 0] = np.float64(0.0)
        y_start[14, 0] = np.float64(0.0)
    

    Flag = 1
    return IC, y_start, GasThrust, Condense, Flag

def solve(Params,SolParams,IC,Met):

    IC, y_start, GasThrust, Condense, Flag = integrateICs(IC, Params, Met, SolParams["threeD"][0])

    plume_state, soln, solverFlag, msg = integrate(IC, Params, SolParams, Met, GasThrust, Condense, y_start)

    print(msg)

    return soln, solverFlag