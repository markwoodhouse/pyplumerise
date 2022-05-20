import numpy as np
from numba import njit
from numba.types import float64, int64, string

from .IC import ICSet
from .ice_eqns import CheckStep, fice
from .MetDataSet import MetDataArrays, SpecHumidity, esfunc, eswfunc
from .ParamSet import ParamsSet
from .rk import rkstep
from .setupsoln import solnArray
from .SolverParams import SolverSet


@njit
def NextStep(h_next: float64,
    Params: ParamsSet,
    SolParams: SolverSet,
    Met: MetDataArrays,
    y: np.ndarray,
    yold: np.ndarray,
    z_pos: int64,
    GasThrust: int64
):
        
    zpost = z_pos
    s2 = y[0,0]
    s1 = yold[0,0]
    
    z2 = y[14,0]
    z1 = yold[14,0]
    
    theta2 = y[10,0]
    theta1 = yold[10,0]
    
    rho2 = y[8,0]
    rho1 = yold[8,0]

    h = h_next
    
    match SolParams.StopCond:
        case 'Top':
            U2 = y[1,0]/y[0,0]
            U1 = yold[1,0]/yold[0,0]

            f2 = U2*np.sin(theta2) - SolParams.Umin
            f1 = U1*np.sin(theta1) - SolParams.Umin
        
            send = s2 - (s2-s1)*f2/(f2-f1)
        
            h = (send-s2)
      
            h = np.minimum(np.abs(h), np.abs(h_next))

        case 'HorizDistance':
          
            f2 = np.sqrt(y[12,0]*y[12,0]+y[13,0]*y[13,0])-SolParams.MaxHorizDistance
            f1 = np.sqrt(yold[12,0]*yold[12,0]+yold[13,0]*yold[13,0])-SolParams.MaxHorizDistance
        
            send = (s1*f2-s2*f1)/(f2-f1)
        
            h = (send-s2)
      
            h = np.minimum(np.abs(h), np.abs(h_next))
          
        case 'Neutral':
            
            if (GasThrust==0):
                MetVals2, zpost = Met.interp(z2, zpost)
                MetVals1, zpost = Met.interp(z1, zpost)
          
                Pa2 = MetVals2['Pa']
                Pa1 = MetVals1['Pa']
                Ta2 = MetVals2['Ta']
                Ta1 = MetVals1['Ta']
                RH2 = MetVals2['RH']
                RH1 = MetVals1['RH']
                esa2, _ = eswfunc(Ta2)
                esa1, _ = eswfunc(Ta1)

                phivA2 = SpecHumidity(RH2, esa2, Pa2, Params.Ra, Params.Rv)
                phivA1 = SpecHumidity(RH1, esa1, Pa1, Params.Ra, Params.Rv)
          
                # atmospheric density
                Rga2 = phivA2*Params.Rv + (1.0-phivA2)*Params.Ra
                Rga1 = phivA1*Params.Rv + (1.0-phivA1)*Params.Ra
                rhoAtm2 = Pa2/Rga2/Ta2
                rhoAtm1 = Pa1/Rga1/Ta1
        
                f2 = rho2 - rhoAtm2
                f1 = rho1 - rhoAtm1
          
                if (f2<0):
                  send = s2 - (s2-s1)*f2/(f2-f1)
                  h = (send-s2)
                  h = np.minimum(np.abs(h), np.abs(h_next))
            else:
                h = h_next
      
    h = np.abs(h)*np.sign(h_next)
      
    return h


@njit
def integrate(IC: ICSet, Params: ParamsSet, SolParams: SolverSet, Met: MetDataArrays, GT: int64, Cond: int64, y_start: np.ndarray):

    z_pos = 0
    s0 = np.float64(0.0)
    sold = s0
    
    Nsize = len(y_start)
    
    soln = solnArray(SolParams.MaxSolnPnts)
      
    soln.Q0 = np.pi*y_start[0]
    soln_pos = 0
    z_pos, soln_pos = soln.update(s0, y_start, Params, Met, SolParams, z_pos, soln_pos)
    
    s = s0
    y = y_start
    h = SolParams.InitStep
    h_max = SolParams.MaxStep
    
    GasThrust = GT
    
    GotCondHeight = -1
    
    if (Cond==0):
        GotCondHeight=0
        soln.HCondense = np.float64(0.0)
    
    while True:
        yold = y
        dy, z_pos = fice(y, IC, Params, Met, z_pos, GasThrust)
    
        solverFlag = 0
      
        z_pos, yt, h_done, h_next, solverFlag = rkstep(y, dy, h, IC, Params, SolParams, Met, z_pos, GasThrust)
                  
        s = s+h_done
        y = yt
      
        z_pos, GasThrust, Condense, StepFlag = CheckStep(s, y, Params, Met, SolParams, z_pos, GasThrust)
        if ((GotCondHeight==0) and (Condense==1)):
            soln.HCondense = y[14,0]
            GotCondHeight = 1

        if (solverFlag==0):
            soln.clip(soln_pos)
            msg = 'Something went wrong'
            break
      
        if (StepFlag<0):
            soln.clip(soln_pos)
            msg = 'Something went wrong'
            break
        elif (StepFlag==1): # passed end point for integration
            z_pos, soln_pos = soln.update(s, y, Params, Met, SolParams, z_pos, soln_pos)
            soln.clip(soln_pos)
            msg = 'Integrated past end point'
            break
        elif (StepFlag==2): # stop condition has been satisfied
            z_pos, soln_pos = soln.update(s, y, Params, Met, SolParams, z_pos, soln_pos)
            soln.clip(soln_pos)
            msg = 'Stop condition satisfied'
            break
        elif (StepFlag==0):
            if ((s-sold)>SolParams.OutStep):
                z_pos, soln_pos = soln.update(s, y, Params, Met, SolParams, z_pos, soln_pos)
                sold = s
            if (solverFlag==-1):
                soln.clip(soln_pos)
                msg = 'In solver, step size below minimum'
                break
            
        h =  NextStep(h_next, Params, SolParams, Met, y, yold, z_pos, GasThrust)
               
    match SolParams.StopCond:
        case 'Top' | 'HorizDistance':
            topPos = np.argmax(soln.z)
            soln.Htop = soln.z[topPos]
            soln.HtopL = soln.z[topPos] + soln.L[topPos]
            soln.HtopLg = soln.z[topPos] + soln.Lgaussian[topPos]
        case 'Neutral':
            soln.Htop = np.float64(-1.0)
        
    if (GasThrust==1):
        soln.Collapse = 1
    else:
        soln.Collapse = 0

    return soln, solverFlag, msg


def integrateICs(IC: ICSet, Params: ParamsSet, Met: MetDataArrays):
  
    # Calculate some starting values

    z_pos = 1
    
    MetVals, z_pos = Met.interp(IC.z0, z_pos)
    
    psi0 = MetVals['psia']
    Ta0 = MetVals['Ta']

    esa0, _ = eswfunc(Ta0) # saturation vapor pressure in atmosphere
  
    Pa0 = MetVals['Pa']
    
    IC.Pa0 = Pa0
    
    RH0 = MetVals['RH']
  
    phivA0 = SpecHumidity(RH0, esa0, Pa0, Params.Ra, Params.Rv)
      
    phiaA0 = 1.0-phivA0
  
    Catm0 = phiaA0*Params.Ca + phivA0*Params.Cv
    # atmospheric density
    Rga0 = phiaA0*Params.Ra + phivA0*Params.Rv
    rhoAtm0 = Pa0/Rga0/Ta0
  
    # Set some related initial quantities
    IC.phia0 = np.float64(0.0) # no dry air at vent -- all gas is water vapour
    IC.phiv0 = IC.n0 # all gas at vent is water vapour
    IC.phiw0 = np.float64(0.0) # no liquid water at vent
    IC.phii0 = np.float64(0.0) # no ice at vent
    IC.phis0 = 1.0-IC.n0 # solids mass fraction at vent
    IC.Rg0 = (IC.phiv0 * Params.Rv + IC.phia0 * Params.Ra)/(IC.phia0+IC.phiv0) # bulk gas constant of volcanic gas at vent
    IC.rho0 = 1.0/(IC.n0 * IC.Rg0 * IC.T0 / IC.Pa0 + (1.0-IC.n0) / Params.rhos) # bulk density of the plume
    
    Pv0 = IC.phiv0 * Params.Rv / (IC.phia0 * Params.Ra + IC.phiv0 * Params.Rv)*IC.Pa0
    es0, _ = esfunc(IC.T0, Params.Ti0, Params.Ti00)
    
    if (Pv0>es0):
      Condense = 1
    else:
      Condense = 0
      
    # Do some checks on initial conditions
    if (IC.rho0<=0):
        raise ValueError('Inputs produce unphysical rho0 < 0.')
        
    IC.Cp0 = IC.phia0 * Params.Ca + IC.phis0 * Params.Cs + IC.phiv0 * Params.Cv + IC.phiw0 * Params.Cw + IC.phii0 * Params.Ci # specific heat capacity of bulk plume

    # Source fluxes
    if (IC.L0>0.0 and IC.U0>0.0):
        IC.Q0 = IC.rho0 * IC.U0 * IC.L0 * IC.L0  # [kg/s] mass flux at vent
        IC.M0 = IC.rho0 * IC.U0 * IC.U0 * IC.L0 * IC.L0  # [kg*m/s^2] momentum flux at vent
    elif (IC.Q0>0.0 and IC.M0>0.0):
        IC.Q0 = IC.Q0 / np.pi
        IC.U0 = IC.M0 / IC.Q0
        IC.L0 = IC.Q0 / np.sqrt(IC.M0 * IC.rho0)
    elif (IC.L0>0.0 and IC.Q0>0.0):
        IC.Q0 = IC.Q0 / np.pi
        IC.U0 = IC.Q0 / IC.rho0 / IC.L0 / IC.L0
        IC.M0 = IC.Q0 * IC.Q0 / IC.rho0 / IC.L0 / IC.L0 # [kg*m/s^2] momentum flux at vent
    elif (IC.L0>0.0 and IC.M0>0.0):
        IC.M0 = IC.M0 / np.pi
        IC.U0 = np.sqrt(IC.M0 / IC.rho0) / IC.L0
        IC.Q0 = IC.L0 * np.sqrt(IC.rho0 * IC.M0)  # [kg/s] mass flux at vent
    elif (IC.U0>0.0 and IC.Q0>0.0):
        IC.Q0 = IC.Q0 / np.pi
        IC.L0 = np.sqrt(IC.Q0 / IC.rho0 / IC.U0)
        IC.M0 = IC.Q0 * IC.U0  # [kg*m/s^2] momentum flux at vent
    elif (IC.U0>0.0 and IC.M0>0.0):
        IC.M0 = IC.M0 / np.pi
        IC.L0 = np.sqrt(IC.M0 / IC.rho0) / IC.U0
        IC.Q0 = IC.M0 / IC.U0 # [kg/s] mass flux at vent
    else:
        raise ValueError("Error in setting initial conditions.  Incomplete input data found.")
      
    IC.M0 = IC.rho0 * IC.U0 * IC.U0 * IC.L0 * IC.L0  # [kg*m/s^2] momentum flux at vent
    IC.E0 = IC.Q0 * IC.Cp0 * IC.T0
    
    if (IC.rho0 > rhoAtm0):
        GasThrust = 1 # density of plume at source is bigger than atmospheric
    else:
        GasThrust = 0
    
    # Set initial conditions
    y_start = np.zeros((15,1), dtype=np.float64)
    y_start[0,0] = IC.Q0
    y_start[1,0] = IC.M0
    y_start[2,0] = IC.phia0
    y_start[3,0] = IC.phiv0
    y_start[4,0] = IC.phis0
    y_start[5,0] = IC.phiw0
    y_start[6,0] = IC.phii0
    y_start[7,0] = IC.Cp0
    y_start[8,0] = IC.rho0
    y_start[9,0] = IC.T0
    y_start[10,0] = 0.5*np.pi
    y_start[11,0] = psi0
    y_start[12,0] = 0.0
    y_start[13,0] = 0.0
    y_start[14,0] = IC.z0
    
    Flag = 1
    return IC, y_start, GasThrust, Condense, Flag
