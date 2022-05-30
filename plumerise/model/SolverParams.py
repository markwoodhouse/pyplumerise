import numpy as np

# from numba.experimental import jitclass
from numba.types import float64, int64, string


def set_solver_params(
    sMax: float = 100.0e3,  # Default end point of integration, so eventually integration will stop
    # StopCond: str = "Top",  # Stop when plume vertical velocity first reaches zero (i.e. at first maxima).
    Umin: float = 1.0e-4,  # Default minimum rise speed.  If StopCond='Top' and U falls below Umin during integration, the integration stops
    MaxHorizDistance: float = 100e3,  # Default horizontal distance from source.  If StopCond='HorizDistance' and sqrt(x^2+y^2)>MaxHorizDistance the integration stops
    rtol: float = 1.0e-8,  # Default relative error tolerance
    MinStep: float = 1.0e-10,  # Default minimum rk step size
    MaxStep: float = 1.0e1,  # Default minimum rk step size
    InitStep: float = 1.0e-3,  # Default initial rk step size
    OutStep: float = 1.0,
    MaxSolnPnts: int = 1000000,  # Maximum number of solution points allowed by the solver.
    invsolMaxIts: int = 50,  # Default maximum number of iterations for inverse solver
    invsolrtol: float = 1.0e-3,  # Default relative error tolerance for inverse solver
    invsolatol: float = 1.0e-1,  # Default absolute error tolerance for inverse solver
    invsolxtol: float = 1.0e-14,  # Default tolerance for interval in inverse solver
):

    sp_dtype = np.dtype(
        [
            ("sMax", np.float64),
            ("Umin", np.float64),
            ("MaxHorizDistance", np.float64),
            ("rtol", np.float64),
            ("MinStep", np.float64),
            ("MaxStep", np.float64),
            ("InitStep", np.float64),
            ("OutStep", np.float64),
            ("MaxSolnPnts", np.int64),
            ("invsolMaxIts", np.int64),
            ("invsolrtol", np.float64),
            ("invsolatol", np.float64),
            ("invsolxtol", np.float64),
        ]
    )

    sp = np.zeros((1), dtype=sp_dtype)

    sp["sMax"][0] = np.float64(sMax)
    sp["Umin"][0] = np.float64(Umin)
    sp["MaxHorizDistance"][0] = np.float64(MaxHorizDistance)
    sp["rtol"][0] = np.float64(rtol)
    sp["MinStep"][0] = np.float64(MinStep)
    sp["MaxStep"][0] = np.float64(MaxStep)
    sp["InitStep"][0] = np.float64(InitStep)
    sp["OutStep"][0] = np.float64(OutStep)
    sp["MaxSolnPnts"][0] = MaxSolnPnts
    sp["invsolMaxIts"][0] = invsolMaxIts
    sp["invsolrtol"][0] = np.float64(invsolrtol)
    sp["invsolatol"][0] = np.float64(invsolatol)
    sp["invsolxtol"][0] = np.float64(invsolxtol)
    return sp


# SolverSetSpec = [
#     ("sMax", float64),
#     ("StopCond", string),
#     ("Umin", float64),
#     ("MaxHorizDistance", float64),
#     ("rtol", float64),
#     ("MinStep", float64),
#     ("MaxStep", float64),
#     ("InitStep", float64),
#     ("OutStep", float64),
#     ("MaxSolnPnts", int64),
#     ("invsolMaxIts", int64),
#     ("invsolrtol", float64),
#     ("invsolatol", float64),
#     ("invsolxtol", float64),
# ]


# @jitclass(SolverSetSpec)
# class SolverSet(object):
#     sMax: float64  # Default end point of integration, so eventually integration will stop
#     StopCond: string  # Stop when plume vertical velocity first reaches zero (i.e. at first maxima).
#     Umin: float64  # Default minimum rise speed.  If StopCond='Top' and U falls below Umin during integration, the integration stops
#     MaxHorizDistance: float64  # Default horizontal distance from source.  If StopCond='HorizDistance' and sqrt(x^2+y^2)>MaxHorizDistance the integration stops
#     rtol: float64  # Default relative error tolerance
#     MinStep: float64  # Default minimum rk step size
#     MaxStep: float64  # Default minimum rk step size
#     InitStep: float64  # Default initial rk step size
#     OutStep: float64
#     MaxSolnPnts: int64  # Maximum number of solution points allowed by the solver.
#     invsolMaxIts: int64  # Default maximum number of iterations for inverse solver
#     invsolrtol: float64  # Default relative error tolerance for inverse solver
#     invsolatol: float64  # Default absolute error tolerance for inverse solver
#     invsolxtol: float64  # Default tolerance for interval in inverse solver

#     def __init__(
#         self,
#         sMax: float = 100.0e3,  # Default end point of integration, so eventually integration will stop
#         StopCond: str = "Top",  # Stop when plume vertical velocity first reaches zero (i.e. at first maxima).
#         Umin: float = 1.0e-4,  # Default minimum rise speed.  If StopCond='Top' and U falls below Umin during integration, the integration stops
#         MaxHorizDistance: float = 100e3,  # Default horizontal distance from source.  If StopCond='HorizDistance' and sqrt(x^2+y^2)>MaxHorizDistance the integration stops
#         rtol: float = 1.0e-8,  # Default relative error tolerance
#         MinStep: float = 1.0e-10,  # Default minimum rk step size
#         MaxStep: float = 1.0e1,  # Default minimum rk step size
#         InitStep: float = 1.0e-3,  # Default initial rk step size
#         OutStep: float = 1.0,
#         MaxSolnPnts: int = 100000,  # Maximum number of solution points allowed by the solver.
#         invsolMaxIts: int = 50,  # Default maximum number of iterations for inverse solver
#         invsolrtol: float = 1.0e-3,  # Default relative error tolerance for inverse solver
#         invsolatol: float = 1.0e-1,  # Default absolute error tolerance for inverse solver
#         invsolxtol: float = 1.0e-14,  # Default tolerance for interval in inverse solver
#     ):
#         self.sMax = np.float64(sMax)
#         self.StopCond = StopCond
#         self.Umin = np.float64(Umin)
#         self.MaxHorizDistance = np.float64(MaxHorizDistance)
#         self.rtol = np.float64(rtol)
#         self.MinStep = np.float64(MinStep)
#         self.MaxStep = np.float64(MaxStep)
#         self.InitStep = np.float64(InitStep)
#         self.OutStep = np.float64(OutStep)
#         self.MaxSolnPnts = MaxSolnPnts
#         self.invsolMaxIts = invsolMaxIts
#         self.invsolrtol = np.float64(invsolrtol)
#         self.invsolatol = np.float64(invsolatol)
#         self.invsolxtol = np.float64(invsolxtol)
#         return
