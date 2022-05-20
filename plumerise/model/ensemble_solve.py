from numpy import exp
from scipy.stats import lognorm, truncnorm, uniform

from .IC import ICSet
from .MetDataSet import MetData, MetDataArrays
from .ParamSet import ParamsSet
from .sampling_variables import Params
from .solver import integrate, integrateICs
from .SolverParams import SolverSet


class DistributedVariable():

    def __init__(self, name: str, min: float, max: float, dist: str, param0: float, param1: float):

        self.name = name
        self.min = min
        self.max = max
        self.param0 = param0
        self.param1 = param1

        match dist:
            case "Uniform":
                self.dist = uniform(loc=min, scale=max-min)
            case "TruncNorm":
                a, b = (min - param0)/param1, (max-param0)/param1
                self.dist = truncnorm(a, b, loc=param0, scale=param1)
            case "LogNorm":
                s = param1
                self.dist = lognorm(s, loc=param0)


def make_ensemble(sample_settings, design="LHC"):

    fixed = []


    # for p in Params:
    #     label = p['label']
    #     if sample_settings[label+'']


    return


# IC = ICSet()

# Params = ParamsSet()

# SolParams = SolverSet()

# IC, y_start, GasThrust, Condense, Flag = integrateICs(IC, Params, Met)

# soln, solverFlag, msg = integrate(
#     IC, Params, SolParams, Met, GasThrust, Condense, y_start
# )
