from numpy import exp
from scipy.stats import lognorm, truncnorm, uniform

from .IC import ICSet
from .MetDataSet import MetData, MetDataArrays
from .ParamSet import ParamsSet
from .sampling_variables import Params
from .solver import integrate, integrateICs
from .SolverParams import SolverSet


class DistributedVariable():

    def __init__(self, name: str , dist: str, param0: float, param1: float, param2: float, param3: float):

        self.name = name
        self.min = param0
        self.max = param1
        self.param0 = param2
        self.param1 = param3

        match dist:
            case "Uniform":
                self.dist = uniform(loc=param0, scale=param1-param0)
            case "TruncNorm":
                a, b = (param0 - param2)/param3, (param1-param2)/param3
                self.dist = truncnorm(a, b, loc=param2, scale=param3)
            case "LogNorm":
                s = param3
                self.dist = lognorm(s, loc=param2)


def make_ensemble(sample_settings, design="LHC"):

    fixed = []

    for p in Params:
        label = p['label']
        print(label)
        print(sample_settings[label+'-fixed'])
        if sample_settings[label+'-fixed']:
            fixed.append({label: float(sample_settings[label+'-dist-param-0'])})

    print(fixed)

    return


# IC = ICSet()

# Params = ParamsSet()

# SolParams = SolverSet()

# IC, y_start, GasThrust, Condense, Flag = integrateICs(IC, Params, Met)

# soln, solverFlag, msg = integrate(
#     IC, Params, SolParams, Met, GasThrust, Condense, y_start
# )
