
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
from numba import njit, prange
from pyDOE import lhs
from scipy.stats import beta, truncnorm, uniform

from .IC import set_IC
from .MetDataSet import MetData, MetDataArray
from .ParamSet import set_params
from .sampling_variables import Params
from .solver import integrate, integrateICs
from .SolverParams import set_solver_params


class DistributedVariable():

    def __init__(self, name: str , dist: str, param0: float, param1: float, param2: float, param3: float):

        self.name = name
        self.param0 = param0
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3
        self.dist_name = dist

        match dist:
            case "Uniform":
                self.dist = uniform(loc=param0, scale=param1-param0)
            case "Truncated normal":
                a, b = (param0 - param2)/param3, (param1-param2)/param3
                self.dist = truncnorm(a, b, loc=param2, scale=param3)
            case "Beta":
                self.dist = beta(param2, param3, loc=param0, scale=(param1-param0))

    def __str__(self):
        return f"{self.name} has distribution {self.dist_name} with parameters {self.param0}, {self.param1}, {self.param2}, {self.param3}."
    
    def sample_from_ppf(self, ppf_vals: np.ndarray):
        self.samples = self.dist.ppf(ppf_vals)
    



def make_ensemble(sample_settings, ensemble_size, design="LHC"):

    fixed = []
    dists = []

    for label in Params.keys():
        if sample_settings[label]['fixed']:
            fixed.append({'name':label, 'value':float(sample_settings[label]['dist-param-0'])})
        else:
            dists.append(DistributedVariable(
                name=label,
                dist=sample_settings[label]['dist'],
                param0=sample_settings[label]['dist-param-0'],
                param1=sample_settings[label]['dist-param-1'],
                param2=sample_settings[label]['dist-param-2'],
                param3=sample_settings[label]['dist-param-3']
                )
            )

    N_dists = len(dists)

    if design=='LHC':
        lhd = lhs(N_dists, samples=ensemble_size)

        for j, d in enumerate(dists):
            dists[j].sample_from_ppf(lhd[:,j])
            
    df = pd.DataFrame(columns=list(Params.keys()))
    for d in dists:
        df[d.name] = d.samples.tolist()
    
    for f in fixed:
        df.loc[:, f['name']] = f['value']

    return df


def solve_plume(vent_elevation,
                vent_radius,
                gas_mass_fraction,
                source_temperature,
                source_velocity,
                pyroclast_density,
                ks,
                kw,
                met_data
    ):
    
    IC = set_IC(z0=vent_elevation,
               L0=vent_radius,
               n0=gas_mass_fraction,
               T0=source_temperature,
               U0=source_velocity)

    Parameters = set_params(rhos=pyroclast_density,
                           ks=ks,
                           kw=kw
    )

    SolParams = set_solver_params()

    Met = MetDataArray(
        met_data[:,2],
        met_data[:,0],
        met_data[:,1],
        met_data[:,3],
        met_data[:,4],
        met_data[:,5]
    )

    IC, y_start, GasThrust, Condense, Flag = integrateICs(IC, Parameters, Met)

    plume_state, _, _, _ = integrate(IC, Parameters, SolParams, Met, GasThrust, Condense, y_start)

    return {
        'Q0': plume_state['Q0'],
        'Qs0': plume_state['Qs0'],
        'Hn': plume_state['Hn'],
        'Htop': plume_state['Htop'],
        'HtopL': plume_state['HtopL'],
        'HtopLg': plume_state['HtopLg'],
        'HCondense': plume_state['HCondense'],
        'Collapse': plume_state['Collapse']
    }


def run_solve_plume(params,
                    met_data
):
    vent_elevation = params[0]
    vent_radius = params[1]
    gas_mass_fraction = params[2]
    source_temperature = params[3]
    source_velocity = params[4]
    pyroclast_density = params[5]
    ks = params[6]
    kw = params[7]

    soln = solve_plume(
            vent_elevation,
            vent_radius,
            gas_mass_fraction,
            source_temperature,
            source_velocity,
            pyroclast_density,
            ks,
            kw,
            met_data
    )

        # ret = [soln['Q0'],
        #     soln['Hn'],
        #     soln['Htop'],
        #     soln['HtopL'],
        #     soln['HtopLg'],
        #     soln['HCondense'],
        #     soln['Collapse']]
        
    ret = {
        'Q0': soln['Q0'],
        'Qs0': soln['Qs0'],
        'Hn': soln['Hn'],
        'Htop': soln['Htop'],
        'HtopL': soln['HtopL'],
        'HtopLg': soln['HtopLg'],
        'HCondense': soln['HCondense'],
        'Collapse': int(soln['Collapse'])
    }
    return ret


def solve_ensemble(sample_settings, ensemble_size, met_data):

    met_data = np.array(met_data)

    df = make_ensemble(sample_settings, ensemble_size)

    num_cores = mp.cpu_count()-1
    with mp.Pool(num_cores) as pool:
        results = pool.map(partial(run_solve_plume, met_data=met_data), list(df.itertuples(index=False, name=None)))
    
    df_out = pd.DataFrame(results)
    print(df_out)

    return pd.merge(df, df_out, left_index=True, right_index=True)



# IC = ICSet()

# Params = ParamsSet()

# SolParams = SolverSet()

# IC, y_start, GasThrust, Condense, Flag = integrateICs(IC, Params, Met)

# soln, solverFlag, msg = integrate(
#     IC, Params, SolParams, Met, GasThrust, Condense, y_start
# )
