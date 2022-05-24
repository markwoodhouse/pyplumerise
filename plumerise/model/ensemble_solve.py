
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

        print('Dist: ', dist)

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

    for p in Params:
        label = p['label']
        print(label)
        print(sample_settings[label+'-fixed'])
        if sample_settings[label+'-fixed']:
            fixed.append({'name':label, 'value':float(sample_settings[label+'-dist-param-0'])})
        else:
            print("dist = ", sample_settings[label+'-dist'])
            dists.append(DistributedVariable(
                name=label,
                dist=sample_settings[label+'-dist'],
                param0=sample_settings[label+'-dist-param-0'],
                param1=sample_settings[label+'-dist-param-1'],
                param2=sample_settings[label+'-dist-param-2'],
                param3=sample_settings[label+'-dist-param-3']
                )
            )

    N_dists = len(dists)

    if design=='LHC':
        lhd = lhs(N_dists, samples=ensemble_size)

        for j, d in enumerate(dists):
            dists[j].sample_from_ppf(lhd[:,j])
            
    df = pd.DataFrame(columns=[p['label'] for p in Params])
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

    Met = MetDataArray(met_data["height"],
                        met_data["temperature"],
                        met_data["pressure"],
                        met_data["direction"],
                        met_data["speed"],
                        met_data["relhum"]
    )

    IC, y_start, GasThrust, Condense, Flag = integrateICs(IC, Parameters, Met)

    plume_state, _, _, _ = integrate(IC, Parameters, SolParams, Met, GasThrust, Condense, y_start)

    return {
        'Q0': plume_state['Q0'][0],
        'Hn': plume_state['Hn'][0],
        'Htop': plume_state['Htop'][0],
        'HtopL': plume_state['HtopL'][0],
        'HtopLg': plume_state['HtopLg'][0],
        'HCondense': plume_state['HCondense'][0],
        'Collapse': plume_state['Collapse'][0]
    }


def run_solve_plume(df, met_data):
    for row in df.iterrows():
        idx = row[0]
        vent_elevation = row[1]['vent-elevation']
        vent_radius = row[1]['vent-radius']
        gas_mass_fraction = row[1]['gas-mass-fraction']
        source_temperature = row[1]['source-temperature']
        source_velocity = row[1]['source-velocity']
        pyroclast_density = row[1]['pyroclast-density']
        ks = row[1]['no-wind-entrainment-coef']
        kw = row[1]['wind-entrainment-coef']
        
        soln = solve_plume(vent_elevation,
                vent_radius,
                gas_mass_fraction,
                source_temperature,
                source_velocity,
                pyroclast_density,
                ks,
                kw,
                met_data
        )

        df.loc[idx, 'Q0'] = soln['Q0']
        df.loc[idx, 'Hn'] = soln['Hn']
        df.loc[idx, 'Htop'] = soln['Htop']
        df.loc[idx, 'HtopL'] = soln['HtopL']
        df.loc[idx, 'HtopLg'] = soln['HtopLg']
        df.loc[idx, 'HCondense'] = soln['HCondense']
        df.loc[idx, 'Collapse'] = soln['Collapse']

        return df


def solve_ensemble(sample_settings, ensemble_size, met_data):

    # Params = ParamsSet()
    # SolParams = SolverSet()
    # Met = MetDataArrays(met_data["height"],
    #                     met_data["temperature"],
    #                     met_data["pressure"],
    #                     met_data["direction"],
    #                     met_data["speed"],
    #                     met_data["relhum"]
    # )
    # IC, y_start, GasThrust, Condense, Flag = integrateICs(IC, Params, Met)
    # soln, solverFlag, msg = integrate(
    #     IC, Params, SolParams, Met, GIC = ICSet()asThrust, Condense, y_start
    # )

    df = make_ensemble(sample_settings, ensemble_size)

    num_cores = mp.cpu_count()-1
    chunk_size = int(df.shape[0]/num_cores)
    chunks =[df.iloc[df.index[i:i+chunk_size]] for i in range(0, df.shape[0], chunk_size)]
    with mp.Pool(num_cores) as pool:
        result = pool.map(partial(run_solve_plume, met_data=met_data), chunks)
    
    for i in range(len(result)):
        df.iloc[result[i].index] = result[i]

    print(df)

    return df



# IC = ICSet()

# Params = ParamsSet()

# SolParams = SolverSet()

# IC, y_start, GasThrust, Condense, Flag = integrateICs(IC, Params, Met)

# soln, solverFlag, msg = integrate(
#     IC, Params, SolParams, Met, GasThrust, Condense, y_start
# )
