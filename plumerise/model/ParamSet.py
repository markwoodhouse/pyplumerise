import numpy as np
from numba.experimental import jitclass
from numba.types import float64, string


def set_params(
    g: float = np.float64(9.81),  # gravitational acceleration [m/s2]
    rhos: float = np.float64(1200.0),  # density of solids [kg/m3]
    rhow: float = np.float64(1000.0),  # density of liquid water [kg/m3]
    rhoi: float = np.float64(900.0),  # density of ice [kg/m3]
    Ca: float = np.float64(998.0),  # Specific heat capacities of dry air [J/kg/K]
    Cs: float = np.float64(1617.0),  # Specific heat capacities of solids [J/kg/K]
    Cv: float = np.float64(1850.0),  # Specific heat capacities of water vapour [J/kg/K]
    Cw: float = np.float64(4200.0),  # Specific heat capacities of liquid water[J/kg/K]
    Ci: float = np.float64(1850.0),  # Specific heat capacities of ice [J/kg/K]
    Ra: float = np.float64(285.0),  # Specific gas constants of dry air  [J/kg/K]
    Rv: float = np.float64(462.0),  # Specific gas constants of water vapour [J/kg/K]
    Lvt0: float = np.float64(2.50e6),  # Latent heat of vapourization at 0 deg C [J/kg].
    Lmt0: float = np.float64(3.34e6),  # Latent heat of melting at 0 deg C [J/kg].
    ks: float = np.float64(0.09),  # No-wind Entrainment coefficient [1].
    kw: float = np.float64(0.9),  # Wind Entrainment coefficient [1].
    Ti0: float = np.float64(263.15),  # Ice temperature bounds.  No ice for T>Ti0, mixed water and ice for Ti00<T<Ti0, all ice for T<Ti00
    Ti00: float = np.float64(233.15),  # Ice temperature bounds.  No ice for T>Ti0, mixed water and ice for Ti00<T<Ti0, all ice for T<Ti00
    Devenish_m: float = np.float64(1.5),  # m-parameter in Devenish entrainment formulation.
):

    params_dtype = np.dtype(
        [
            ("g", np.float64),
            ("rhos", np.float64),
            ("rhow", np.float64),
            ("rhoi", np.float64),
            ("Ca", np.float64),
            ("Cs", np.float64),
            ("Cv", np.float64),
            ("Cw", np.float64),
            ("Ci", np.float64),
            ("Ra", np.float64),
            ("Rv", np.float64),
            ("Lvt0", np.float64),
            ("Lmt0", np.float64),
            ("Lv0", np.float64),
            ("Lm0", np.float64),
            ("ks", np.float64),
            ("kw", np.float64),
            ("Ti0", np.float64),
            ("Ti00", np.float64),
            ("Devenish_m", np.float64),
        ]
    )

    params = np.zeros((1), dtype=params_dtype)

    params["g"][0] = g
    params["rhos"][0] = rhos
    params["rhow"][0] = rhow
    params["rhoi"][0] = rhoi
    params["Ca"][0] = Ca
    params["Cs"][0] = Cs
    params["Cv"][0] = Cv
    params["Cw"][0] = Cw
    params["Ci"][0] = Ci
    params["Ra"][0] = Ra
    params["Rv"][0] = Rv
    params["Lvt0"][0] = Lvt0
    params["Lmt0"][0] = Lmt0
    params["ks"][0] = ks
    params["kw"][0] = kw
    params["Ti0"][0] = Ti0
    params["Ti00"][0] = Ti00
    params["Devenish_m"][0] = Devenish_m

    # Latent heat of vapourization at 0 K assuming Kirchoff's approximation. [J/kg].
    params["Lv0"][0] = params["Lvt0"][0] - np.float64(273.15) * (params["Cw"][0] - params["Cv"][0])
    params["Lm0"][0] = params["Lmt0"][0] - np.float64(273.15) * (params["Ci"][0] - params["Cw"][0])

    return params


# ParamSetSpec = [
#     ("g", np.float64),
#     ("rhos", np.float64),
#     ("rhow", np.float64),
#     ("rhoi", np.float64),
#     ("Ca", np.float64),
#     ("Cs", np.float64),
#     ("Cv", np.float64),
#     ("Cw", np.float64),
#     ("Ci", np.float64),
#     ("Ra", np.float64),
#     ("Rv", np.float64),
#     ("Lvt0", np.float64),
#     ("Lmt0", np.float64),
#     ("Lv0", np.float64),
#     ("Lm0", np.float64),
#     ("ks", np.float64),
#     ("kw", np.float64),
#     ("Ti0", np.float64),
#     ("Ti00", np.float64),
#     ("Entrainment", string),
#     ("Devenish_m", np.float64),
# ]


# @jitclass(ParamSetSpec)
# class ParamsSet(object):
#     g: np.float64
#     rhos: np.float64
#     rhow: np.float64
#     rhoi: np.float64
#     Ca: np.float64
#     Cs: np.float64
#     Cv: np.float64
#     Cw: np.float64
#     Ci: np.float64
#     Ra: np.float64
#     Rv: np.float64
#     Lvt0: np.float64
#     Lmt0: np.float64
#     Lv0: np.float64
#     Lm0: np.float64
#     ks: np.float64
#     kw: np.float64
#     Ti0: np.float64
#     Ti00: np.float64
#     Entrainment: string
#     Devenish_m: np.float64

#     def __init__(
#         self,
#         g: float = 9.81,  # gravitational acceleration [m/s2]
#         rhos: float = 1200.0,  # density of solids [kg/m3]
#         rhow: float = 1000.0,  # density of liquid water [kg/m3]
#         rhoi: float = 900.0,  # density of ice [kg/m3]
#         Ca: float = 998.0,  # Specific heat capacities of dry air [J/kg/K]
#         Cs: float = 1617.0,  # Specific heat capacities of solids [J/kg/K]
#         Cv: float = 1850.0,  # Specific heat capacities of water vapour [J/kg/K]
#         Cw: float = 4200.0,  # Specific heat capacities of liquid water[J/kg/K]
#         Ci: float = 1850.0,  # Specific heat capacities of ice [J/kg/K]
#         Ra: float = 285.0,  # Specific gas constants of dry air  [J/kg/K]
#         Rv: float = 462.0,  # Specific gas constants of water vapour [J/kg/K]
#         Lvt0: float = 2.50e6,  # Latent heat of vapourization at 0 deg C [J/kg].
#         Lmt0: float = 3.34e6,  # Latent heat of melting at 0 deg C [J/kg].
#         ks: float = 0.09,  # No-wind Entrainment coefficient [1].
#         kw: float = 0.9,  # Wind Entrainment coefficient [1].
#         Ti0: float = 263.15,  # Ice temperature bounds.  No ice for T>Ti0, mixed water and ice for Ti00<T<Ti0, all ice for T<Ti00
#         Ti00: float = 233.15,  # Ice temperature bounds.  No ice for T>Ti0, mixed water and ice for Ti00<T<Ti0, all ice for T<Ti00
#         Entrainment: str = "Hewett",  # Choice for entrainment formulation: can be 'Hewett' or 'Devenish'.
#         Devenish_m: float = 1.5,  # m-parameter in Devenish entrainment formulation.
#     ):
#         self.g = g
#         self.rhos = rhos
#         self.rhow = rhow
#         self.rhoi = rhoi
#         self.Ca = Ca
#         self.Cs = Cs
#         self.Cv = Cv
#         self.Cw = Cw
#         self.Ci = Ci
#         self.Ra = Ra
#         self.Rv = Rv
#         self.Lvt0 = Lvt0
#         self.Lmt0 = Lmt0
#         self.ks = ks
#         self.kw = kw
#         self.Ti0 = Ti0
#         self.Ti00 = Ti00
#         self.Entrainment = Entrainment
#         self.Devenish_m = Devenish_m

#         # Latent heat of vapourization at 0 K assuming Kirchoff's approximation. [J/kg].
#         self.Lv0 = self.Lvt0 - 273.15 * (self.Cw - self.Cv)
#         self.Lm0 = self.Lmt0 - 273.15 * (self.Ci - self.Cw)

#         return
