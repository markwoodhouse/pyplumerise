import numpy as np
from numba.experimental import jitclass
from numba.types import float64, string

ParamSetSpec = [
    ("g", float64),
    ("rhos", float64),
    ("rhow", float64),
    ("rhoi", float64),
    ("Ca", float64),
    ("Cs", float64),
    ("Cv", float64),
    ("Cw", float64),
    ("Ci", float64),
    ("Ra", float64),
    ("Rv", float64),
    ("Lvt0", float64),
    ("Lmt0", float64),
    ("Lv0", float64),
    ("Lm0", float64),
    ("ks", float64),
    ("kw", float64),
    ("Ti0", float64),
    ("Ti00", float64),
    ("Entrainment", string),
    ("Devenish_m", float64),
]


@jitclass(ParamSetSpec)
class ParamsSet(object):
    g: float64
    rhos: float64
    rhow: float64
    rhoi: float64
    Ca: float64
    Cs: float64
    Cv: float64
    Cw: float64
    Ci: float64
    Ra: float64
    Rv: float64
    Lvt0: float64
    Lmt0: float64
    Lv0: float64
    Lm0: float64
    ks: float64
    kw: float64
    Ti0: float64
    Ti00: float64
    Entrainment: string
    Devenish_m: float64

    def __init__(
        self,
        g: float = 9.81,  # gravitational acceleration [m/s2]
        rhos: float = 1200.0,  # density of solids [kg/m3]
        rhow: float = 1000.0,  # density of liquid water [kg/m3]
        rhoi: float = 900.0,  # density of ice [kg/m3]
        Ca: float = 998.0,  # Specific heat capacities of dry air [J/kg/K]
        Cs: float = 1617.0,  # Specific heat capacities of solids [J/kg/K]
        Cv: float = 1850.0,  # Specific heat capacities of water vapour [J/kg/K]
        Cw: float = 4200.0,  # Specific heat capacities of liquid water[J/kg/K]
        Ci: float = 1850.0,  # Specific heat capacities of ice [J/kg/K]
        Ra: float = 285.0,  # Specific gas constants of dry air  [J/kg/K]
        Rv: float = 462.0,  # Specific gas constants of water vapour [J/kg/K]
        Lvt0: float = 2.50e6,  # Latent heat of vapourization at 0 deg C [J/kg].
        Lmt0: float = 3.34e6,  # Latent heat of melting at 0 deg C [J/kg].
        ks: float = 0.09,  # No-wind Entrainment coefficient [1].
        kw: float = 0.9,  # Wind Entrainment coefficient [1].
        Ti0: float = 263.15,  # Ice temperature bounds.  No ice for T>Ti0, mixed water and ice for Ti00<T<Ti0, all ice for T<Ti00
        Ti00: float = 233.15,  # Ice temperature bounds.  No ice for T>Ti0, mixed water and ice for Ti00<T<Ti0, all ice for T<Ti00
        Entrainment: str = "Hewett",  # Choice for entrainment formulation: can be 'Hewett' or 'Devenish'.
        Devenish_m: float = 1.5,  # m-parameter in Devenish entrainment formulation.
    ):
        self.g = g
        self.rhos = rhos
        self.rhow = rhow
        self.rhoi = rhoi
        self.Ca = Ca
        self.Cs = Cs
        self.Cv = Cv
        self.Cw = Cw
        self.Ci = Ci
        self.Ra = Ra
        self.Rv = Rv
        self.Lvt0 = Lvt0
        self.Lmt0 = Lmt0
        self.ks = ks
        self.kw = kw
        self.Ti0 = Ti0
        self.Ti00 = Ti00
        self.Entrainment = Entrainment
        self.Devenish_m = Devenish_m

        # Latent heat of vapourization at 0 K assuming Kirchoff's approximation. [J/kg].
        self.Lv0 = self.Lvt0 - 273.15 * (self.Cw - self.Cv)
        self.Lm0 = self.Lmt0 - 273.15 * (self.Ci - self.Cw)

        return
