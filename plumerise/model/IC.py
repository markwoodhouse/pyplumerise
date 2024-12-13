import numpy as np
from typing import Optional

def set_IC(*, 
    L0: Optional[float] = None,  # vent radius [m]
    U0: Optional[float] = None,  # exit velocity [m/s]
    n0: Optional[float] = None,  # gas mass fraction [1] -- if this is set, it corresponds to phia0=0, phiv0 = n0, phis0 = 1-n0, phiv0=0, phiw0=0
    T0: Optional[float] = None,  # temperature at the vent [K]
    z0: float = 0.0,  # vent altitude [m]
    Q0: Optional[float] = None,
    M0: Optional[float] = None,
    E0: Optional[float] = None,
    phia0: Optional[float] = None,
    phiv0: Optional[float] = None,
    phis0: Optional[float] = None,
    phiw0: Optional[float] = None,
    phii0: Optional[float] = None,
    theta0: float = 0.5 * np.pi, # plume inclination
    with_pi: bool = True,
):

    IC = dict()

    non_none_count = sum(1 for _ in filter(lambda x: x is not None, [L0, U0, Q0, M0]))
    if non_none_count != 2:
        raise ValueError(f'Only two of L0, U0, Q0, M0 should be given, received {non_none_count}')

    if L0 is not None:
        if L0<=0:
            raise ValueError("Must have L0 > 0")
    IC["L0"] = np.float64(L0)

    if U0 is not None:
        if U0<=0:
            raise ValueError("Must have U0 > 0")
    IC["U0"] = np.float64(U0)

    if Q0 is not None:
        if Q0<=0:
            raise ValueError("Must have Q0 > 0")
        if not with_pi: Q0 = np.pi * Q0
    IC["Q0"] = np.float64(Q0)

    if M0 is not None:
        if M0<=0:
            raise ValueError("Must have M0 > 0")
        if not with_pi: M0 = np.pi * M0
    IC["M0"] = np.float64(M0)

    if n0 is not None:
        if 0<n0<1:
            check_fraction("n0", n0, required=False)
        if phia0 is not None: raise UserWarning('n0 is set, so ignoring phia0')
        if phiv0 is not None: raise UserWarning('n0 is set, so ignoring phiv0')
        if phis0 is not None: raise UserWarning('n0 is set, so ignoring phis0')
        if phiw0 is not None: raise UserWarning('n0 is set, so ignoring phiw0')
        if phii0 is not None: raise UserWarning('n0 is set, so ignoring phii0')
        IC["phia0"] = np.float64(0.0)
        IC["phiv0"] = np.float64(n0)
        IC["phis0"] = np.float64(1.0 - n0)
        IC["phiw0"] = np.float64(0.0)
        IC["phii0"] = np.float64(0.0)
    else:
        check_fraction("phia0", phia0, required=True)
        IC["phia0"] = np.float64(phia0)
        check_fraction("phiv0", phiv0, required=True)
        IC["phiv0"] = np.float64(phiv0)
        check_fraction("phis0", phis0, required=True)
        IC["phis0"] = np.float64(phis0)
        if phiw0 is None:
            IC["phiw0"] = np.float64(0.0)
        else:
            if phiw0<0 or phiw0>1: raise ValueError("Must have 0<phiw0<1")
        if phii0 is None:
            IC["phii0"] = np.float64(0.0)
        else:
            if phii0<0 or phii0>1: raise ValueError("Must have 0<phii0<1")
        
        if IC["phia0"] + IC["phiv0"] + IC["phis0"] + IC["phiw0"] + IC["phii0"] != 1:
            raise ValueError("Must have phia0 + phiv0 + phis0 + phiw0 + phii0 = 1")

    if all(v is None for v in [T0,E0]):
        raise ValueError("Either T0 or E0 is required")
    
    if all(v is not None for v in [T0, E0]):
        raise UserWarning("T0 is set so ignoring E0")
    
    if T0 is not None:
        if T0<=0:
            raise ValueError("Must have T0>0")
        else:
            IC["T0"] = np.float64(T0)
    else:
        if E0<=0:
            raise ValueError("Must have E0>0")
        else:
            if not with_pi: E0 = np.pi * E0
            IC["E0"] = np.float64(E0)

    IC["theta0"] = np.float64(theta0)
    IC["z0"] = np.float64(z0)

    return IC

def check_fraction(name: str, val: float | None, required=True):
    if required:
        if val is None:
            raise ValueError(f"{name} not set")
    else:
        if val<0 or val>1:
            raise ValueError(f"Must have 0<{name}<1")