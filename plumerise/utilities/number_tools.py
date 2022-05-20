from math import ceil, floor
from typing import Any, List, Optional

import numpy as np


def isnumber(x: Any) -> bool:
    """Test whether argument is a number

    Args:
        x (Any): input to check if number

    Returns:
        bool: True if argument is number (can be converted to float)
              False if argument is not a number (cannot be converted to float)
    """
    try:
        float(x)
        return True
    except:
        return False


def sci_string(f: np.float_, precision: Optional[int] = 1, start_sci: int = 1) -> str:
    """Generate a string representation of a float point number in scientific notation a x 10^b.

    Args:
        f (np.float_): Number to represent as string
        precision (int, optional): precision (number of digits) of the mantissa. Defaults to 1.
        start_sci (int, optional): do not convert to scientific notation if |b|<start_sci. Defaults to 1.

    Returns:
        str: A string representation of the number in scientific notation.
    """

    if np.isnan(f):
        return "No data"

    f_e = int(np.floor(np.log10(f)))
    f_m = f / (10**f_e)

    if precision is not None:
        if np.abs(f_e) <= start_sci:
            f_str = "{mantissa:.{precision}f}".format(mantissa=f, precision=precision)
        else:
            f_str = "{mantissa:.{precision}f} \u00D7 10<sup>{exponent:d}</sup>".format(
                mantissa=f_m, exponent=f_e, precision=precision
            )
    else:
        if np.abs(f_e) <= start_sci:
            f_str = "{mantissa:f}".format(mantissa=f)
        else:
            f_str = "{mantissa:f} \u00D7 10<sup>{exponent:d}</sup>".format(
                mantissa=f_m, exponent=f_e
            )

    return f_str


def float_string(f, precision=1):

    if np.isnan(f):
        return "No data"

    if precision is not None:
        f_str = "{val:.{precision}f}".format(val=f, precision=precision)
    else:
        f_str = "{val:f}".format(val=f)

    return f_str


def nice_round_up(val, mag=None):
    """Rounds up a number to a nice value of the form a*10^b for integer
        a and b.

    Args:
      val: the number to round.
      **mag: Optional; the value of the exponent b.

    Returns:
      A number rounded.
    """
    if mag is None:
        # mag = 10**int(np.log10(val))
        mag = 10 ** int(floor(np.log10(val)))
    return ceil(val / mag) * mag


def nice_round_down(val: np.float_, mag: Optional[int] = None) -> np.float_:
    """Rounds down a number to a nice value of the form a*10^b for integer
        a and b.

    Args:
      val: the number to round.
      **mag: Optional; the value of the exponent b.

    Returns:
      A number rounded.
    """
    if np.isclose(val, 0.0):
        return 0
    if mag is None:
        expn = int(floor(np.log10(np.abs(val))))
    else:
        expn = mag
    base = 10**expn
    if expn < 0:
        fval = floor(val / base) / (10 ** (-expn))
    else:
        fval = floor(val / base) * (10 ** (expn))

    return fval


def nice_round_nearest(val: np.float_, mag: Optional[int] = None) -> np.float_:

    val_down = nice_round_down(val, mag)
    val_up = nice_round_up(val, mag)

    val_round = np.array([val_down, val_up])

    i = np.abs(val_round - val).argmin()

    return val_round[i]


def log_levels(vmin, vmax):
    mag_low = floor(np.log10(vmin))
    mag_high = floor(np.log10(vmax))

    vals = np.arange(1, 10)
    levels = []
    for mag in range(mag_low, mag_high + 1):
        levels.append(vals * 10**mag)

    levels = np.asarray(levels).flatten()

    return levels[(levels >= vmin) & (levels <= vmax)]


def log_steps(vmin, vmax, step=10, include_max=True):
    if vmin <= 0:
        raise ValueError("vmin must be positive, received {}".format(vmin))
    levels = []
    this_level = vmin
    while this_level < vmax:
        levels.append(this_level)
        this_level *= step

    if include_max:
        if vmax not in levels:
            levels.append(vmax)

    return levels


def step_direction(vmin: np.float_, vmax: np.float_, step: np.float_) -> int:
    fmin = vmin // step
    if np.isclose(fmin, int(fmin)):
        return 1

    fmax = vmax // step
    if np.isclose(fmax, int(fmax)):
        return -1

    return 0


def lin_steps(
    vmin: np.float_,
    vmax: np.float_,
    step: np.float_ = 1.0,
    include_endpoints: bool = True,
    direction: Optional[int] = None,
) -> List[np.float_]:

    if direction is None:
        direction = step_direction(vmin, vmax, step)

    if direction == -1:
        levels = [vmax]
        this_level = vmax
        while this_level > vmin:
            this_level -= step
            if this_level > vmin:
                levels.append(nice_round_nearest(this_level))
        if include_endpoints:
            if vmin not in levels:
                levels.append(vmin)
        levels.sort()

    else:
        levels = [vmin]
        if vmin > 0:
            this_level = 10 ** floor(np.log10(vmin))
        else:
            this_level = 0
        if this_level == 1:
            this_level = 0
        while this_level < vmax:
            this_level += step
            if this_level < vmax:
                levels.append(nice_round_nearest(this_level))

        if include_endpoints:
            if vmax not in levels:
                levels.append(vmax)

    return levels


def lin_levels(vmin, vmax, num=10):
    return np.linspace(vmin, vmax, num)
