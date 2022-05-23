from datetime import datetime
from functools import cache

import metpy as metpy
import metpy.calc as mpcalc
import metpy.units as mpunits
import numpy as np
from numba import jit, njit
from numba.experimental import jitclass
from numba.types import float64, int64, string
from siphon.simplewebservice.wyoming import WyomingUpperAir


@njit(cache=True)
def locate(xx: np.ndarray, x: float64):
    """locate: From Numerical Recipes in Fortran 90
        Given an array xx, and given a value x, returns a value j such that x is between
        xx(j) and xx(j + 1). xx must be monotonic, either increasing or decreasing. j = 0 or
        j = N is returned to indicate that x is out of range.
        Inputs: xx [real array] - data list; x [real] - target value.
        Output: j [integer] - location in list xx with xx(j)<x<xx(j+1)
    """
    
    n = len(xx)
    ascnd = (xx[-1] >= xx[0]) # True if ascending order of table, false otherwise.
    jl=0     # Initialize lower
    ju=n+1   # and upper limits.
    while (ju-jl > 1):
      jm=(ju+jl)//2 # Compute a midpoint,
      if (ascnd == (x >= xx[jm])):
        jl=jm # and replace either the lower limit
      else:
        ju=jm # or the upper limit, as appropriate.

    if (x == xx[0]): # Then set the output, being careful with the endpoints.
      locate=1
    elif (x == xx[-1]):
      locate=n-1
    else:
      locate=jl
    
    return locate

@njit(cache=True)
def hunt(xx: np.ndarray, x: float64, jlo: int64):
    """hunt: From Numerical Recipes in Fortran 90
        Given an array xx, and given a value x, returns a value jlo such that x is between
        xx(jlo) and xx(jlo+1). xx must be monotonic, either increasing or decreasing. jlo = 0
        or jlo = N is returned to indicate that x is out of range. jlo on input is taken as the
        initial guess for jlo on output.
        Inputs: jlo [integer] - guess for position in data list.; 
                x [real] - target value; xx [real array] - data list.
        Output: jlo [integer] - location in list xx with xx(jlo)<x<xx(jlo+1)
    """

    
    n = len(xx)
    ascnd = (xx[-1] >= xx[0]) # True if ascending order of table, false otherwise.
    if (jlo <= 0) or (jlo > n): # Input guess not useful. Go immediately to bisection.
        jhi=n+1
    else:
        inc=1 # Set the hunting increment.
        if ((x >= xx[jlo]) == ascnd): # Hunt up:
            while True:
                jhi=jlo+inc
                if (jhi > n): # Done hunting, since off end of table.
                    jhi=n+1
                    break
                else:
                    if ((x < xx[jhi]) == ascnd): break
                    jlo=jhi # Not done hunting,
                    inc=inc+inc # so double the increment
        else: # Hunt down
            jhi=jlo
            while True:
                jlo=jhi-inc
                if (jlo < 0): # Done hunting, since off end of table.
                    jlo=0
                    break
                else:
                    if (x >= xx[jlo] == ascnd): break
                    jhi=jlo # Not done hunting,
                    inc=inc+inc # so double the increment
    
    while True: # Hunt is done, so begin the final bisection phase:
        if (jhi-jlo <= 1):
            if (x == xx[-1]): jlo=np.int64(n)
            if (x == xx[0]): jlo=np.int64(1)
            break
        else:
            jm=(jhi+jlo)//2
            if ((x >= xx[jm]) == ascnd):
                jlo=jm
            else:
                jhi=jm
    
    return jlo

@njit(cache=True)
def interp1angle(xy_list: np.ndarray, x_req: float64, jpos: int64):
    """ Interpolation within an angle data array where y data is contrained to be between 0 and 2*pi, or constant extrapolation outside.
        Inputs: xy_list - contains x values in xy_list[:,0] and y values in xy_list[:,1];
                x_req [real] - the required x value;
                jpos [integer] - a guess for where x-value would appear in the list (i.e. xy_list(jpos,0)<x_req<xy_list(jpos+1,0));
    
        Outputs: y_req [real] - the required y value from linear interpolation, or extrapolation;
                 jpos [integer] - the location in the list with xy_list(jpos,1)<x_req<xy_list(jpos+1,1).
    """
        
    Nx = len(xy_list[:,0])
      
    if (x_req < xy_list[0,0]):
        x1 = xy_list[0,0]
        x2 = xy_list[1,0]
        y1 = xy_list[0,1]
        y2 = xy_list[1,1]
    elif (x_req > xy_list[-1,0]):
        x1 = xy_list[-1,0]
        x2 = x_req
        y1 = xy_list[-1,1]
        y2 = xy_list[-1,1]          
    else:
        if (jpos>Nx):
            j = locate(xy_list[:,0], x_req)
            x1 = xy_list[j,0]
            x2 = xy_list[j+1,0]
            y1 = xy_list[j,1]
            y2 = xy_list[j+1,1]
        else:
          jpos = hunt(xy_list[:,0], x_req, jpos)
          x1 = xy_list[jpos,0]
          x2 = xy_list[jpos+1,0]
          y1 = xy_list[jpos,1]
          y2 = xy_list[jpos+1,1]
      
    if (np.abs(y2-y1)>np.pi):
        if (y2>y1):
            y1 = y1+2.0*np.pi
        else:
            y2 = y2+2.0*np.pi
      
    y_req = y1 + (y2-y1)*(x_req-x1)/(x2-x1)
        
    return y_req, jpos


@njit(cache=True)
def interp1(xy_list: np.ndarray, x_req: float64, jpos: int64): 
    """Linear interpolation within a data array, or extrapolation outside.
       Inputs: xy_list - contains x values in xy_list[:,0] and y values in xy_list[:,1];
               x_req - the required x value;
               jpos - a guess for where x-value would appear in the list (i.e. xy_list[jpos,0]<x_req<xy_list[jpos+1,0]);
       Outputs: y_req [float64] - the required y value from linear interpolation, or extrapolation;
                jpos [integer] - the location in the list with xy_list[jpos,1]<y_req<xy_list[jpos+1,1].
    """
          
    Nx = len(xy_list[:,0])
      
    if (Nx==1):
        y_req = xy_list[0,1]
        return y_req, 0
                        
    if (x_req<xy_list[0,0]):
        x1 = xy_list[0,0]
        x2 = xy_list[1,0]
        y1 = xy_list[0,1]
        y2 = xy_list[1,1]
    elif (x_req>xy_list[-1,0]):
        x1 = xy_list[-1,0]
        x2 = x_req
        y1 = xy_list[-1,1]
        y2 = xy_list[-1,1]
    else:
        if (jpos>Nx):
            j = locate(xy_list[:,0], x_req)
            x1 = xy_list[j,0]
            x2 = xy_list[j+1,0]
            y1 = xy_list[j,1]
            y2 = xy_list[j+1,1]
        else:
            jpos = hunt(xy_list[:,0], x_req, jpos)
            x1 = xy_list[jpos,0]
            x2 = xy_list[jpos+1,0]
            y1 = xy_list[jpos,1]
            y2 = xy_list[jpos+1,1]
      
    y_req = y1 + (y2-y1)*(x_req-x1)/(x2-x1)
        
    return y_req, jpos


@njit(cache=True)
def interpexp(xy_list: np.ndarray, x_req: float64, jpos: int64):
    """Exponential interpolation within a data array, or exponential extrapolation outside.
    Inputs: xy_list - contains x values in xy_list[:,0] and y values in xy_list[:,1];
            x_req - the required x value;
            jpos - a guess for where x-value would appear in the list (i.e. xy_list[jpos,0]<x_req<xy_list[jpos+1,0]);
    Outputs: y_req [float64] - the required y value from linear interpolation, or extrapolation;
             jpos [integer] - the location in the list with xy_list[jpos,1]<y_req<xy_list[jpos+1,1].
    """
        
    Nx = len(xy_list[:,0])
      
    if (x_req<xy_list[0,0]):
        x1 = xy_list[0,0]
        x2 = xy_list[1,0]
        y1 = xy_list[0,1]
        y2 = xy_list[1,1]
    elif (x_req>xy_list[-1,0]):
        x1 = xy_list[-2,0]
        x2 = xy_list[-1,0]
        y1 = xy_list[-2,1]
        y2 = xy_list[-1,1]
    else:
        if (jpos>Nx):
            j = locate(xy_list[:,0], x_req)
            x1 = xy_list[j,0]
            x2 = xy_list[j+1,0]
            y1 = xy_list[j,1]
            y2 = xy_list[j+1,1]
        else:
            jpos = hunt(xy_list[:,0], x_req, jpos)
            x1 = xy_list[jpos,0]
            x2 = xy_list[jpos+1,0]
            y1 = xy_list[jpos,1]
            y2 = xy_list[jpos+1,1]
      
    logy1 = np.log(y1)
    logy2 = np.log(y2)
      
    logy_req = logy1 + (logy2-logy1)*(x_req-x1)/(x2-x1)
      
    y_req = np.exp(logy_req)
        
    return y_req, jpos


MetDataArraysSpec = [
    # MetData stores meteorological data.
    # Each variable has values as function of height
    ('height', float64[:]),
    ('temperature', float64[:,:]),
    ('pressure', float64[:,:]),
    ('direction', float64[:,:]),
    ('speed', float64[:,:]),
    ('relhum', float64[:,:]),
]

#@jitclass([('height', float64[:,:]), ('temperature', float64[:,:])])
@jitclass(MetDataArraysSpec)
class MetDataArrays(object):
    height: np.ndarray
    temperature: np.ndarray
    pressure: np.ndarray
    direction: np.ndarray
    speed: np.ndarray
    relhum: np.ndarray

    def __init__(self, height: np.ndarray, temperaure: np.ndarray, pressure: np.ndarray, direction: np.ndarray, speed: np.ndarray, relhum: np.ndarray):
        # self.temperature = np.zeros((len(temperaure,2)), dtype=np.float64)
        # self.pressure = np.zeros((len(pressure,2)), dtype=np.float64)
        # self.direction = np.zeros((len(direction,2)), dtype=np.float64)
        # self.speed = np.zeros((len(speed,2)), dtype=np.float64)
        # self.relhum = np.zeros((len(relhum,2)), dtype=np.float64)

        self.height = np.zeros((len(height)), dtype=np.float64)
        self.temperature = np.zeros((len(temperaure),2), dtype=np.float64)
        self.pressure = np.zeros((len(pressure),2), dtype=np.float64)
        self.direction = np.zeros((len(direction),2), dtype=np.float64)
        self.speed = np.zeros((len(speed),2), dtype=np.float64)
        self.relhum = np.zeros((len(relhum),2), dtype=np.float64)

        self.height[:] = height

        self.temperature[:,0] = height
        self.temperature[:,1] = temperaure


        self.pressure[:,0] = height
        self.pressure[:,1] = pressure

        self.direction[:,0] = height
        self.direction[:,1] = direction

        self.speed[:,0] = height
        self.speed[:,1] = speed

        self.relhum[:,0] = height
        self.relhum[:,1] = relhum

        return

    def interp(self, z: float64, z_pos: int64):
        """MetDataArrays.interp : interpolate met data
        
            Inputs:
                z [float] : height to get data
                z_pos [integer] : estimate of the index of z in the data set
            
            Outputs:
                vals [dict] : dictionary of float values for each variable
                z_pos [integer] : update index of z in the data set
        
        """

        vals = dict()

        psia, z_pos = interp1angle(self.direction, z, z_pos) # Interpolate to find wind direction - constant extrapolation.
        V, z_pos = interp1(self.speed, z, z_pos) # Interpolate to find wind speed - constant extrapolation.
        RH, z_pos = interp1(self.relhum, z, z_pos) # Interpolate to find relative humidity - constant extrapolation.
        Ta, z_pos = interp1(self.temperature, z, z_pos) # Interpolate to find temperature - constant extrapolation.
        Pa, z_pos = interpexp(self.pressure, z, z_pos) # Interpolate to find pressure - exponential interpolation and extrapolation.

        vals['psia'] = psia
        vals['V'] = V
        vals['RH'] = RH
        vals['Ta'] = Ta
        vals['Pa'] = Pa

        #! Find temperature gradient
        if (z_pos<0):
            raise ValueError('requested z in MetFieldInterp is below lowest height in Met data.')
        elif (z_pos < self.temperature.shape[0]):
            dTadz = (self.temperature[z_pos,1]-self.temperature[z_pos-1,1])/(self.temperature[z_pos,0]-self.temperature[z_pos-1,0])
        else:
            dTadz = 0.0
        vals['dTadz'] = dTadz
      
        if (z_pos+1 < len(self.pressure[:,0])):
            L = (np.log(self.pressure[z_pos+1,1]) - np.log(self.pressure[z_pos,1]))/(self.pressure[z_pos+1,0]-self.pressure[z_pos,0])
            logA = np.log(self.pressure[z_pos,1]) - L*self.pressure[z_pos,0]
        else:
            L = (np.log(self.pressure[z_pos,1]) - np.log(self.pressure[z_pos-1,1]))/(self.pressure[z_pos,0]-self.pressure[z_pos-1,0])
            logA = np.log(self.pressure[z_pos-1,1]) - L*self.pressure[z_pos-1,0]
        A = np.exp(logA)
        dPadz = L*A*np.exp(L*z)
        vals['dPadz'] = dPadz
      
        return vals, z_pos


@njit
def esfunc(T: float64, Ti0: float64, Ti00: float64):
    """
        Saturation vapour pressure over water, ice or mixture
    """
      
    if (T >= Ti0):
        f = 1.0
        dfdT = 0.0
    elif (T <= Ti00):
        f = 0.0
        dfdT = 0.0
    else:
        f = (T-Ti00)/(Ti0-Ti00)
        dfdT = 1.0/(Ti0-Ti00)
      
    esw, deswdT = eswfunc(T)
    esi, desidT = esifunc(T)
    
    es = f*esw + (1.0-f)*esi
    desdT = f*deswdT + (1.0-f)*desidT + (esw-esi)*dfdT
              
    return es,desdT


@njit
def eswfunc(T: float64):
    """Saturation vapour pressure over water
    [Alduchov & Eskridge 1996]"""
    
    esw0 = 610.94 # Pa
    Aw = 17.625
    Bw = 243.04
    
    # Magnus formula
    tdegC = T - 273.15
    
    esw = esw0*np.exp(Aw*tdegC/(Bw+tdegC))
    
    deswdT = esw*Aw*Bw/(Bw+tdegC)/(Bw+tdegC)
          
    return esw, deswdT


@njit
def esifunc(T: float64):
    
    """Saturation vapour pressure over ice
    [Alduchov & Eskridge 1996]"""

    esi0 = 611.21 # Pa
    Ai = 22.587
    Bi = 273.86
    
    # Magnus formula
    tdegC = T - 273.15
      
    esi = esi0*np.exp(Ai*tdegC/(Bi+tdegC))
      
    desidT = esi*Ai*Bi/(Bi+tdegC)/(Bi+tdegC)
              
    return esi, desidT



@njit    
def SpecHumidity(RH, esa, Pa, Ra, Rv):
    wa = RH*esa*Ra/(Rv*Pa - RH*esa*(Rv-Ra))
    return wa




class MetData(object):

    def __init__(self, Date, Time,
                StationID = '',
                MetType='', MetSource=''
                ):

        match MetType.lower():
            case 'radiosonde':
                self.MetType = 'Radiosonde'
                self.MetSource = MetSource
                if MetSource=='Wyoming':
                    self.Date = Date
                    self.Time = Time
                    self.StationID = StationID
                    self.Data = self.get_wyoming()
            case 'nwp':
                self.MetType = 'NWP'
            case _:
                raise ValueError("In MetData initialization 'MetType' not recognized. \n "
                                 + " 'MetType' must be one of 'Radiosonde', 'NWP'")
            
    def get_wyoming(self):
        """get_wyoming: get radiosonde data from Wyoming Weather Web at specified date, time and station.

        Inputs:
            date [string]: date in ISO format YYYY-MM-DD
            time [string]: time in HH:MM format
            station [string]: station label

        Returns:
            df: DataFrame with radiosonde data

        """

        date = datetime.fromisoformat(self.Date+' '+self.Time)

        df = WyomingUpperAir.request_data(date, self.StationID)
        df.dropna(axis=0, inplace=True)

        temperature = df['temperature'].values * mpunits.units(df.units['temperature'])
        pressure = df['pressure'].values * mpunits.units(df.units['pressure'])
        height = df['height'].values * mpunits.units(df.units['height'])
        dewpoint = df['dewpoint'].values * mpunits.units(df.units['dewpoint'])
        direction = df['direction'].values * mpunits.units(df.units['direction'])
        speed = df['speed'].values * mpunits.units(df.units['speed'])

        relhum = mpcalc.relative_humidity_from_dewpoint(temperature, dewpoint)

        temperature = temperature.to(mpunits.units.K)
        pressure = pressure.to(mpunits.units.Pa)
        height = height.to(mpunits.units.m)
        direction = direction.to(mpunits.units.radian)
        speed = speed.to(mpunits.units.mps)
        relhum = relhum.to(mpunits.units.dimensionless)

        met_types = np.dtype([
            ('temperature', temperature.dtype),
            ('pressure', pressure.dtype),
            ('height', height.dtype),
            ('direction', direction.dtype),
            ('speed', speed.dtype),
            ('relhum', relhum.dtype),
        ])

        metdata = np.empty(len(height), dtype=met_types)
        metdata['temperature'] = temperature
        metdata['pressure'] = pressure
        metdata['height'] = height
        metdata['direction'] = direction
        metdata['speed'] = speed
        metdata['relhum'] = relhum

        return metdata

