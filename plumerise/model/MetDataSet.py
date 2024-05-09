from datetime import datetime
from functools import cache
from math import ceil, floor
import os
import re

from typing import Optional

import metpy as metpy
import metpy.calc as mpcalc
import metpy.units as mpunits
import numpy as np
from numba import jit, njit
# from numba.experimental import jitclass
from numba.types import float64, int64, string
from siphon.simplewebservice.wyoming import WyomingUpperAir
import cdsapi
from netCDF4 import Dataset


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
def interp1angle(x: np.ndarray, y: np.ndarray, x_req: float64, jpos: int64):
    """ Interpolation within an angle data array where y data is contrained to be between 0 and 2*pi, or constant extrapolation outside.
        Inputs: x [real[:]] -  contains x values;
                y [real[:]] -  contains y values;
                x_req [real] - the required x value;
                jpos [integer] - a guess for where x-value would appear in the list (i.e. xy_list(jpos,0)<x_req<xy_list(jpos+1,0));
    
        Outputs: y_req [real] - the required y value from linear interpolation, or extrapolation;
                 jpos [integer] - the location in the list with xy_list(jpos,1)<x_req<xy_list(jpos+1,1).
    """
        
    Nx = x.shape[0]
      
    if (x_req < x[0]):
        x1 = x[0]
        x2 = x[1]
        y1 = y[0]
        y2 = y[1]
    elif (x_req > x[-1]):
        x1 = x[-1]
        x2 = x_req
        y1 = y[-1]
        y2 = y[-1]          
    else:
        if (jpos>Nx):
            j = locate(x, x_req)
            x1 = x[j]
            x2 = x[j+1]
            y1 = y[j]
            y2 = y[j+1]
        else:
          jpos = hunt(x[:], x_req, jpos)
          x1 = x[jpos]
          x2 = x[jpos+1]
          y1 = y[jpos]
          y2 = y[jpos+1]
      
    if (np.abs(y2-y1)>np.pi):
        if (y2>y1):
            y1 = y1+2.0*np.pi
        else:
            y2 = y2+2.0*np.pi
      
    y_req = y1 + (y2-y1)*(x_req-x1)/(x2-x1)
        
    return y_req, jpos


@njit(cache=True)
def interp1(x: np.ndarray, y: np.ndarray, x_req: float64, jpos: int64): 
    """Linear interpolation within a data array, or extrapolation outside.
       Inputs: x [real[:]] -  contains x values;
               y [real[:]] -  contains y values;
               x_req - the required x value;
               jpos - a guess for where x-value would appear in the list (i.e. xy_list[jpos,0]<x_req<xy_list[jpos+1,0]);
       Outputs: y_req [float64] - the required y value from linear interpolation, or extrapolation;
                jpos [integer] - the location in the list with xy_list[jpos,1]<y_req<xy_list[jpos+1,1].
    """
          
    Nx = x.shape[0]
      
    if (Nx==1):
        y_req = y[0]
        return y_req, 0
                        
    if (x_req<x[0]):
        x1 = x[0]
        x2 = x[1]
        y1 = y[0]
        y2 = y[1]
    elif (x_req>x[-1]):
        x1 = x[-1]
        x2 = x_req
        y1 = y[-1]
        y2 = y[-1]
    else:
        if (jpos>Nx):
            j = locate(x, x_req)
            x1 = x[j]
            x2 = x[j+1]
            y1 = y[j]
            y2 = y[j+1]
        else:
            jpos = hunt(x, x_req, jpos)
            x1 = x[jpos]
            x2 = x[jpos+1]
            y1 = y[jpos]
            y2 = y[jpos+1]
      
    y_req = y1 + (y2-y1)*(x_req-x1)/(x2-x1)
        
    return y_req, jpos


@njit(cache=True)
def interpexp(x: np.ndarray, y: np.ndarray, x_req: float64, jpos: int64):
    """Exponential interpolation within a data array, or exponential extrapolation outside.
    Inputs: x [real[:]] -  contains x values;
            y [real[:]] -  contains y values;
            x_req - the required x value;
            jpos - a guess for where x-value would appear in the list (i.e. xy_list[jpos,0]<x_req<xy_list[jpos+1,0]);
    Outputs: y_req [float64] - the required y value from linear interpolation, or extrapolation;
             jpos [integer] - the location in the list with xy_list[jpos,1]<y_req<xy_list[jpos+1,1].
    """
        
    Nx = x.shape[0]
      
    if (x_req<x[0]):
        x1 = x[0]
        x2 = x[1]
        y1 = y[0]
        y2 = y[1]
    elif (x_req>x[-1]):
        x1 = x[-2]
        x2 = x[-1]
        y1 = y[-2]
        y2 = y[-1]
    else:
        if (jpos>Nx):
            j = locate(x, x_req)
            x1 = x[j]
            x2 = x[j+1]
            y1 = y[j]
            y2 = y[j+1]
        else:
            jpos = hunt(x, x_req, jpos)
            x1 = x[jpos]
            x2 = x[jpos+1]
            y1 = y[jpos]
            y2 = y[jpos+1]
      
    logy1 = np.log(y1)
    logy2 = np.log(y2)
      
    logy_req = logy1 + (logy2-logy1)*(x_req-x1)/(x2-x1)
      
    y_req = np.exp(logy_req)
        
    return y_req, jpos


def MetDataArray(height: np.ndarray, temperaure: np.ndarray, pressure: np.ndarray, direction: np.ndarray, speed: np.ndarray, relhum: np.ndarray):
        
        Met = np.zeros((height.shape[0], 6), dtype=np.float64)
        Met[:,0] = height

        Met[:,1] = temperaure
        Met[:,2] = pressure
        Met[:,3] = direction
        Met[:,4] = speed
        Met[:,5] = relhum

        return Met

@njit(cache=True)
def Met_interp(Met: np.ndarray, z_in: float64, z_pos: int64):
        """MetDataArrays.interp : interpolate met data
        
            Inputs:
                z [float] : height to get data
                z_pos [integer] : estimate of the index of z in the data set
            
            Outputs:
                vals [dict] : dictionary of float values for each variable
                z_pos [integer] : update index of z in the data set
        
        """

        # vals_type = np.dtype([
        #     ('psia', np.float64),
        #     ('V', np.float64),
        #     ('RH', np.float64),
        #     ('Ta', np.float64),
        #     ('Pa', np.float64),
        #     ('dTadz', np.float64),
        #     ('dPadz', np.float64),
        # ])

        z = Met[:,0]
        Nz = z.shape[0]

        psia, z_pos = interp1angle(z, Met[:,3], z_in, z_pos) # Interpolate to find wind direction - constant extrapolation.
        V, z_pos = interp1(z, Met[:,4], z_in, z_pos) # Interpolate to find wind speed - constant extrapolation.
        RH, z_pos = interp1(z, Met[:,5], z_in, z_pos) # Interpolate to find relative humidity - constant extrapolation.
        Ta, z_pos = interp1(z, Met[:,1], z_in, z_pos) # Interpolate to find temperature - constant extrapolation.
        Pa, z_pos = interpexp(z, Met[:,2], z_in, z_pos) # Interpolate to find pressure - exponential interpolation and extrapolation.

        vals = dict()

        vals['psia'] = psia
        vals['V'] = V
        vals['RH'] = RH
        vals['Ta'] = Ta
        vals['Pa'] = Pa

        temperature = Met[:,1]
        #! Find temperature gradient
        if (z_pos<0):
            raise ValueError('requested z in MetFieldInterp is below lowest height in Met data.')
        elif (z_pos < Nz):
            dTadz = (temperature[z_pos]-temperature[z_pos-1])/(z[z_pos]-z[z_pos-1])
        else:
            dTadz = 0.0
        vals['dTadz'] = dTadz
      
        # Find pressure gradient
        pressure = Met[:,2]
        if (z_pos+1 < Nz):
            L = (np.log(pressure[z_pos+1]) - np.log(pressure[z_pos]))/(z[z_pos+1]-z[z_pos])
            logA = np.log(pressure[z_pos]) - L*z[z_pos]
        else:
            L = (np.log(pressure[z_pos]) - np.log(pressure[z_pos-1]))/(z[z_pos]-z[z_pos-1])
            logA = np.log(pressure[z_pos-1]) - L*z[z_pos-1]
        A = np.exp(logA)
        dPadz = L*A*np.exp(L*z_in)
        vals['dPadz'] = dPadz
      
        return vals, z_pos


# MetDataArraysSpec = np.dtype([
#     # MetData stores meteorological data.
#     # Each variable has values as function of height
#     ('height', float64[:]),
#     ('temperature', float64[:,:]),
#     ('pressure', float64[:,:]),
#     ('direction', float64[:,:]),
#     ('speed', float64[:,:]),
#     ('relhum', float64[:,:]),
# ])

#@jitclass([('height', float64[:,:]), ('temperature', float64[:,:])])
# @jitclass(MetDataArraysSpec)
# class MetDataArrays(object):
#     height: np.ndarray
#     temperature: np.ndarray
#     pressure: np.ndarray
#     direction: np.ndarray
#     speed: np.ndarray
#     relhum: np.ndarray

#     def __init__(self, height: np.ndarray, temperaure: np.ndarray, pressure: np.ndarray, direction: np.ndarray, speed: np.ndarray, relhum: np.ndarray):
#         # self.temperature = np.zeros((len(temperaure,2)), dtype=np.float64)
#         # self.pressure = np.zeros((len(pressure,2)), dtype=np.float64)
#         # self.direction = np.zeros((len(direction,2)), dtype=np.float64)
#         # self.speed = np.zeros((len(speed,2)), dtype=np.float64)
#         # self.relhum = np.zeros((len(relhum,2)), dtype=np.float64)

#         self.height = np.zeros((len(height)), dtype=np.float64)
#         self.temperature = np.zeros((len(temperaure),2), dtype=np.float64)
#         self.pressure = np.zeros((len(pressure),2), dtype=np.float64)
#         self.direction = np.zeros((len(direction),2), dtype=np.float64)
#         self.speed = np.zeros((len(speed),2), dtype=np.float64)
#         self.relhum = np.zeros((len(relhum),2), dtype=np.float64)

#         self.height[:] = height

#         self.temperature[:,0] = height
#         self.temperature[:,1] = temperaure


#         self.pressure[:,0] = height
#         self.pressure[:,1] = pressure

#         self.direction[:,0] = height
#         self.direction[:,1] = direction

#         self.speed[:,0] = height
#         self.speed[:,1] = speed

#         self.relhum[:,0] = height
#         self.relhum[:,1] = relhum

#         return

#     def interp(self, z: float64, z_pos: int64):
#         """MetDataArrays.interp : interpolate met data
        
#             Inputs:
#                 z [float] : height to get data
#                 z_pos [integer] : estimate of the index of z in the data set
            
#             Outputs:
#                 vals [dict] : dictionary of float values for each variable
#                 z_pos [integer] : update index of z in the data set
        
#         """

#         vals = dict()

#         psia, z_pos = interp1angle(self.direction, z, z_pos) # Interpolate to find wind direction - constant extrapolation.
#         V, z_pos = interp1(self.speed, z, z_pos) # Interpolate to find wind speed - constant extrapolation.
#         RH, z_pos = interp1(self.relhum, z, z_pos) # Interpolate to find relative humidity - constant extrapolation.
#         Ta, z_pos = interp1(self.temperature, z, z_pos) # Interpolate to find temperature - constant extrapolation.
#         Pa, z_pos = interpexp(self.pressure, z, z_pos) # Interpolate to find pressure - exponential interpolation and extrapolation.

#         vals['psia'] = psia
#         vals['V'] = V
#         vals['RH'] = RH
#         vals['Ta'] = Ta
#         vals['Pa'] = Pa

#         #! Find temperature gradient
#         if (z_pos<0):
#             raise ValueError('requested z in MetFieldInterp is below lowest height in Met data.')
#         elif (z_pos < self.temperature.shape[0]):
#             dTadz = (self.temperature[z_pos,1]-self.temperature[z_pos-1,1])/(self.temperature[z_pos,0]-self.temperature[z_pos-1,0])
#         else:
#             dTadz = 0.0
#         vals['dTadz'] = dTadz
      
#         if (z_pos+1 < len(self.pressure[:,0])):
#             L = (np.log(self.pressure[z_pos+1,1]) - np.log(self.pressure[z_pos,1]))/(self.pressure[z_pos+1,0]-self.pressure[z_pos,0])
#             logA = np.log(self.pressure[z_pos,1]) - L*self.pressure[z_pos,0]
#         else:
#             L = (np.log(self.pressure[z_pos,1]) - np.log(self.pressure[z_pos-1,1]))/(self.pressure[z_pos,0]-self.pressure[z_pos-1,0])
#             logA = np.log(self.pressure[z_pos-1,1]) - L*self.pressure[z_pos-1,0]
#         A = np.exp(logA)
#         dPadz = L*A*np.exp(L*z)
#         vals['dPadz'] = dPadz
      
#         return vals, z_pos


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
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



@njit(cache=True)
def SpecHumidity(RH, esa, Pa, Ra, Rv):
    wa = RH*esa*Ra/(Rv*Pa - RH*esa*(Rv-Ra))
    return wa




class MetData:
    date: str
    time: str
    type: str
    stationID: Optional[str]
    source: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]

    def __init__(self, date: str, time: str, type: str,
                stationID: Optional[str]=None,
                source: Optional[str]=None,
                latitude: Optional[float]=None,
                longitude: Optional[float]=None
                ):
        
        if type is None:
            raise ValueError("MetData 'type' must be given")

        if date is None:
            raise ValueError("MetData 'date' must be given")
        if not re.match('^[0-9]{4}-[0-9]{2}-[0-9]{2}$', date):
            raise ValueError("MetData 'date' must be in the format 'YYYY-MM-DD'")
        
        if time is None:
            raise ValueError("MetData 'time' must be given")
        if not re.match('^[0-9]{2}:[0-9]{2}$', time):
            raise ValueError("MetData 'time' must be in the format 'HH:MM'")

        self.type = type.lower()

        match self.type:
            case 'radiosonde':
                if source is None:
                    raise ValueError("MetData 'source' must be given for type 'radiosonde'")
                self.source = source.lower()
                if self.source=='wyoming':
                    if stationID is None:
                        raise ValueError("MetData 'stationID' must be given for source 'wyoming'")
                    self.date = date
                    self.time = time
                    self.stationID = stationID
                    self.data = self.get_wyoming()
            case 'era5':
                self.source = 'cds'
                self.date = date
                self.time = time
                if latitude is None:
                    raise ValueError("MetData 'latitude' must be given for type 'era5'")
                try:
                    self.latitude = np.float64(latitude)
                except:
                    raise ValueError("MetData 'latitude' must be given in decimal degrees")
                if np.abs(self.latitude)>90:
                    raise ValueError("MetData 'latitude' must be in the range -90..90 degrees")
                
                if longitude is None:
                    raise ValueError("MetData 'longitude' must be given for type 'era5'")
                try:
                    self.longitude = np.float64(longitude)
                except:
                    raise ValueError("MetData 'longitude' must be given in decimal degrees")
                if self.longitude>360 or self.longitude<-180:
                    raise ValueError("MetData 'longitude' must be in the range 0..360 degrees, or -180...180 degrees")

                self.data = self.get_era5()
            case _:
                raise ValueError("In MetData initialization 'MetType' not recognized. \n "
                                 + " 'MetType' must be one of 'Radiosonde', 'ERA5'")
            
    def to_array(self):
        met_array = np.zeros((self.data.size, 6), dtype=np.float64)
        met_array[:,0] = self.data['height']
        met_array[:,1] = self.data['temperature']
        met_array[:,2] = self.data['pressure']
        met_array[:,3] = self.data['direction']
        met_array[:,4] = self.data['speed']
        met_array[:,5] = self.data['relhum']

        return met_array
            
    def get_wyoming(self):
        """get_wyoming: get radiosonde data from Wyoming Weather Web at specified date, time and station.

        Required:
            date [string]: date in ISO format YYYY-MM-DD
            time [string]: time in HH:MM format
            station [string]: station label

        Returns:
            metdata: structured array with radiosonde data

        """

        date = datetime.fromisoformat(self.date+' '+self.time)

        df = WyomingUpperAir.request_data(date, self.stationID)
        df.dropna(axis=0, inplace=True)

        temperature = mpunits.units.Quantity(df['temperature'].values, df.units['temperature'])
        pressure = mpunits.units.Quantity(df['pressure'].values, df.units['pressure'])
        height = mpunits.units.Quantity(df['height'].values, df.units['height'])
        dewpoint = mpunits.units.Quantity(df['dewpoint'].values, df.units['dewpoint'])
        direction = mpunits.units.Quantity(df['direction'].values, df.units['direction'])
        speed = mpunits.units.Quantity(df['speed'].values, df.units['speed'])

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
        metdata['temperature'] = temperature.data
        metdata['pressure'] = pressure.data
        metdata['height'] = height.data
        metdata['direction'] = direction.data
        metdata['speed'] = speed.data
        metdata['relhum'] = relhum.data

        return metdata
    
    def get_era5(self):
        """get_era5: get reanalysis data from ECMWF ERA5 at specified date, time and location (latitude, longitude).

        Requires:
            date [string]: date in ISO format YYYY-MM-DD
            time [string]: time in HH:MM format
            latitude [float]: latitude in decimal degrees (WGS84)
            longitude [float]: latitude in decimal degrees (WGS84)

        Returns:
            metdata: structured array with era5 data

        """

        date = datetime.fromisoformat(self.date+' '+self.time)

        era5data = era5('era5_data.nc', self.latitude, self.longitude)
        era5data.download(datetime=date)
        era5data.extract()

        met_types = np.dtype([
            ('temperature', np.float64),
            ('pressure', np.float64),
            ('height', np.float64),
            ('direction', np.float64),
            ('speed', np.float64),
            ('relhum', np.float64),
        ])

        metdata = np.empty(era5data.altitude.size, dtype=met_types)
        metdata['temperature'] = era5data.temperature
        metdata['pressure'] = era5data.pressure
        metdata['height'] = era5data.altitude
        metdata['direction'] = era5data.direction
        metdata['speed'] = era5data.speed
        metdata['relhum'] = era5data.relative_humidity

        return metdata


class era5:
    def __init__(self, file, lat, lon):
        self.file = file
        self.lat = lat
        self.lon = lon

        self.altitude = np.empty((0), dtype=np.float64)
        self.wind_U = np.empty((0), dtype=np.float64)
        self.wind_V = np.empty((0), dtype=np.float64)
        self.direction = np.empty((0), dtype=np.float64)
        self.speed = np.empty((0), dtype=np.float64)
        self.temperature = np.empty((0), dtype=np.float64)
        self.pressure = np.empty((0), dtype=np.float64)
        self.geopotential = np.empty((0), dtype=np.float64)
        self.relative_humidity = np.empty((0), dtype=np.float64)

    def extract(self):
        nc = Dataset(self.file, "r")
        P = mpunits.units.Quantity(nc.variables["level"][:].data, nc.variables["level"].units)

        lats = nc.variables["latitude"][:].data
        lons = nc.variables["longitude"][:].data

        if np.amax(lons) > 180.0:
            lons[lons > 180.0] = 180.0 - lons[lons > 180.0]

        lat_i0, _, lon_i0, _ = _near_lat_lon(self.lat, self.lon, lats, lons)

        geopotential = mpunits.units.Quantity(nc.variables["z"][0, :, lat_i0, lon_i0].data, nc.variables["z"].units)

        altitude = mpcalc.geopotential_to_height(geopotential)

        Uij = mpunits.units.Quantity(_interp_latlon(self.lat, self.lon, nc.variables["u"], lats, lons), nc.variables["u"].units)
        Vij = mpunits.units.Quantity(_interp_latlon(self.lat, self.lon, nc.variables["v"], lats, lons), nc.variables["v"].units)
        Tij = mpunits.units.Quantity(_interp_latlon(self.lat, self.lon, nc.variables["t"], lats, lons), nc.variables["t"].units)
        RelHum = mpunits.units.Quantity(_interp_latlon(self.lat, self.lon, nc.variables['r'], lats, lons), nc.variables["r"].units)

        Dirc = mpcalc.wind_direction(Uij, Vij)
        Speed = mpcalc.wind_speed(Uij,Vij)

        Tij = Tij.to(mpunits.units.K)
        P = P.to(mpunits.units.Pa)
        altitude = altitude.to(mpunits.units.m)
        Dirc = Dirc.to(mpunits.units.radian)
        Speed = Speed.to(mpunits.units.mps)
        RelHum = RelHum.to(mpunits.units.dimensionless)

        self.altitude = np.float64(np.asarray(np.flipud(altitude.data)))
        self.geopotential = np.flipud(np.float64(geopotential.data))
        self.wind_U = np.float64(np.asarray(np.flipud(Uij.data)))
        self.wind_V = np.float64(np.asarray(np.flipud(Vij.data)))
        self.direction = np.float64(np.asarray(np.flipud(Dirc.data)))
        self.speed = np.float64(np.asarray(np.flipud(Speed.data)))
        self.temperature = np.float64(np.asarray(np.flipud(Tij.data)))
        self.pressure = np.float64(np.asarray(np.flipud(P.data)))
        self.relative_humidity = np.float64(np.asarray(np.flipud(RelHum.data)))

    def download(self, *, datetime):
        if os.path.isfile(self.file):
            print(f"Met file {self.file} exists and will be overwritten".format())

        cds = cdsapi.Client()
        cds.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": [
                    "geopotential",
                    "temperature",
                    "u_component_of_wind",
                    "v_component_of_wind",
                    "relative humidity"
                ],
                "pressure_level": [
                    "1",
                    "2",
                    "3",
                    "5",
                    "7",
                    "10",
                    "20",
                    "30",
                    "50",
                    "70",
                    "100",
                    "125",
                    "150",
                    "175",
                    "200",
                    "225",
                    "250",
                    "300",
                    "350",
                    "400",
                    "450",
                    "500",
                    "550",
                    "600",
                    "650",
                    "700",
                    "750",
                    "775",
                    "800",
                    "825",
                    "850",
                    "875",
                    "900",
                    "925",
                    "950",
                    "975",
                    "1000",
                ],
                "year": str(datetime.year),
                "month": "{:02d}".format(datetime.month),
                "day": "{:02d}".format(datetime.day),
                "time": "{:02d}:00".format(datetime.hour),
                "area": [
                    ceil(self.lat + 0.25),
                    floor(self.lon - 0.25),
                    floor(self.lat - 0.25),
                    ceil(self.lon + 0.25),
                ],
            },
            self.file,
        )

def _near_lat_lon(target_lat, target_lon, lats, lons):

    lat_i_near = np.abs(lats - target_lat).argmin()
    lon_i_near = np.abs(lons - target_lon).argmin()

    if target_lat in lats:
        lat_i0 = lat_i_near
        lat_i1 = lat_i_near  # won't need this
    else:
        lat_i0 = lat_i_near if lat_i_near < target_lat else lat_i_near + 1
        lat_i1 = lat_i0 - 1
    if target_lon in lons:
        lon_i0 = lon_i_near
        lon_i1 = lon_i_near  # won't need this
    else:
        lon_i0 = lon_i_near if lon_i_near < target_lon else lon_i_near - 1
        lon_i1 = lon_i0 + 1

    return lat_i0, lat_i1, lon_i0, lon_i1


def _interp_latlon(target_lat, target_lon, data, lats, lons):

    lat_i0, lat_i1, lon_i0, lon_i1 = _near_lat_lon(target_lat, target_lon, lats, lons)

    d_lat = target_lat - lats[lat_i0]
    d_lon = target_lon - lons[lon_i0]

    data_00 = data[0, :, lat_i0, lon_i0]
    data_01 = data[0, :, lat_i1, lon_i0]
    data_10 = data[0, :, lat_i0, lon_i1]
    data_11 = data[0, :, lat_i1, lon_i1]

    data_i0 = data_00 + d_lon * (data_10 - data_00)
    data_i1 = data_01 + d_lon * (data_11 - data_01)
    data_ij = data_i0 + d_lat * (data_i1 - data_i0)

    return data_ij