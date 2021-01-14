#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 15:20:09 2020

@author: insauer
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import os
import multiprocessing as mp
import statsmodels.api as sm
import netCDF4 as nc4
#from numba import autojit, prange, njit
import pymannkendall as mk

"""This script produces the trend from 1971-2010 in annual maximum discharge the 
   file for the median discharge from the entire ISIMIP ensemble needs to be
   calculated first in /code/scripts_reconstruction/basin_trend_assessment/discharge_median.sh
   Afterwards the output can be used in 
   /code/scripts_reconstruction/basin_trend_assessment/basin_assignment.py
   Note that this script uses needs at least 7 threads to calculate the results, if other
   hardware restrictions are given please adjust this in line 104
"""

def gather_grid_cell(lat,lon):
    
    disch = get_dis_gridcell(path, lat, lon)
    
    return disch

def write_netCDF(trends, p_values, lat_step, lon_step):

    # name of the output file
    f = nc4.Dataset('~/data/repro/trends_median_discharge.nc','w', format='NETCDF4')
    
    """The following file needs to be calculated with the bash script 
       in /code/scripts_reconstruction/basin_trend_assessment/discharge_median.sh
       all files containg discharge as output provided by ISIMIP must be used to get
       the ensemble median of the annual maximum discharge.
    """
    path = '~/Data/Median_Merged_1971_2010.nc'
    dis = xr.open_dataset(path)
    lat = dis.lat.data[::lat_step]
    lon = dis.lon.data[::lon_step]
    #lat = lat[lat_inds]
    #lon = lon[lon_inds]
    
    f.createDimension('lon', len(lon))
    f.createDimension('lat', len(lat))
    f.createDimension('time', None) 
    time = f.createVariable('time', 'f8', ('time',))
    longitude = f.createVariable('lon', 'f4', ('lon'))
    latitude = f.createVariable('lat', 'f4', ('lat')) 
    disch = f.createVariable('Discharge', 'f4', ('time','lat', 'lon'))
    pval = f.createVariable('disch', 'f4', ('time','lat', 'lon'))
    longitude[:] = lon #The "[:]" at the end of the variable instance is necessary
    latitude[:] = lat
    disch[0,:,:] = trends
    #pval[0,:,:] = p_values
    f.close()


def get_dis_gridcell(path, lat, lon):
    
    dis = xr.open_dataset(path)
    #print(dis.discharge[:,lat,lon].data)
    
    return dis.discharge[:,lat,lon].data
    
    
def model_median(x):
    
    dis_median = x['Discharge'].median()
    
    return pd.Series(dis_median, index =['Discharge'])

def multip_function(x):
    
    
    lat,lon = x
    years = np.arange(1971, 2011)
    
    
    if (lat == -1000) or (lon == -1000):
        return np.nan, np.nan
    print(lon, lat)
    data = gather_grid_cell(lat, lon)
    #data = data.groupby(['Year'])[['Discharge']].apply(model_median).reset_index()
    if np.isnan(data).all():     
        return np.nan, np.nan

    else:
        
        reg= mk.original_test(data, alpha = 0.1)
        
    return reg.slope, reg.p
    

def main():
    pooli = mp.Pool(7)
    lat_step = 1
    lon_step = 1
    
    lat_len = 720
    lon_len = 1440
    
    trend_lon = int(lon_len/lon_step)
    trend_lat = int(lat_len/lat_step)
    
    trends = np.zeros((trend_lat, trend_lon))
    p_values =np.zeros((trend_lat, trend_lon))
    
    dis = xr.open_dataset('~/Data/Median_Merged_1971_2010.nc')
    lat = dis.lat.data
    lon = dis.lon.data
    diai = dis.discharge[20,::lat_step, ::lon_step].data
    dis.close()
    mask= (np.isnan(diai)).flatten()


    x_grid,y_grid = np.meshgrid(np.arange(0,lon_len,lon_step), np.arange(0,lat_len,lat_step))
    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()
    x_grid[mask] = -1000
    y_grid[mask] = -1000
    a = pooli.map(multip_function, zip(y_grid, x_grid))
    a,b = zip(*a)
    a = np.array(a).reshape(trend_lat, trend_lon)
    b = np.array(b).reshape(trend_lat, trend_lon)

    trends =a
    p_values = b


        
    write_netCDF(trends, p_values, lat_step, lon_step)

    return


if __name__ == "__main__":
    main()