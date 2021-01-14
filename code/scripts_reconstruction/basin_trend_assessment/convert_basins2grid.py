#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 14 16:11:11 2020

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
import geopandas as gpd
from shapely.geometry import Point

"""This script provides the the overall discharge trend in each basin on the
   grid-level. The outputfile ~/data/hazard_settings/basin_trends.nc
   is needed for the damage generation, see /code/scripts_reconstruction/run_climada/schedule_sim.py.
"""

Note that this script uses needs at least 7 threads to calculate the results, if other
   hardware restrictions are given please adjust this in line 104

GDF = gpd.read_file('/home/insauer/projects/RiverDischarge/Data/river_basins/wmobb_basins.shp')
# set outpufile from /code/scripts_reconstruction/basin_trend_assessment/basin_assignment.py here
BASIN_TRENDS = pd.read_csv('~/data/repro/basin_assignment.csv')
# set outpufile from /code/scripts_reconstruction/basin_trend_assessment/discharge_median.sh here
DIS_PATH = '~/data/repro/trends_median_discharge.nc'

N_BASINS = GDF.count().max()

def write_netCDF(geo_trends, lat_step, lon_step):


    f = nc4.Dataset('~/data/hazard_settings/basin_trends_repro.nc','w', format='NETCDF4')

    dis = xr.open_dataset(DIS_PATH)
    lat = dis.lat.data[::lat_step]
    lon = dis.lon.data[::lon_step]


    f.createDimension('lon', len(lon))
    f.createDimension('lat', len(lat))
    f.createDimension('time', None) 
    time = f.createVariable('time', 'f8', ('time',))
    longitude = f.createVariable('lon', 'f4', ('lon'))
    latitude = f.createVariable('lat', 'f4', ('lat')) 
    # keep this name "basin_trend" fixed!
    disch = f.createVariable('basin_trend', 'f4', ('time','lat', 'lon'))
    #pval = f.createVariable('pvalues', 'f4', ('time','lat', 'lon'))
    longitude[:] = lon #The "[:]" at the end of the variable instance is necessary
    latitude[:] = lat
    disch[0,:,:] = geo_trends
    #pval[0,:,:] = p_values
    f.close()


def get_trend(lat, lon):
    point = Point(lon, lat)

    geo_trend = np.nan

    bas_id = 0
    for bas_idx in range(N_BASINS):
        basin_shp = GDF['geometry'].iloc[bas_idx]
        if point.within(basin_shp):
            bas_id = GDF['WMOBB'].iloc[bas_idx]
            geo_trend = BASIN_TRENDS.loc[BASIN_TRENDS['WMOBB'] == bas_id, 'DIS_REG_GEO'].sum()
            break
    return geo_trend


def multip_function(x):

    lat, lon = x

    if (lat == -1000) or (lon == -1000):
        return np.nan, np.nan
    print(lon, lat)

     geo_trend = get_trend(lat, lon)

    return geo_trend

def main():
    pooli = mp.Pool(7)
    lat_step = 1
    lon_step = 1
    
    
    lat_len = 720
    lon_len = 1440
    
    trend_lon = 1440
    trend_lat = 720

    
    dis = xr.open_dataset(DIS_PATH)
    lat = dis.lat.data
    lon = dis.lon.data
    diai = dis.discharge[20, ::lat_step, ::lon_step].data
    dis.close()
    mask = (np.isnan(diai)).flatten()


    x_grid, y_grid = np.meshgrid(lon, lat)
    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()
    x_grid[mask] = -1000
    y_grid[mask] = -1000
    a = pooli.map(multip_function, zip(y_grid, x_grid))
    a = zip(*a)
    a = np.array(a).reshape(trend_lat, trend_lon)

    geo_trends = a


        
    write_netCDF(geo_trends, lat_step, lon_step)
    
    

    return



if __name__ == "__main__":
    main()