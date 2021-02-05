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
   is needed for the damage generation, see ../run_climada/schedule_sim.py.
   Note that this script uses needs at least 7 threads to calculate the results, if other
   hardware restrictions are given please adjust this in line 104
"""
# please download the basin shapefile from
# https://www.bafg.de/GRDC/EN/02_srvcs/22_gslrs/223_WMO/wmo_regions_2020.html?nn=201570
GDF = gpd.read_file('../../../data/downloads/wmobb_basins.shp')
# set outpufile from basin_assignment.py here
BASIN_TRENDS = pd.read_csv('../../../data/reconstruction/basin_assignment.csv')
# set outpufile from discharge_median.sh here
DIS_PATH = '../../../data/reconstruction/trends_discharge.nc'

N_BASINS = GDF.count().max()

def write_netCDF(geo_trends, lat_step, lon_step):
    """
    This function stores the results in a netCDF file.
    Parameters
    ----------
    trends : np.array
        slope of trend in discharge in each grid-cell
    p_values : np.array
        p-value of trend in discharge in each grid-cell
    lat_step : int
        distance used to sample the lat resolution
    lon_step : TYPE
        distance used to sample the lon resolution
    """

    f = nc4.Dataset('../../../data/reconstruction/basin_trends_repro.nc','w', format='NETCDF4')

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
    """
    This function reads the general trend in the basin (1,-1) and writes it to the grid-cell
    Parameters
    ----------
    lat : TYPE
        DESCRIPTION.
    lon : TYPE
        DESCRIPTION.

    Returns
    -------
    geo_trend : int
        discharge trend in the river-basin containing this grid-cell(1,-1)
    bas_id : int
        basin id

    """
    
    point = Point(lon, lat)

    geo_trend = np.nan

    bas_id = 0
    for bas_idx in range(N_BASINS):
        basin_shp = GDF['geometry'].iloc[bas_idx]
        if point.within(basin_shp):
            bas_id = GDF['WMOBB'].iloc[bas_idx]
            geo_trend = BASIN_TRENDS.loc[BASIN_TRENDS['WMOBB'] == bas_id, 'DIS_REG_GEO'].sum()
            break

    return geo_trend, bas_id


def multip_function(x):
    """This function reads the general trend in the river-basin and assigns it to the gridcell.

    Parameters
    ----------
    x : tuple
        lat, lon coordinates

    Returns
    -------
    reg.slope
        slope of the trend
    reg.p
        v-value of the trend
    """
    lat, lon = x

    if (lat == -1000) or (lon == -1000):
        return np.nan, np.nan
    print(lon, lat)

    geo_trend, bas_id = get_trend(lat, lon)

    return geo_trend, bas_id


def main():
    """Function generates coordinates the multicore processing and assignes gridcells to
       different basin-groups with either positive or negative trends in annual max. discharge.
       It finally calls the function to write the output file.
    """
    #  adjust number of threads here
    pooli = mp.Pool(7)
    # to fasten the process larger steps can be taken...this produces an output of lower resolution
    lat_step = 1
    lon_step = 1

    lat_len = 720
    lon_len = 1440
    trend_lon = int(lon_len/lon_step)
    trend_lat = int(lat_len/lat_step)

    dis = xr.open_dataset(DIS_PATH)
    lat = dis.lat.data
    lon = dis.lon.data
    diai = dis.Discharge[0, ::lat_step, ::lon_step].data
    dis.close()
    mask = (np.isnan(diai)).flatten()

    x_grid, y_grid = np.meshgrid(lon[::lon_step], lat[::lat_step])
    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()
    x_grid[mask] = -1000
    y_grid[mask] = -1000
    a = pooli.map(multip_function, zip(y_grid, x_grid))
    a, b = zip(*a)
    a = np.array(a).reshape(trend_lat, trend_lon)

    geo_trends = a

    write_netCDF(geo_trends, lat_step, lon_step)

    return



if __name__ == "__main__":
    main()