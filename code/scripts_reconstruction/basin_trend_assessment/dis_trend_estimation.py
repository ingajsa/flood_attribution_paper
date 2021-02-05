#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 15:20:09 2020

@author: insauer
"""

import numpy as np
import xarray as xr
import multiprocessing as mp
import netCDF4 as nc4
import pymannkendall as mk

"""This script produces the trend from 1971-2010 in annual maximum discharge the
   file for the median discharge from the entire ISIMIP ensemble needs to be
   calculated first in discharge_median.sh
   Afterwards the output of this script can be used in basin_assignment.py
   Note that this script uses needs at least 7 threads to calculate the results, if other
   hardware restrictions are given please adjust this in line 125.
"""

"""The following file needs to be calculated with the bash script
   in /code/scripts_reconstruction/basin_trend_assessment/discharge_median.sh
   all files containg discharge as output provided by ISIMIP must be used to get
   the ensemble median of the annual maximum discharge.
"""

PATH = '../../../data/reconstruction/Median_Merged_1971_2010.nc'


def write_netCDF(trends, p_values, lat_step, lon_step):
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
    # name of the output file
    f = nc4.Dataset('../../../data/reconstruction/trends_discharge.nc', 'w', format='NETCDF4')

    dis = xr.open_dataset(PATH)
    lat = dis.lat.data[::lat_step]
    lon = dis.lon.data[::lon_step]
    f.createDimension('lon', len(lon))
    f.createDimension('lat', len(lat))
    f.createDimension('time', None)
    longitude = f.createVariable('lon', 'f4', ('lon'))
    latitude = f.createVariable('lat', 'f4', ('lat'))
    disch = f.createVariable('Discharge', 'f4', ('time', 'lat', 'lon'))
    #  pval = f.createVariable('disch', 'f4', ('time', 'lat', 'lon'))
    longitude[:] = lon  # The "[:]" at the end of the variable instance is necessary
    latitude[:] = lat
    disch[0, :, :] = trends
    #  pval[0, :, :] = p_values
    f.close()


def get_dis_gridcell(lat, lon):
    """
    Function provides discharge time-series for one grid-cell

    Parameters
    ----------
    lat : float
        latitude coordinate
    lon : float
        longitude coordinate.

    Returns
    -------
   np. array
        time-series of discharge for one grid-cell

    """
    dis = xr.open_dataset(PATH)
    dis.close()
    return dis.discharge[:, lat, lon].data


def multip_function(x):
    """This function estimates the trend in the discharge time series of one grid cell.

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
    # exclude coordinates over sea
    if (lat == -1000) or (lon == -1000):
        return np.nan, np.nan
    print(lon, lat)
    data = get_dis_gridcell(lat, lon)

    if np.isnan(data).all():
        return np.nan, np.nan
    else:
        reg = mk.original_test(data, alpha=0.1)

    return reg.slope, reg.p


def main():
    """Function generates coordinates the multicore processing and assignes gridcells to
       different threads to estimate trends. It finally calls the function to write the
       output file.
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

    trends = np.zeros((trend_lat, trend_lon))
    p_values = np.zeros((trend_lat, trend_lon))
    #  insert the file containing the ensemble median for annual maximum discharge
    #  this file needs to be generated with discharge_median.sh
    dis = xr.open_dataset(PATH)

    diai = dis.discharge[20, ::lat_step, ::lon_step].data
    dis.close()
    mask = (np.isnan(diai)).flatten()

    x_grid, y_grid = np.meshgrid(np.arange(0, lon_len, lon_step), np.arange(0, lat_len, lat_step))
    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()
    x_grid[mask] = -1000
    y_grid[mask] = -1000
    a = pooli.map(multip_function, zip(y_grid, x_grid))
    a, b = zip(*a)
    a = np.array(a).reshape(trend_lat, trend_lon)
    b = np.array(b).reshape(trend_lat, trend_lon)

    trends = a
    p_values = b

    write_netCDF(trends, p_values, lat_step, lon_step)

    return


if __name__ == "__main__":
    main()