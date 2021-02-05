#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 16:41:50 2020

@author: insauer
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy as sp
from climada.hazard.base import Hazard
from shapely.geometry.multipolygon import MultiPolygon

"""This script assignes each basin an overarching trend direction (positive or negative)
   Afterwards the output can be converted to the grid level in
   /code/scripts_reconstruction/basin_trend_assessment/convert_basins2grid.py
"""

"""The following file needs to be calculated with the script
   dis_trend_estimation.py.
"""
trend_file = '../../../data/reconstruction/trends_discharge.nc'

# please download the basin shapefile from
# https://www.bafg.de/GRDC/EN/02_srvcs/22_gslrs/223_WMO/wmo_regions_2020.html?nn=201570
gdf = gpd.read_file('../../../data/downloads/wmobb_basins.shp')
n_basins = gdf.count().max()

df_bas_assig = pd.DataFrame(columns=['Index', 'WMOBB', 'WMOBB_NAME', 'WMOBB_BASI', 'WMOBB_SUBB',
                                     'REGNUM', 'REGNAME',  'GRID_POS', 'GRID_NEG', 'DIS_REG_GEO'])

for bas_idx in range(n_basins):

    df_sing_bas = pd.DataFrame(columns=['Index', 'WMOBB', 'WMOBB_NAME', 'WMOBB_BASI', 'WMOBB_SUBB',
                                        'REGNUM', 'REGNAME', 'GRID_POS', 'GRID_NEG', 'DIS_REG_GEO'])

    basin_shp = gdf['geometry'].iloc[bas_idx]

    haz_pos = Hazard('RF')
    haz_neg = Hazard('RF')

    try:

        haz_pos.set_raster(files_intensity=[trend_file], files_fraction=[trend_file],
                           band=[1], geometry=basin_shp)
        haz_neg.set_raster(files_intensity=[trend_file], files_fraction=[trend_file],
                           band=[1], geometry=basin_shp)

    except TypeError:

        basin_shp = MultiPolygon([basin_shp])
        haz_pos.set_raster(files_intensity=[trend_file], files_fraction=[trend_file],
                           band=[1], geometry=basin_shp)
        haz_neg.set_raster(files_intensity=[trend_file], files_fraction=[trend_file],
                           band=[1], geometry=basin_shp)

    dis_map_pos = np.greater(haz_pos.intensity.todense(), 0)
    new_trends = dis_map_pos.astype(int)

    haz_pos.fraction = sp.sparse.csr_matrix(new_trends)
    haz_pos.intensity = sp.sparse.csr_matrix(new_trends)

    dis_map_neg = np.less(haz_neg.fraction.todense(), 0)

    new_trends_neg = dis_map_neg.astype(int)

    haz_neg.fraction = sp.sparse.csr_matrix(new_trends_neg)
    haz_neg.intensity = sp.sparse.csr_matrix(new_trends_neg)

    if new_trends.sum() > new_trends_neg.sum():
        dis_reg_geo = 1
    elif new_trends.sum() < new_trends_neg.sum():
        dis_reg_geo = -1
    else:
        dis_reg_geo = 0

    df_sing_bas.loc[0, 'Index'] = bas_idx
    df_sing_bas.loc[0, 'WMOBB'] = gdf['WMOBB'].iloc[bas_idx]
    df_sing_bas.loc[0, 'WMOBB_NAME'] = gdf['WMOBB_NAME'].iloc[bas_idx]
    df_sing_bas.loc[0, 'WMOBB_BASI'] = gdf['WMOBB_BASI'].iloc[bas_idx]
    df_sing_bas.loc[0, 'WMOBB_SUBB'] = gdf['WMOBB_SUBB'].iloc[bas_idx]
    df_sing_bas.loc[0, 'REGNUM'] = gdf['REGNUM'].iloc[bas_idx]
    df_sing_bas.loc[0, 'REGNAME'] = gdf['REGNAME'].iloc[bas_idx]
    df_sing_bas.loc[0, 'GRID_POS'] = new_trends.sum()
    df_sing_bas.loc[0, 'GRID_NEG'] = new_trends_neg.sum()
    df_sing_bas.loc[0, 'DIS_REG_GEO'] = dis_reg_geo

    df_bas_assig = df_bas_assig.append(df_sing_bas, ignore_index=True)

    df_bas_assig.to_csv('../../../data/reconstruction/basin_assignment.csv')
df_bas_assig.to_csv('../../../data/reconstruction/basin_assignment.csv')
