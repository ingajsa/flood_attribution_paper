#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 22:09:32 2019

@author: insauer
"""
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import pandas as pd
import netCDF4 as nc4
import matplotlib.patches as mpatches

def smooth():
    return
    

sig = '/home/insauer/projects/RiverDischarge/Data/TrendsMedianDischarge_MK_pval.nc'
disch= '/home/insauer/projects/RiverDischarge/Data/TrendsMedianDischarge_MK.nc'

dis_bin = '/home/insauer/projects/RiverDischarge/Data/basin_trends_pop.nc'

natcat_id = '/home/insauer/projects/RiverDischarge/Data/NatID_0_25.nc'




file = xr.open_dataset(disch)
file_p = xr.open_dataset(sig)

lat  = file.lat.data
lon  = file.lon.data

#data = file.variables['p-value'][0,:,:]
#p-value =1
data_dis =file.Discharge.data[0,:,:]

data_p =file_p.pvalues.data[0,:,:]

data_dis[np.where(data_dis>0)]= 1
data_dis[np.where(data_dis<0)]= -1

data_p[np.where(data_p>0.1)]= 0.5
data_p[np.where(data_p<0.1)]= 0.8

data_dis = data_dis*data_p

#file.p-value[0,:,:].data
file.close()


fig = plt.figure(dpi=600)
fig.subplots_adjust(left=0.5, hspace = -0.2, wspace = 0.05)



ax = fig.add_subplot(111)
ax.set_anchor('W')

m=Basemap(projection='mill',lat_ts=10,llcrnrlon=lon.min(), \
  urcrnrlon=lon.max(),llcrnrlat=-60.,urcrnrlat=80., \
  resolution='c')

x, y = m(*np.meshgrid(lon,lat))

m.pcolormesh(x,y,data_dis,shading='flat',cmap=plt.cm.BrBG, vmin = -1, vmax = 1)
#m.pcolormesh(x,y,data_dis,shading='flat',cmap=plt.cm.tab10)

# Add a coastline and axis values.

m.drawcoastlines(linewidth=0.2)
#m.drawcountries()
m.drawmapboundary(fill_color= 'white')
m.drawlsmask(ocean_color='aqua',lakes=True)

m.drawparallels(np.arange(-90.,90.,30.),labels=[1,0,0,0],fontsize=3, linewidth=0.2)
m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1], fontsize=3, linewidth=0.2)





cmap = plt.cm.get_cmap('BrBG')
rgba_dark_g = cmap(0.9)
rgba_light_g = cmap(0.75)

rgba_dark_b = cmap(0.1)
rgba_light_b = cmap(0.25)



pos_box_s = mpatches.Rectangle((0, 0), 1,1, color=rgba_dark_g, label ='significant positive discharge trend')
neg_box_s = mpatches.Rectangle((0, 0), 1, 1, color=rgba_dark_b, label ='significant negative discharge trend')
pos_box = mpatches.Rectangle((0, 0), 1,1, color=rgba_light_g, label ='positive discharge trend')
neg_box = mpatches.Rectangle((0, 0), 1, 1, color=rgba_light_b, label ='negative discharge trend')
leg1 = plt.legend(handles = [pos_box, neg_box,pos_box_s, neg_box_s], frameon=1, fontsize = 2.2,bbox_to_anchor=(0.67, 0.2))
frame = leg1.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('white')
frame.set_linewidth(0)

plt.savefig('/home/insauer/projects/NC_Submission/Data/Figures/Supplement/SI3_pvalues.png',  bbox_inches = 'tight', resolution=600)