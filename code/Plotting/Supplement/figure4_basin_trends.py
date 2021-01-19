#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 12:49:30 2020

@author: insauer
"""

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

dis_bin = '/home/insauer/projects/RiverDischarge/Data/basin_trends_geo.nc'

natcat_id = '/home/insauer/projects/RiverDischarge/Data/NatID_0_25.nc'



reg_frame = pd.read_csv('/home/insauer/Climada/climada_python/data/system/NatRegIDs.csv')

reg_frame.loc[reg_frame['Reg_name']=='NAM', 'Reg_plot']= 509
reg_frame.loc[reg_frame['Reg_name']=='CHN', 'Reg_plot']= 507
reg_frame.loc[(reg_frame['Reg_name']=='EUR') |
              (reg_frame['Reg_name']=='EUA')
              , 'Reg_plot']= 505
reg_frame.loc[(reg_frame['Reg_name']=='CAR') |
              (reg_frame['Reg_name']=='LAN') |
              (reg_frame['Reg_name']=='LAS')
              , 'Reg_plot']= 503
reg_frame.loc[(reg_frame['Reg_name']=='CAS') |
              (reg_frame['ISO']=='RUS')
              , 'Reg_plot']= 501


reg_frame.loc[(reg_frame['Reg_name']=='SSA') |
              (reg_frame['Reg_name']=='SAF') 
              , 'Reg_plot']= 507

reg_frame.loc[(reg_frame['Reg_name']=='SWA') |
              (reg_frame['Reg_name']=='SEA') 
              , 'Reg_plot']= 505

reg_frame.loc[(reg_frame['Reg_name']=='ARA') |
              (reg_frame['Reg_name']=='NAF') 
              , 'Reg_plot']= 503

reg_frame.loc[reg_frame['Reg_name']=='AUS', 'Reg_plot']= 501

file = xr.open_dataset(dis_bin)

lat  = file.lat.data
lon  = file.lon.data

#data = file.variables['p-value'][0,:,:]
#p-value =1
data_dis =file.basin_trend.data[0,:,:]
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

m.pcolormesh(x,y,data_dis,shading='flat',cmap=plt.cm.BrBG, vmin = -1.5, vmax = 1.5)
#m.pcolormesh(x,y,data_dis,shading='flat',cmap=plt.cm.tab10)
#col = m.colorbar(location='right', size = 0.08)
#col.set_label('Trend in discharge $m^{3}$/s', fontsize = 5)
#col.ax.tick_params(axis="y", length = 4, labelsize =4)
# Add a coastline and axis values.

m.drawcoastlines(linewidth=0.15)
#m.drawcountries()
m.drawmapboundary(fill_color= 'white')
m.drawlsmask(ocean_color='aqua',lakes=True)
m.readshapefile('/home/insauer/projects/RiverDischarge/Data/river_basins/wmobb_basins','geometry', linewidth = 0.15)

m.drawparallels(np.arange(-90.,90.,30.),labels=[1,0,0,0],fontsize=3, linewidth=0.2)
m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1], fontsize=3, linewidth=0.2)

cmap = plt.cm.get_cmap('BrBG')
rgba_dark = cmap(0.82)
rgba_light = cmap(0.22)

pos_box = mpatches.Rectangle((0, 0), 1,1, color=rgba_dark, label ='Basins with positive discharge trend')
neg_box = mpatches.Rectangle((0, 0), 1, 1, color=rgba_light, label ='Basins with negative discharge trend')
leg1 = ax.legend(handles = [pos_box, neg_box], frameon=1, fontsize = 2.2,bbox_to_anchor=(0.67, 0.18))  
frame = leg1.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('white')
frame.set_linewidth(0)

plt.savefig('/home/insauer/projects/NC_Submission/Data/Figures/Supplement/SI4_basin_trends.png',  bbox_inches = 'tight', resolution=600)

#plt.savefig('/home/insauer/projects/Attribution/Floods/Plots/FinalPaper/PreliminaryPlots/Region_map.svg', bbox_inches = 'tight',format = 'svg')
#plt.savefig('/home/insauer/projects/NC_Submission/Data/Figures/Supplement/SI4_basin_trends.png',  bbox_inches = 'tight', resolution=600)