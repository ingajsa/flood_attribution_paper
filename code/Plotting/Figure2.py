#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 22:09:13 2020

@author: insauer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""Script to reproduce main Figure 2!"""

fig3 = plt.figure(constrained_layout=True, figsize=(7., 9.9))
gs = fig3.add_gridspec(40, 15)
plt.subplots_adjust(wspace=0., hspace=0)


DATA_TSFull= pd.read_csv('../../data/reconstruction/attribution_TimeSeries_regions.csv')
DATA_TS= pd.read_csv('../../data/reconstruction/attribution_TimeSeries_subregions.csv')

DATA_FIT_Full= pd.read_csv('../../../data/reconstruction/vulnerability_adjustment_MetaData_regions.csv')
DATA_FIT= pd.read_csv('../../data/reconstruction/vulnerability_adjustment_MetaData_subregions.csv')

region_names={'GLB': 'Global (GLB)',
              'NAM':'North America (NAM)',
              'EAS':'Eastern Asia (EAS)',
              'OCE':'Oceania (OCE)',
              'LAM':'Latin America (LAM)',
              'EUR':'Europe (EUR)',
              'SSA':'South & Sub-Sahara Africa (SSA)',
              'SEA':'South & South-East Asia (SEA)',
              'CAS':'Central Asia & Russia (CAS)',
              'NAF':'North Africa & Middle East (NAF)'
              }

region_abs={'GLB': 'GLB',
            'NAM':'NAM', 
            'EAS':'EAS',
            'LAM':'LAM', 
            'EUR':'EUR',
            'OCE':'OCE',
            'CAS':'CAS',
            'SSA':'SSA',
            'SEA':'SEA', 
            'NAF': 'NAF'}

regions = list(region_names)
r =0

for i in range(10):
    for j in range(4):
        
        DATA_regionFull = DATA_TSFull[(DATA_TSFull['Region']==regions[r]) & 
                      (DATA_TSFull ['Year']<2011) & (DATA_TSFull ['Year']>1979)]
        
        DATA_region = DATA_TS[(DATA_TS['Region']==regions[r]) & 
                      (DATA_TS['Year']<2011) & (DATA_TS['Year']>1979)]
        
        if j<3:
        
            f3_ax1 = fig3.add_subplot(gs[4*i:4*i+4,j*5:(j*5)+5])
            
            if j ==0:

                f3_ax1.plot(DATA_regionFull['Year'], np.log10(DATA_regionFull['natcat_flood_damages_2005_CPI']), label='D$_{Obs}$', color='black', linewidth = 1.) 
                f3_ax1.scatter(DATA_regionFull['Year'], np.log10(DATA_regionFull['natcat_flood_damages_2005_CPI']), color='black', marker = '_', s = 3) 
                f3_ax1.plot(DATA_regionFull['Year'], np.log10(DATA_regionFull['D_Full']), label='D$_{Full}$', color='#8856a7', linewidth = 1.)
                
                f3_ax1.plot(DATA_regionFull['Year'], np.log10(DATA_regionFull['D_CliExp']), label='D$_{CliExp}$', color='#ff7f00', linewidth = 1.)
            
                f3_ax1.plot(DATA_regionFull['Year'], np.log10(DATA_regionFull['D_1980']), label='D$_{1980}$', color='#4575b4', linewidth = 1.)
                
                f3_ax1.set_title(' '+ region_names[regions[r]], position = (0.5,0.75), fontsize = 6.5)

                if i ==0 and j ==0:
                    
                    handles, labels = f3_ax1.get_legend_handles_labels()
                    leg =f3_ax1.legend(handles[:2], labels[:2], loc ='lower left', labelspacing = 0.1, frameon=True, fontsize = 6, handlelength = 1.1) 
                    f3_ax1.legend(handles[2:], labels[2:], loc ='lower right', labelspacing = 0.1, frameon=True, fontsize = 6,  handlelength = 1.1)
                    f3_ax1.add_artist(leg)
 

                r_lin = DATA_FIT_Full.loc[DATA_FIT_Full['Region']==regions[r], 'R2_D_Full_D_Obs'].sum()

                #r2 = DATA_FIT_Full.loc[DATA_FIT_Full['Region']==regions[r], 'New_explained_variance'].sum()
               
            else:
                
                if j ==1:
                    dis = 'pos'
                    f3_ax1.set_title('{}'.format(region_abs[regions[r]])+'$_{+}$', position = (0.5,0.75), fontsize = 6.5)
                else:
                    dis = 'neg'
                    f3_ax1.set_title('{}'.format(region_abs[regions[r]])+'$_{-}$', position = (0.5,0.75), fontsize = 6.5)
                
                
                #f3_ax1.plot(DATA_region['Year'], np.log10(DATA_region['Impact_Pred_1thrd_{}'.format(dis)]), color='#8856a7', alpha = 0.5, linewidth = 1.)
                #f3_ax1.plot(DATA_region['Year'], np.log10(DATA_region['Impact_Pred_2thrd_{}'.format(dis)]), color='#8856a7', alpha = 0.5, linewidth = 1.)
                
                f3_ax1.plot(DATA_region['Year'], np.log10(DATA_region['natcat_damages_2005_CPI_{}'.format(dis)]), label='Observed Flood Losses (NatCat)', color='black', linewidth = 1.) 
                f3_ax1.scatter(DATA_region['Year'], np.log10(DATA_region['natcat_damages_2005_CPI_{}'.format(dis)]), label='Observed Flood Losses (NatCat)', color='black', marker = '.', s = 3) 
            
                
                if not (i==1 and dis=='pos'):  
                    
                 
                    f3_ax1.plot(DATA_region['Year'], np.log10(DATA_region['D_CliExp_{}'.format(dis)]), label='$Loss_{HazExp}$', color='#ff7f00', linewidth = 1.)

                
                    f3_ax1.plot(DATA_region['Year'], np.log10(DATA_region['D_1980_{}'.format(dis)]), label='$Loss_{Haz}$', color='#4575b4', linewidth = 1.)
                    
                    #f3_ax1.plot(DATA_region['Year'], np.log10(DATA_region['NormHaz_Imp2010_2y{}_offset'.format(dis)]), label='$Loss2010_{Haz}$', color='mediumseagreen', linewidth = 1.,linestyle ='--', alpha = 0.5)
            
                    f3_ax1.plot(DATA_region['Year'], np.log10(DATA_region['D_Full_{}'.format(dis)]), label='$Loss_{Full}$', color='#8856a7', linewidth = 1.)
                    

                r_lin = DATA_FIT.loc[DATA_FIT['Region']==regions[r]+'_'+dis, 'R2_D_Full_D_Obs'].sum()


                
            #text_LOG = 'R²='+str(round(r_log*100,1))+ '% (LOG)'
            text_lin = '='+str(round(r_lin,1))+ '%'
            
            
            if r_lin> 0.2*100:
                
                f3_ax1.set_facecolor('gainsboro')
            
            f3_ax1.set_yticks([6, 8, 10])
            f3_ax1.set_yticklabels(['','',''])
            if j == 0 :
                f3_ax1.set_yticklabels(['6','8','10'], fontsize = 5.5)
            if i in [2]:
                f3_ax1.set_ylim((5, 11.5))
            
            elif i in [1]:
                f3_ax1.set_ylim((4.5, 11.5))
                
            elif i in [4]:
                f3_ax1.set_ylim((5, 11.5))
            
            elif i in [3]:
                f3_ax1.set_ylim((3., 10))
                f3_ax1.set_yticks([4, 6, 8])
                if j == 0:
                    f3_ax1.set_yticklabels(['4','6','8'])
                
            elif i in [6]:
                f3_ax1.set_ylim((3, 10))
                f3_ax1.set_yticks([4, 6, 8])
                if j == 0:
                    f3_ax1.set_yticklabels(['4','6','8'])
                
            elif i in [5,7]:
                f3_ax1.set_ylim((5.5, 11.5))
            
            elif i in [8]:
                f3_ax1.set_ylim((3.5, 10.5))
                f3_ax1.set_yticks([4, 6, 8])
                if j == 0:
                    f3_ax1.set_yticklabels(['4','6','8'])
            
            elif i in [0]:
                f3_ax1.set_ylim((7, 12))
                f3_ax1.set_yticks([8, 10])
                
                if j == 0:
                    f3_ax1.set_yticklabels(['8','10'])
            else:
                f3_ax1.set_ylim((3.65, 11))
            
            f3_ax1.set_xlim((1978 ,2013))
            if not (i==1 and j==1):
                f3_ax1.annotate( xy=(1991.8, f3_ax1.get_ylim()[0]+0.08*(f3_ax1.get_ylim()[1]-f3_ax1.get_ylim()[0]) ) ,s='R²', fontsize=6, fontstyle='italic')
                f3_ax1.annotate( xy=(1993.5, f3_ax1.get_ylim()[0]+0.08*(f3_ax1.get_ylim()[1]-f3_ax1.get_ylim()[0]) ) ,s=text_lin, fontsize=6)

            f3_ax1.set_xticks([1980,1990,2000,2010])
            f3_ax1.set_xticklabels(['','','',''])
            
            
            if i == 9:
                f3_ax1.set_xticklabels(['1980','1990','2000','2010'], fontsize =6)
            
            if i ==4 and j ==0:
                f3_ax1.set_ylabel('log$_{10}$(Damages in 2005 USD)', fontsize=6.5, labelpad=-0.)
            

            
            if i == 9 and j ==1:
                f3_ax1.set_xticklabels(['1980','1990','2000', '2010'],fontsize=6)
                f3_ax1.set_xlabel('Year', fontsize=7, labelpad=-2)
            
            f3_ax1.tick_params(axis="x", direction = 'in',length = 3)
            
            handles, labels = f3_ax1.get_legend_handles_labels() 
        
    else:
        

        
        r_linPos = DATA_FIT.loc[DATA_FIT['Region']==regions[r]+'_pos', 'R2_D_Full_D_Obs'].sum()
        r_linNeg = DATA_FIT.loc[DATA_FIT['Region']==regions[r]+'_neg', 'R2_D_Full_D_Obs'].sum()
        r_lin = DATA_FIT_Full.loc[DATA_FIT_Full['Region']==regions[r], 'R2_D_Full_D_Obs'].sum()

        
        
    r+=1
plt.savefig('../../data/figures/Figure2.pdf',bbox_inches = 'tight')
