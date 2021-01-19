#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 22:09:13 2020

@author: insauer
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cartopy
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import matplotlib.patches as mpatches
import matplotlib.font_manager as font_manager


fig3 = plt.figure(constrained_layout=True, figsize=(8.3, 11.7))
gs = fig3.add_gridspec(40, 17)
plt.subplots_adjust(wspace=0., hspace=0)

DATA_TSFull= pd.read_csv('/home/insauer/projects/NC_Submission/Data/postprocessing/AttributionTimeSeriesRegions.csv')
DATA_TS= pd.read_csv('/home/insauer/projects/NC_Submission/Data/postprocessing/AttributionTimeSeriesSubregions.csv')

DATA_FIT_Full= pd.read_csv('/home/insauer/projects/NC_Submission/Data/postprocessing/VulnerabilityAdjustmentMetaDataRegions.csv')
DATA_FIT= pd.read_csv('/home/insauer/projects/NC_Submission/Data/postprocessing/VulnerabilityAdjustmentMetaDataSubregions.csv')


DATA_ATTR_Full = pd.read_csv('/home/insauer/projects/NC_Submission/Data/postprocessing/AttributionMetaDataRegions.csv')

DATA_ATTR = pd.read_csv('/home/insauer/projects/NC_Submission/Data/postprocessing/AttributionMetaDataSubregions.csv')

region_names={'GLB': 'Global (GLB)',
              'NAM':'North America (NAM)',
              'CHN':'Eastern Asia (EAS)',
              'AUS':'Oceania (OCE)',
              'LAM':'Latin America (LAM)',
              'EUR':'Europe (EUR)',
              'SSAF':'South & Sub-Sahara Africa (SSA)',
              'SWEA':'South & South-East Asia (SEA)',
              'CAS':'Central Asia & Russia (CAS)',
              'NAFARA':'North Africa & Middle East (NAF)'}

region_abs={'GLB': 'GLB',
            'NAM':'NAM', 
            'CHN':'EAS',
            'LAM':'LAM', 
            'EUR':'EUR',
            'AUS':'OCE',
            'CAS':'CAS',
            'SSAF':'SSA',
            'SWEA':'SEA', 
            'NAFARA': 'NAF'}

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
            
            
            
        
                # f3_ax1.plot(DATA_regionFull['Year'], np.log10(DATA_regionFull['Impact_Pred_1thrd']), color='#8856a7', alpha = 0.5, linewidth = 1.)
                # f3_ax1.plot(DATA_regionFull['Year'], np.log10(DATA_regionFull['Impact_Pred_2thrd']), color='#8856a7', alpha = 0.5, linewidth = 1.)
                f3_ax1.plot(DATA_regionFull['Year'], np.log10(DATA_regionFull['natcat_flood_damages_2005_CPI']), label='D$_{Obs}$', color='black', linewidth = 1.) 
                f3_ax1.scatter(DATA_regionFull['Year'], np.log10(DATA_regionFull['natcat_flood_damages_2005_CPI']), color='black', marker = '_', s = 1) 
                f3_ax1.plot(DATA_regionFull['Year'], np.log10(DATA_regionFull['Norm_Impact_Pred']), label='D$_{Full}$', color='#8856a7', linewidth = 1.)
                
                f3_ax1.plot(DATA_regionFull['Year'], np.log10(DATA_regionFull['Norm_Impact_2y_offset']), label='D$_{CliExp}$', color='#ff7f00', linewidth = 1.)
            
                #f3_ax1.plot(DATA_regionFull['Year'], np.log10(DATA_regionFull['Norm_ImpFix_2y_offset']), label='$Loss_{Haz}$', color='#4575b4', linewidth = 1.)
                
                f3_ax1.plot(DATA_regionFull['Year'], np.log10(DATA_regionFull['Norm_Imp2010_2y_offset']), label='D$_{2010}$', color='seagreen', linewidth = 1.)
        
                
                
                #f3_ax1.fill_between(DATA_regionFull['Year'],np.log10(DATA_regionFull['Impact_Pred_1thrd']) , np.log10(DATA_regionFull['Impact_Pred_2thrd']), color='#8856a7', alpha=0.2, linewidth = 1.)
                if i ==5:
                
                    f3_ax1.set_title(' '+ region_names[regions[r]], position = (0.5,0.78), fontsize = 7.5)
                
                else:
                    
                    f3_ax1.set_title(' '+ region_names[regions[r]], position = (0.5,0.78), fontsize = 7.5)
                
                if i ==0 and j ==0:
                    handles, labels = f3_ax1.get_legend_handles_labels()
                    
                    # font = font_manager.FontProperties(family= 'Comic Sans MS',  # 'Times new roman', 
                    #                weight='bold',
                    #                style='oblique', size=6)
                    leg =f3_ax1.legend(handles[:2], labels[:2], loc ='lower left', labelspacing = 0.1, frameon=True, fontsize = 6, handlelength = 1.1) 
                    f3_ax1.legend(handles[2:], labels[2:], loc ='lower right', labelspacing = 0.1, frameon=True, fontsize = 6,  handlelength = 1.1)
                    f3_ax1.add_artist(leg)
        
                r_lin = DATA_FIT_Full.loc[DATA_FIT_Full['Region']==regions[r], 'P_ExpVar_pred_observed'].sum()

                #r2 = DATA_FIT_Full.loc[DATA_FIT_Full['Region']==regions[r], 'New_explained_variance'].sum()
               
            else:
                
                if j ==1:
                    dis = 'Pos'
                    f3_ax1.set_title('{}'.format(region_abs[regions[r]])+'$_{+}$', position = (0.5,0.78), fontsize = 7.5)
                else:
                    dis = 'Neg'
                    f3_ax1.set_title('{}'.format(region_abs[regions[r]])+'$_{-}$', position = (0.5,0.78), fontsize = 7.5)
                
                #f3_ax1.plot(DATA_region['Year'], np.log10(DATA_region['Impact_Pred_1thrd_{}'.format(dis)]), color='#8856a7', alpha = 0.5, linewidth = 1.)
                #f3_ax1.plot(DATA_region['Year'], np.log10(DATA_region['Impact_Pred_2thrd_{}'.format(dis)]), color='#8856a7', alpha = 0.5, linewidth = 1.)
                f3_ax1.plot(DATA_region['Year'], np.log10(DATA_region['natcat_damages_2005_CPI_{}'.format(dis)]), label='Observed Flood Losses (NatCat)', color='black', linewidth = 1.) 
                f3_ax1.scatter(DATA_region['Year'], np.log10(DATA_region['natcat_damages_2005_CPI_{}'.format(dis)]), label='Observed Flood Losses (NatCat)', color='black', marker = '.', s = 1) 
        
                if not (i ==1 and j==1):
                    f3_ax1.plot(DATA_region['Year'], np.log10(DATA_region['NormExp_Impact_2y{}_offset'.format(dis)]), label='$D_{CliExp}$', color='#ff7f00', linewidth = 1.)
                
                    #f3_ax1.plot(DATA_region['Year'], np.log10(DATA_region['NormHaz_ImpFix_2y{}_offset'.format(dis)]), label='$Loss_{Haz}$', color='#4575b4', linewidth = 1.)
                    
                    f3_ax1.plot(DATA_region['Year'], np.log10(DATA_region['NormHaz_Imp2010_2y{}_offset'.format(dis)]), label='$D_{2010}$', color='seagreen', linewidth = 1.)
            
                    f3_ax1.plot(DATA_region['Year'], np.log10(DATA_region['Norm_Impact_Pred_{}'.format(dis)]), label='$D_{Full}$', color='#8856a7', linewidth = 1.)
                

                
                r_lin = DATA_FIT.loc[DATA_FIT['Region']==regions[r]+'_'+dis, 'ExpVar_model_pred_observed'].sum()


            text_lin = '='+str(round(r_lin*100,1))+ '%'

            
            if r_lin> 0.2:
                
            
                f3_ax1.set_facecolor('gainsboro')
            
            f3_ax1.set_yticks([6,8,10])
            f3_ax1.set_yticklabels(['','',''])
            if j == 0 :
                f3_ax1.set_yticklabels(['6','8','10'], fontsize = 7.5)
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
                f3_ax1.set_ylim((2.8, 10))
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
                f3_ax1.annotate( xy=(1991.8, f3_ax1.get_ylim()[0]+0.08*(f3_ax1.get_ylim()[1]-f3_ax1.get_ylim()[0]) ) ,s='R²', fontsize=7, fontstyle='italic')
                f3_ax1.annotate( xy=(1993.6, f3_ax1.get_ylim()[0]+0.08*(f3_ax1.get_ylim()[1]-f3_ax1.get_ylim()[0]) ) ,s=text_lin, fontsize=7)
   
            f3_ax1.set_xticks([1980,1990,2000,2010])
            f3_ax1.set_xticklabels(['','','',''])
            
            
            if i == 9:
                f3_ax1.set_xticklabels(['1980','1990','2000','2010'], fontsize =7.5)
            
            if i ==4 and j ==0:
                f3_ax1.set_ylabel('log$_{10}$(Damages in 2005 USD)', fontsize=8, labelpad=-1)
            

            
            if i == 9 and j ==1:
                f3_ax1.set_xticklabels(['1980','1990','2000', '2010'],fontsize=7.5)
                f3_ax1.set_xlabel('Year', fontsize=9, labelpad=-2)
            f3_ax1.tick_params(axis="x", direction = 'in',length = 4)
            
            handles, labels = f3_ax1.get_legend_handles_labels() 
        
    else:
        
        f3_ax2 = fig3.add_subplot(gs[4*i:4*i+4,j*5:(j*5)+2])
        data_attr_reg = DATA_ATTR_Full[DATA_ATTR_Full['Region'] == regions[r]]
        data_attr_reg_pos = DATA_ATTR[DATA_ATTR['Region'] == regions[r]+'_Pos']
        data_attr_reg_neg = DATA_ATTR[DATA_ATTR['Region'] == regions[r]+'_Neg']
        
        r_linPos = DATA_FIT.loc[DATA_FIT['Region']==regions[r]+'_Pos', 'ExpVar_model_pred_observed'].sum()
        r_linNeg = DATA_FIT.loc[DATA_FIT['Region']==regions[r]+'_Neg', 'ExpVar_model_pred_observed'].sum()
        r_lin = DATA_FIT_Full.loc[DATA_FIT_Full['Region']==regions[r], 'P_ExpVar_pred_observed'].sum()
        
        if r_linPos > 0.2:
        
            f3_ax2.axvspan(4, 9, facecolor='gainsboro')
            
        if r_linNeg > 0.2:
        
            f3_ax2.axvspan(9, 14, facecolor='gainsboro')
            
        if r_lin > 0.2:
        
            f3_ax2.axvspan(-1, 4, facecolor='gainsboro')
        
        h8 = data_attr_reg.loc[:,'Change H10nCL'].sum()
        e8 = data_attr_reg.loc[:,'Change En'].sum()
        v8 = data_attr_reg.loc[:,'Change Vn'].sum()
        i8 = data_attr_reg.loc[:,'Change In'].sum()
        n8 = data_attr_reg.loc[:,'Change Nn'].sum()
        
        
        h_pos8 =  data_attr_reg_pos.loc[:,'Change H10nCL'].sum()
        e_pos8 = data_attr_reg_pos.loc[:,'Change En'].sum()
        v_pos8 = data_attr_reg_pos.loc[:,'Change Vn'].sum()
        i_pos8 = data_attr_reg_pos.loc[:,'Change In'].sum()
        n_pos8 = data_attr_reg_pos.loc[:,'Change Nn'].sum()
        
        
        h_neg8 =  data_attr_reg_neg.loc[:,'Change H10nCL'].sum()
        e_neg8 =  data_attr_reg_neg.loc[:,'Change En'].sum()
        v_neg8 =  data_attr_reg_neg.loc[:,'Change Vn'].sum()
        i_neg8 =  data_attr_reg_neg.loc[:,'Change In'].sum()
        n_neg8 =  data_attr_reg_neg.loc[:,'Change Nn'].sum()
        
        h8_sig = data_attr_reg.loc[:,'Sign H10'].sum()
        h8_pos_sig = data_attr_reg_pos.loc[:,'Sign H10'].sum()
        h8_neg_sig = data_attr_reg_neg.loc[:,'Sign H10'].sum()
        sig = [h8_sig, h8_pos_sig, h8_neg_sig]

        y=[h8,e8,v8,i8,h_pos8,e_pos8,v_pos8, i_pos8, h_neg8,e_neg8,v_neg8, i_neg8]
        x=[0,1,2,3,5,6,7,8,10,11,12,13]
        colour_code = ['#06470c','gold','brown','mediumpurple','#06470c','gold','brown','mediumpurple','#06470c','gold','brown','mediumpurple']
        #y= [h_pos8,h8,h_neg8,h_pos7,h7,h_neg7]
        c=0
        for a in range(12):
            if a in [0, 4, 8]:
                if sig[c]<0.1:
                    f3_ax2.bar(x[a],y[a], color =colour_code[a])
                    
                else:
                    f3_ax2.bar(x[a],y[a], color ='#6fc276')
                c+=1
            else:
                f3_ax2.bar(x[a],y[a], color =colour_code[a])
           
        #f3_ax2.scatter([3,8,13], [n8,n_pos8,n_neg8],width = 0.3, color ='black')
        

        
        f3_ax2.set_xticks([0,1,2,3,5,6,7,8,10,11,12,13])
        f3_ax2.tick_params(axis="x", direction = 'in',length = 2)
        
        
        #f3_ax2.set_yticklabels(['-5%','5%'], fontsize = 7)
        f3_ax2.set_xticklabels(['C', 'E', 'V', 'M', 'C', 'E', 'V', 'M','C', 'E', 'V', 'M'], rotation = 0.,fontsize = 5, fontstyle = 'italic')
        f3_ax2.axhline(y=0,linewidth=0.5, color='k')
        f3_ax2.axvline(x=4,linewidth=0.3, color='k', linestyle = '-')
        f3_ax2.axvline(x=9,linewidth=0.3, color='k', linestyle = '-')
        #f3_ax2.axhline(y=-5,linewidth=0.2, color='k', linestyle = '-.')
        f3_ax2.yaxis.tick_right()
        f3_ax2.yaxis.set_label_position("right")
        f3_ax2.tick_params(axis="y", direction = 'out',length = 2)
        
        #f3_ax2.set_title('80+   71+ ', fontsize = 7, position = (0.5,0.82))
        
        if i == 4:
            f3_ax2.set_ylabel('Relative Trend in % per year', fontsize=7, labelpad = 2.5)
        
        # if i in [2]:
        #     f3_ax2.set_ylim(-30,30)
        #     f3_ax2.set_yticks([-15,15])
        #     f3_ax2.set_yticklabels(['-15%','15%'], fontsize = 6)
            
        if i in [2]:
            f3_ax2.set_ylim(-7,15)
            f3_ax2.set_yticks([-5,5])
            f3_ax2.set_yticklabels(['-5%','5%'], fontsize = 6)
         
        elif i in [6]:
            f3_ax2.set_ylim(-2, 37)
            
            f3_ax2.set_yticks([0, 20])
            f3_ax2.set_yticklabels(['0%','20%'], fontsize = 6)
        
        elif i in [8]:
            f3_ax2.set_ylim(-2, 7)
            
            f3_ax2.set_yticks([-1,2])
            f3_ax2.set_yticklabels(['-1%','2%'], fontsize = 6)
            
        elif i in [3]:
            f3_ax2.set_ylim(-1.5, 2)
            
            f3_ax2.set_yticks([-1, 1])
            f3_ax2.set_yticklabels(['-1%','1%'], fontsize = 6)
            
        elif i in [5]:
            f3_ax2.set_ylim(-5,10)
            
            f3_ax2.set_yticks([-2,5])
            f3_ax2.set_yticklabels(['-2%','5%'], fontsize = 6)
            
        elif i in [7]:
            f3_ax2.set_ylim(-5,27)
            
            f3_ax2.set_yticks([-2,10])
            f3_ax2.set_yticklabels(['-2%','10%'], fontsize = 6)

            
        elif i in [9]:
            f3_ax2.set_ylim(-5,15)
            
            f3_ax2.set_yticks([-2,5])
            f3_ax2.set_yticklabels(['-2%','5%'], fontsize = 6)
            
        elif i in [4]:
            f3_ax2.set_ylim(-5,5)
            
            f3_ax2.set_yticks([-2,2])
            f3_ax2.set_yticklabels(['-5%','5%'], fontsize = 6)
        
        
        elif i in [1]:
            f3_ax2.set_ylim(-1,3)
            f3_ax2.set_yticks([0,2])
            f3_ax2.set_yticklabels(['0%','2%'], fontsize = 6)
        
        else:

            f3_ax2.set_ylim(-5,15)
            f3_ax2.set_yticks([-2,5])
            f3_ax2.set_yticklabels(['-2%','5%'], fontsize = 6)
        
        
        f3_ax2.set_xlim(-1.,14)
        
        nes = [n8,n_pos8,n_neg8]
        
        xn = [4,9,14]
        
        xx= [3,8,13]
        
        for ns in range(3):
            if nes[ns]>0:
                lims = np.abs(f3_ax2.get_ylim()[0])/(f3_ax2.get_ylim()[1]-f3_ax2.get_ylim()[0]) + nes[ns]/(f3_ax2.get_ylim()[1]-f3_ax2.get_ylim()[0])
                zerlim = np.abs(f3_ax2.get_ylim()[0])/(f3_ax2.get_ylim()[1]-f3_ax2.get_ylim()[0])
                f3_ax2.axhline(y=nes[ns], xmin=(xn[ns]-0.4)/15, xmax=(xn[ns]+0.4)/15,linewidth=1, color='k')
                f3_ax2.axvline(x=xx[ns]-0.4, ymin=zerlim, ymax=lims,linewidth=1., color='k')
                f3_ax2.axvline(x=xx[ns]+0.4, ymin=zerlim, ymax=lims,linewidth=1., color='k')
            else:
                lims = np.abs(f3_ax2.get_ylim()[0])/(f3_ax2.get_ylim()[1]-f3_ax2.get_ylim()[0]) + nes[ns]/(f3_ax2.get_ylim()[1]-f3_ax2.get_ylim()[0])
                zerlim = np.abs(f3_ax2.get_ylim()[0])/(f3_ax2.get_ylim()[1]-f3_ax2.get_ylim()[0])
                f3_ax2.axhline(y=nes[ns], xmin=(xn[ns]-0.4)/15, xmax=(xn[ns]+0.4)/15,linewidth=1, color='k')
                f3_ax2.axvline(x=(xx[ns]-0.4), ymin=lims, ymax=zerlim,linewidth=1, color='k')
                f3_ax2.axvline(x=(xx[ns]+0.4), ymin=lims, ymax=zerlim,linewidth=1, color='k')
            
        f3_ax2.annotate( xy=(1., f3_ax2.get_ylim()[1]-0.11*(f3_ax2.get_ylim()[1]-f3_ax2.get_ylim()[0]) ) ,s= '$R$', fontsize=7 )
        f3_ax2.annotate( xy=(6., f3_ax2.get_ylim()[1]-0.11*(f3_ax2.get_ylim()[1]-f3_ax2.get_ylim()[0]) ) ,s= '$R_{+}$', fontsize=7 )    
        f3_ax2.annotate( xy=(11., f3_ax2.get_ylim()[1]-0.11*(f3_ax2.get_ylim()[1]-f3_ax2.get_ylim()[0]) ) ,s= '$R_{-}$', fontsize=7 )
    r+=1
        

plt.savefig('/home/insauer/projects/NC_Submission/Data/Figures/Supplement/SI6_Figure_2010.png',bbox_inches = 'tight',dpi =600)
plt.savefig('/home/insauer/projects/NC_Submission/Data/Figures/Supplement/SI6_Figure_2010.pdf',bbox_inches = 'tight', format = 'pdf')




#f3_ax1.set_title('gs[0, :]')
#f3_ax2 = fig3.add_subplot(gs[1, :-1])
#f3_ax2.set_title('gs[1, :-1]')
#f3_ax3 = fig3.add_subplot(gs[1:, -1])
#f3_ax3.set_title('gs[1:, -1]')
#f3_ax4 = fig3.add_subplot(gs[-1, 0])
#f3_ax4.set_title('gs[-1, 0]')
#f3_ax5 = fig3.add_subplot(gs[-1, -2])
#f3_ax5.set_title('gs[-1, -2]')