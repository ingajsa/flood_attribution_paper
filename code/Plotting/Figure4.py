#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 22:09:13 2020

@author: insauer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

"""Script to reproduce main Figure 4!"""

fig3 = plt.figure(constrained_layout=True, figsize=(7., 9.9))
gs = fig3.add_gridspec(20, 14)
plt.subplots_adjust(wspace=-0.2, hspace=0)


DATA_ATTR_Full = pd.read_csv('../../data/reconstruction/attribution_MetaData_regions.csv')

DATA_ATTR = pd.read_csv('../../data/reconstruction/attribution_MetaData_subregions.csv')

DATA_FIT_Full = pd.read_csv('../../data/reconstruction/vulnerability_adjustment_MetaData_regions.csv')
DATA_FIT = pd.read_csv('../../data/reconstruction/vulnerability_adjustment_MetaData_subregions.csv')


region_names={'GLB': 'Global (GLB)',
              'NAM':'North America (NAM)',
              'EAS':'Eastern Asia (EAS)',
              'OCE':'Oceania (OCE)',
              'LAM':'Latin America (LAM)',
              'EUR':'Europe (EUR)',
              'CAS':'Central Asia (CAS)',
              'SSA':'South & Sub-Sahara Africa (SSA)',
              'SEA':'South & South-East Asia (SEA)', 
              'NAF':'North Africa & Middle East (NAF)'
              }

region_abs={'NAM':'NAM', 
          'LAM':'LAM', 
          'EUR':'WEU',
          'NAF':'NAR',
          'SSA':'SSA',
          'CAS':'CAS',
          'SEA':'SEA', 
          'EAS':'EAS', 
          'OCE':'OCE',
          'GLB': 'GLB'}

regions = list(region_names)

#order = [9,0,1,2,]
r =0

for i in range(4):
    for j in range(3):
        
        if i == 0 and (j ==0 or j ==2):
            continue
        
        
        f3_ax1 = fig3.add_subplot(gs[5*i:5*i+2,j*5:(j*5)+4])
        f3_ax2 = fig3.add_subplot(gs[5*i+2:5*i+4,j*5:(j*5)+4])
        

 
        
        r_lin = DATA_FIT_Full.loc[DATA_FIT_Full['Region']==regions[r], 'R2_D_Full_D_Obs'].sum()     

        r_lin_pos = DATA_FIT.loc[DATA_FIT['Region']==regions[r]+'_pos', 'R2_D_Full_D_Obs'].sum()
        r_lin_neg = DATA_FIT.loc[DATA_FIT['Region']==regions[r]+'_neg', 'R2_D_Full_D_Obs'].sum()
        
        data_attr_reg = DATA_ATTR_Full[DATA_ATTR_Full['Region'] == regions[r]]
        data_attr_reg_pos = DATA_ATTR[DATA_ATTR['Region'] == regions[r]+'_pos']
        data_attr_reg_neg = DATA_ATTR[DATA_ATTR['Region'] == regions[r]+'_neg']
        

        h7 = data_attr_reg.loc[:,'C_1980_71'].sum()
        h_pos7 =  data_attr_reg_pos.loc[:,'C_1980_71'].sum()
        h_neg7 =  data_attr_reg_neg.loc[:,'C_1980_71'].sum()
        
        h7_up = data_attr_reg.loc[:,'C_1980_71up'].sum()
        h_pos7_up =  data_attr_reg_pos.loc[:,'C_1980_71up'].sum()
        h_neg7_up =  data_attr_reg_neg.loc[:,'C_1980_71up'].sum()
        
        h7_bot = data_attr_reg.loc[:,'C_1980_71bot'].sum()
        h_pos7_bot =  data_attr_reg_pos.loc[:,'C_1980_71bot'].sum()
        h_neg7_bot =  data_attr_reg_neg.loc[:,'C_1980_71bot'].sum()
        
        
        h8 = data_attr_reg.loc[:,'C_1980_80'].sum()
        h_pos8 =  data_attr_reg_pos.loc[:,'C_1980_80'].sum()
        h_neg8 =  data_attr_reg_neg.loc[:,'C_1980_80'].sum()
        
        h8_up = data_attr_reg.loc[:,'C_1980_80up'].sum()
        h_pos8_up =  data_attr_reg_pos.loc[:,'C_1980_80up'].sum()
        h_neg8_up =  data_attr_reg_neg.loc[:,'C_1980_80up'].sum()
        
        h8_bot = data_attr_reg.loc[:,'C_1980_80bot'].sum()
        h_pos8_bot =  data_attr_reg_pos.loc[:,'C_1980_80bot'].sum()
        h_neg8_bot =  data_attr_reg_neg.loc[:,'C_1980_80bot'].sum()
        
        h8_sig = data_attr_reg.loc[:,'p_val_C_1980_80'].sum()
        h_pos8_sig =  data_attr_reg_pos.loc[:,'p_val_C_1980_80'].sum()
        h_neg8_sig =  data_attr_reg_neg.loc[:,'p_val_C_1980_80'].sum()
        
        h7_sig = data_attr_reg.loc[:,'p_val_C_1980_71'].sum()
        h_pos7_sig =  data_attr_reg_pos.loc[:,'p_val_C_1980_71'].sum()
        h_neg7_sig =  data_attr_reg_neg.loc[:,'p_val_C_1980_71'].sum()
        
        h107 = data_attr_reg.loc[:,'C_2010_71'].sum()
        h_pos107 =  data_attr_reg_pos.loc[:,'C_2010_71'].sum()
        h_neg107 =  data_attr_reg_neg.loc[:,'C_2010_71'].sum()
        
        h107_up = data_attr_reg.loc[:,'C_2010_71up'].sum()
        h_pos107_up =  data_attr_reg_pos.loc[:,'C_2010_71up'].sum()
        h_neg107_up =  data_attr_reg_neg.loc[:,'C_2010_71up'].sum()
        
        h107_bot = data_attr_reg.loc[:,'C_2010_71bot'].sum()
        h_pos107_bot =  data_attr_reg_pos.loc[:,'C_2010_71bot'].sum()
        h_neg107_bot =  data_attr_reg_neg.loc[:,'C_2010_71bot'].sum()
        
        h108 = data_attr_reg.loc[:,'C_2010_80'].sum()
        h_pos108 =  data_attr_reg_pos.loc[:,'C_2010_80'].sum()
        h_neg108 =  data_attr_reg_neg.loc[:,'C_2010_80'].sum()
        
        h108_up = data_attr_reg.loc[:,'C_2010_80up'].sum()
        h_pos108_up =  data_attr_reg_pos.loc[:,'C_2010_80up'].sum()
        h_neg108_up =  data_attr_reg_neg.loc[:,'C_2010_80up'].sum()
        
        h108_bot = data_attr_reg.loc[:,'C_2010_80bot'].sum()
        h_pos108_bot =  data_attr_reg_pos.loc[:,'C_2010_80bot'].sum()
        h_neg108_bot =  data_attr_reg_neg.loc[:,'C_2010_80bot'].sum()
        
        h108_sig = data_attr_reg.loc[:,'p_val_C_2010_80'].sum()
        h_pos108_sig =  data_attr_reg_pos.loc[:,'p_val_C_2010_80'].sum()
        h_neg108_sig =  data_attr_reg_neg.loc[:,'p_val_C_2010_80'].sum()
        
        h107_sig = data_attr_reg.loc[:,'p_val_C_2010_71'].sum()
        h_pos107_sig =  data_attr_reg_pos.loc[:,'p_val_C_2010_71'].sum()
        h_neg107_sig =  data_attr_reg_neg.loc[:,'p_val_C_2010_71'].sum()

        x=[0,1,2,4,5,6]
        colour_code = ['#4575b4', '#4575b4', '#4575b4', '#4575b4', '#4575b4', '#4575b4']
        y1= [h_pos8,h8,h_neg8,h_pos7,h7,h_neg7]
        y1err_up= [h_pos8_up,h8_up,h_neg8_up,h_pos7_up,h7_up,h_neg7_up]
        y1err_bot= [h_pos8_bot,h8_bot,h_neg8_bot,h_pos7_bot,h7_bot,h_neg7_bot]
        y1_sig= [h_pos8_sig,h8_sig,h_neg8_sig,h_pos7_sig,h7_sig,h_neg7_sig]
        
        y2= [h_pos108,h108,h_neg108,h_pos107,h107,h_neg107]
        y2err_up= [h_pos108_up,h108_up,h_neg108_up,h_pos107_up,h107_up,h_neg107_up]
        y2err_bot= [h_pos108_bot,h108_bot,h_neg108_bot,h_pos107_bot,h107_bot,h_neg107_bot]
        
        y2_sig= [h_pos108_sig,h108_sig,h_neg108_sig,h_pos107_sig,h107_sig,h_neg107_sig]
        
        ax1_lims_up = [10,10,10,6,5,8,5,35,20,15]
        
        ax1_lims_low = [-6,-10,-5,-5.2,-4,-8,-5,-5,-10,-10]
        
        ax2_lims_up = [10,10,10,6,5,8,5,35,20,15]
        
        ax2_lims_low = [-6,-10,-5,-5,-4,-8,-5,-5,-10,-10]
        
        ax1_ticks_up = [5,5,5,3,2,5,2,20,10,5]
        
        ax1_ticks_low = [-5,-5,-5,-3,-2,-5,-2,-5,-5,-5]
        
        ax2_ticks_up = [5,5,5,3,2,5,2,20,10,5]
        
        ax2_ticks_low = [-5,-5,-5,-3,-2,-5,-2,-5,-5, -5]
        
        ax1_labels_up = ['5%', '5%','5%','3%','2%','5%', '2%', '20%','10%','5%']
        
        ax1_labels_low = ['-5%','-5%','-5%','-3%','-2%','-5','-2%','','-5%','-5%']
        
        ax2_labels_up = ['5%','5%','5%','3%','2%', '5%','2%', '20%','10%','5%']
        
        ax2_labels_low = ['-5%','-5%','-5%','-3%','-2%','-5%','-2%','','-5%','-5%']
        
        if r_lin_pos > 0.2*100:
        
            f3_ax1.axvspan(-1,0.5, facecolor='gainsboro')
            f3_ax2.axvspan(-1,0.5, facecolor='gainsboro')
            f3_ax1.axvspan(3,4.5, facecolor='gainsboro')
            f3_ax2.axvspan(3,4.5, facecolor='gainsboro')
            #f3_ax2.axvspan(4, 9, facecolor='gainsboro')
            
        if r_lin_neg > 0.2*100:
            f3_ax1.axvspan(1.5,3., facecolor='gainsboro')
            f3_ax2.axvspan(1.5,3., facecolor='gainsboro')
            f3_ax1.axvspan(5.5,7., facecolor='gainsboro')
            f3_ax2.axvspan(5.5,7., facecolor='gainsboro')
            
        if r_lin > 0.2*100:
            
            f3_ax1.axvspan(0.5,1.5, facecolor='gainsboro')
            f3_ax2.axvspan(0.5,1.5, facecolor='gainsboro')
            f3_ax1.axvspan(4.5,5.5, facecolor='gainsboro')
            f3_ax2.axvspan(4.5,5.5, facecolor='gainsboro')
        
        
        for a in range(6):
            
            if not ((r==1 and a==0)or (r==1 and a==3)) :

                if y1_sig[a]<0.1:
                    if y1_sig[a] < 0.01:
                        
                        f3_ax1.errorbar(x[a],y1[a], color =colour_code[a], fmt='D', yerr=np.reshape(np.array([y1err_bot[a],y1err_up[a]]), (2,1)),ecolor=colour_code[a])
                    elif y1_sig[a] < 0.05:
                        
                        f3_ax1.errorbar(x[a],y1[a], color =colour_code[a], fmt='s', yerr=np.reshape(np.array([y1err_bot[a],y1err_up[a]]), (2,1)),ecolor=colour_code[a])
                    
                    else:
                        f3_ax1.errorbar(x[a],y1[a], color =colour_code[a], fmt='^', yerr=np.reshape(np.array([y1err_bot[a],y1err_up[a]]), (2,1)),ecolor=colour_code[a])
                            
                            
                else:
                    f3_ax1.errorbar(x[a],y1[a], color = colour_code[a], fmt='o', yerr=np.reshape(np.array([y1err_bot[a],y1err_up[a]]), (2,1)), ecolor=colour_code[a],mfc ='w')
            
        for a in range(6):
            
            if not ((r==1 and a==0)or (r==1 and a==3)) :

                if y2_sig[a]<0.1:
                    
                    if y2_sig[a] < 0.01:
                        
                        f3_ax2.errorbar(x[a],y2[a], color =colour_code[a], yerr=np.reshape(np.array([y2err_bot[a],y2err_up[a]]), (2,1)), ecolor=colour_code[a], fmt= 'D')
                    
                    elif y2_sig[a] < 0.05:
                        
                        f3_ax2.errorbar(x[a],y2[a], color =colour_code[a], yerr=np.reshape(np.array([y2err_bot[a],y2err_up[a]]), (2,1)), ecolor=colour_code[a], fmt= 's')
                    else:
                        
                        f3_ax2.errorbar(x[a],y2[a], color =colour_code[a], yerr=np.reshape(np.array([y2err_bot[a],y2err_up[a]]), (2,1)), ecolor=colour_code[a], fmt= '^')
                    
                else:
                    f3_ax2.errorbar(x[a],y2[a], color =colour_code[a], yerr=np.reshape(np.array([y2err_bot[a],y2err_up[a]]), (2,1)), ecolor=colour_code[a], fmt='o',mfc ='w')
                    
        
        f3_ax1.set_ylim(ax1_lims_low[r],ax1_lims_up[r])
        
        f3_ax1.set_yticks([ax1_ticks_low[r] , 0, ax1_ticks_up[r]])
        
        f3_ax1.set_yticklabels([ax1_labels_low[r],'0', ax1_labels_up[r] ],fontsize =5.5)
        
        f3_ax2.set_ylim(ax2_lims_low[r],ax2_lims_up[r])
        
        f3_ax2.set_yticks([ax2_ticks_low[r] , 0, ax2_ticks_up[r]])
        
        f3_ax2.set_yticklabels([ax2_labels_low[r],'0', ax2_labels_up[r] ],fontsize =5.5)
        f3_ax1.tick_params(axis="y",length = 3, pad = 0.7)
        f3_ax2.tick_params(axis="y",length = 3, pad = 0.7)
        f3_ax1.set_xlim(-1.,7)
        f3_ax2.set_xlim(-1.,7)
        
        

        f3_ax1.axvline(x=3,linewidth=0.3, color='k', linestyle = '-', alpha = 0.5)
        f3_ax2.axvline(x=3,linewidth=0.3, color='k', linestyle = '-', alpha = 0.5)
        
        f3_ax2.set_xticks([0,1,2,4,5,6])
        
        f3_ax2.set_xticklabels(['$R_{+}$', '$R$', '$R_{-}$','$R_{+}$',  '$R$', '$R_{-}$'], fontsize = 7)
        
        lab = [0,1,4,7]
        
        if r in lab:
            f3_ax1.set_ylabel('$C_{1980}$ in % per year',  fontsize = 6.5, labelpad=0.1)
            f3_ax2.set_ylabel('$C_{2010}$',  fontsize = 6.5, labelpad=0.1, )

        f3_ax1.set_xlabel('1980-2010               1971-2010',  fontsize = 5.5, labelpad=-7)
        f3_ax1.xaxis.set_label_position('top')
        #f3_ax2.axhline(y=-5,linewidth=0.2, color='k', linestyle = '-.')
        f3_ax1.axhline(y=0,linewidth=0.3, color='k', linestyle = '-')
        f3_ax2.axhline(y=0,linewidth=0.3, color='k', linestyle = '-')
        
        if (i == 1) and (j==1):
            f3_ax1.set_title(' '+ region_names[regions[r]], position = (0.5,1.), fontsize = 7)
            
        elif (i == 2) and ((j ==0) or (j==1)):
            
            f3_ax1.set_title(' '+ region_names[regions[r]], position = (0.5,1.), fontsize = 7)
        else:
            f3_ax1.set_title(' '+ region_names[regions[r]], position = (0.5,1.), fontsize = 7)
         
        
        r+=1

f3_ax4 = fig3.add_subplot(gs[0:5,10:14])
handles, labels = f3_ax4.get_legend_handles_labels()
f3_ax4.axis('off')



circle = Line2D([0], [0], marker='o', color='w', label='non-significant',
                        markeredgecolor='#4575b4', markersize=5.5)
triangle = Line2D([0], [0], marker='^', color='w', label='significant at 10%',
                        markerfacecolor='#4575b4', markersize=6.5)
square = Line2D([0], [0], marker='s', color='w', label='significant at 5%',
                        markerfacecolor='#4575b4', markersize=6)
diam = Line2D([0], [0], marker='D', color='w', label='significant at 1%',
                        markerfacecolor='#4575b4', markersize=5.5)


f3_ax4.legend(handles = [diam, square,triangle,circle], frameon=True, fontsize = 6, loc = 'center', edgecolor = 'k')  


plt.savefig('../../data/figures/Figure4.pdf',bbox_inches = 'tight', format = 'pdf')

