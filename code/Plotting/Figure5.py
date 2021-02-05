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

"""Script to reproduce main Figure 5!"""

fig3 = plt.figure(constrained_layout=True, figsize=(7.0, 8.8))
gs = fig3.add_gridspec(19, 14)
plt.subplots_adjust(wspace=0., hspace=0)


DATA_ATTR_Full = pd.read_csv('../../../data/reconstruction/ENSO_GMT_PDO_NAO_regions.csv')

DATA_ATTR = pd.read_csv('../../../data/reconstruction/ENSO_GMT_PDO_NAO_subregions.csv')

teleDataNorm = pd.read_csv('../../../data/climate_oscillations/norm_oscillations_lag.csv')

AMO_DATA_ATTR_Full = pd.read_csv('../../../data/reconstruction/ENSO_AMO_PDO_NAO_regions.csv')

AMO_DATA_ATTR = pd.read_csv('../../data/reconstruction/ENSO_AMO_PDO_NAO_subregions.csv')

region_names={ 'GLB': 'Global (GLB)',
              'NAM':'North America (NAM)',
              'EAS':'Eastern Asia (EAS)',
              'OCE':'Oceania (OCE)',
              'LAM':'Latin America (LAM)',
              'EUR':'Europe (EUR)',
              'SEA':'South & South-East Asia (SEA)',
              'CAS':'Central Asia & Russia (CAS)',
              }

region_abs={'NAM':'NAM', 
          'LAM':'LAM', 
          'EUR':'EUR',
          'NAF':'NAR',
          'SSA':'SSA',
          'CAS':'CAS',
          'SEA':'SEA', 
          'EAS':'EAS', 
          'OCE':'OCE',
          'GLB': 'GLB'}

regions = list(region_names)

three = [2]
two = [0,1,3,4]
one = [5,6,7]

r =0

for i in range(1):
    #r =0
    
    z =0
    for j in range(8):
        
        if r > 8:
            continue
        if r in one:
            f3_ax1 = fig3.add_subplot(gs[5*(i+1):5*(i+1)+4,z:z+1])
            z = z+1
        elif r in two:
            f3_ax1 = fig3.add_subplot(gs[5*(i+1):5*(i+1)+4,z:z+2])
            z = z+2
        else:
            f3_ax1 = fig3.add_subplot(gs[5*(i+1):5*(i+1)+4,z:z+3])
            z = z+3
        #f3_ax2 = fig3.add_subplot(gs[5*i+2:5*i+4,j*5:(j*5)+4])



        data_attr_reg = AMO_DATA_ATTR_Full[AMO_DATA_ATTR_Full['Region'] == regions[r]]
        data_attr_reg_pos = AMO_DATA_ATTR[AMO_DATA_ATTR['Region'] == regions[r]+'_pos']
        data_attr_reg_neg = AMO_DATA_ATTR[AMO_DATA_ATTR['Region'] == regions[r]+'_neg']
        


        tele = data_attr_reg[['gammaAbs_ENSO_amo_run','gammaAbs_PDO_amo_run','gammaAbs_NAO_amo_run',
                              'gammaAbs_AMO_amo_run','gammaAbs_ENSOlag_amo_run',
                              'gammaAbs_PDOlag_amo_run','gammaAbs_NAOlag_amo_run']]
        tele_pos =  data_attr_reg_pos[['gammaAbs_ENSO_amo_run','gammaAbs_PDO_amo_run','gammaAbs_NAO_amo_run',
                                       'gammaAbs_AMO_amo_run','gammaAbs_ENSOlag_amo_run',
                                       'gammaAbs_PDOlag_amo_run','gammaAbs_NAOlag_amo_run']]
        tele_neg =  data_attr_reg_neg[['gammaAbs_ENSO_amo_run','gammaAbs_PDO_amo_run','gammaAbs_NAO_amo_run',
                                       'gammaAbs_AMO_amo_run','gammaAbs_ENSOlag_amo_run',
                                       'gammaAbs_PDOlag_amo_run','gammaAbs_NAOlag_amo_run']]
        
        
        tele_pv = data_attr_reg[['pval_ENSO_amo_run','pval_PDO_amo_run','pval_NAO_amo_run',
                                 'pval_AMO_amo_run','pval_ENSOlag_amo_run','pval_PDOlag_amo_run',
                                 'pval_NAOlag_amo_run']]
        tele_pos_pv =  data_attr_reg_pos[['pval_ENSO_amo_run','pval_PDO_amo_run',
                                          'pval_NAO_amo_run', 'pval_AMO_amo_run',
                                          'pval_ENSOlag_amo_run','pval_PDOlag_amo_run',
                                          'pval_NAOlag_amo_run']]
        tele_neg_pv =  data_attr_reg_neg[['pval_ENSO_amo_run','pval_PDO_amo_run',
                                          'pval_NAO_amo_run', 'pval_AMO_amo_run',
                                          'pval_ENSOlag_amo_run','pval_PDOlag_amo_run',
                                          'pval_NAOlag_amo_run']]
        
        
        
        
        tele_pv = [tele_pos_pv, tele_pv, tele_neg_pv]
        
        tele_cons = [tele_pos, tele, tele_neg]
        
        df_teles=pd.DataFrame(columns= ['limit','tele_pos', 'tele', 'tele_neg'])
                              
        tele_names = ['tele_pos', 'tele', 'tele_neg']
        
        df_teles['limit'] = ['ENSO_up', 'ENSO_bot','ENSO_pv','ENSO_lw',\
                 'PDO_up', 'PDO_bot','PDO_pv', 'PDO_lw','NAO_up', 'NAO_bot', 'NAO_pv', 'NAO_lw',\
                 'AMO_up', 'AMO_bot','AMO_pv','AMO_lw',\
                 'ENSOlag_up', 'ENSOlag_bot','ENSOlag_pv', 'ENSOlag_lw',\
                 'PDOlag_up', 'PDOlag_bot','PDOlag_pv', 'PDOlag_lw', 'NAOlag_up',
                 'NAOlag_bot', 'NAOlag_pv', 'NAOlag_lw']
        
        preds = ['ENSO','PDO','NAO', 'AMO', 'ENSOlag','PDOlag','NAOlag']
        
        for t,tel in enumerate(tele_cons):
            
            abs_tel = np.nan_to_num(np.abs(tel))
            sum_tel = np.sum(np.array(abs_tel))

            perc = np.nan_to_num(tel/sum_tel)
            
            bottom = np.array(perc)[np.less(perc,0)].sum(axis =0)
            
            
            for m in np.argsort(perc)[0]:
                
                df_teles.loc[df_teles['limit']==preds[m]+'_bot', tele_names[t]] = bottom
                
                df_teles.loc[df_teles['limit']==preds[m]+'_up', tele_names[t]] = np.abs(perc[0,m])
            
            
                bottom += np.abs(perc[0,m])
                
        t_iter = 0
        
        shortage=['']
        
        shor = 0
        
        for t,tel_pv in enumerate(tele_pv):
            
            if t in [3,6,9]:
                shor+=1
            
            
            for p in preds:
                
                if tel_pv['pval_{}_amo_run'.format(p)].values[0]<0.1:
                    
                    alpha = 'k'
                    lw = 1.5
                else:
                    alpha = 'w'
                    lw = 0.
                    
                df_teles.loc[df_teles['limit']==p+'_pv', tele_names[t]] = alpha
                df_teles.loc[df_teles['limit']==p+'_lw', tele_names[t]] = lw


        inds= [[0,1],[1,2],[0,1,2],[1,2], [0,1], [1], [0],[0]]
        
        if r in three:
        
            x = [0,1,2]
        
        elif r in two:
        
            x = [0,1]
        
        else:
        
            x = [0]
        
        
        colour_code = ['#5ab4ac', '#4575b4', '#d8b365', '#5ab4ac', '#4575b4', '#d8b365']
        
        df_teles_a1 = df_teles.iloc[:,0:4]
        
        cmap = plt.cm.get_cmap('tab20b')
        
        rgba_oce_d = cmap(0.02)
        rgba_oce_l = cmap(0.17)
        
        col_enso = rgba_oce_d
        
        f3_ax1.bar(np.array(x), np.array(df_teles_a1[df_teles_a1['limit']=='ENSO_up'])[0,1:][inds[r]], 
        bottom = np.array(df_teles_a1[df_teles_a1['limit']=='ENSO_bot'])[0,1:][inds[r]], color = 'steelblue',
        edgecolor=np.array(df_teles_a1[df_teles_a1['limit']=='ENSO_pv'])[0,1:][inds[r]],
        linewidth = np.array(df_teles_a1[df_teles_a1['limit']=='ENSO_lw'])[0,1:][inds[r]])
        
        f3_ax1.bar(np.array(x), np.array(df_teles_a1[df_teles_a1['limit']=='ENSOlag_up'])[0,1:][inds[r]], 
        bottom = np.array(df_teles_a1[df_teles_a1['limit']=='ENSOlag_bot'])[0,1:][inds[r]], color = 'steelblue',
        edgecolor = np.array(df_teles_a1[df_teles_a1['limit']=='ENSOlag_pv'])[0,1:][inds[r]],
        linewidth = np.array(df_teles_a1[df_teles_a1['limit']=='ENSOlag_lw'])[0,1:][inds[r]])
        
        
        f3_ax1.bar(np.array(x), np.array(df_teles_a1[df_teles_a1['limit']=='PDO_up'])[0,1:][inds[r]], 
        bottom = np.array(df_teles_a1[df_teles_a1['limit']=='PDO_bot'])[0,1:][inds[r]], color = 'gold', label = 'PDO',
        edgecolor=np.array(df_teles_a1[df_teles_a1['limit']=='PDO_pv'])[0,1:][inds[r]],
        linewidth = np.array(df_teles_a1[df_teles_a1['limit']=='PDO_lw'])[0,1:][inds[r]])
        
        f3_ax1.bar(np.array(x), np.array(df_teles_a1[df_teles_a1['limit']=='NAO_up'])[0,1:][inds[r]], 
        bottom = np.array(df_teles_a1[df_teles_a1['limit']=='NAO_bot'])[0,1:][inds[r]], color = 'darkgray',
        edgecolor=np.array(df_teles_a1[df_teles_a1['limit']=='NAO_pv'])[0,1:][inds[r]], label = 'NAO',
        linewidth = np.array(df_teles_a1[df_teles_a1['limit']=='NAO_lw'])[0,1:][inds[r]])
 
        f3_ax1.bar(np.array(x), np.array(df_teles_a1[df_teles_a1['limit']=='PDOlag_up'])[0,1:][inds[r]], 
        bottom = np.array(df_teles_a1[df_teles_a1['limit']=='PDOlag_bot'])[0,1:][inds[r]], color = 'gold',
        edgecolor = np.array(df_teles_a1[df_teles_a1['limit']=='PDOlag_pv'])[0,1:][inds[r]],
        linewidth = np.array(df_teles_a1[df_teles_a1['limit']=='PDOlag_lw'])[0,1:][inds[r]])
        
        f3_ax1.bar(np.array(x), np.array(df_teles_a1[df_teles_a1['limit']=='NAOlag_up'])[0,1:][inds[r]], 
        bottom = np.array(df_teles_a1[df_teles_a1['limit']=='NAOlag_bot'])[0,1:][inds[r]], color = 'darkgray',
        edgecolor = np.array(df_teles_a1[df_teles_a1['limit']=='NAOlag_pv'])[0,1:][inds[r]],
        linewidth=np.array(df_teles_a1[df_teles_a1['limit']=='NAOlag_lw'])[0,1:][inds[r]] )
        
        f3_ax1.bar(np.array(x), np.array(df_teles_a1[df_teles_a1['limit']=='AMO_up'])[0,1:][inds[r]], 
        bottom = np.array(df_teles_a1[df_teles_a1['limit']=='AMO_bot'])[0,1:][inds[r]], color = 'darkgreen',
        label ='AMO', edgecolor=np.array(df_teles_a1[df_teles_a1['limit']=='AMO_pv'])[0,1:][inds[r]],
        linewidth = np.array(df_teles_a1[df_teles_a1['limit']=='AMO_lw'])[0,1:][inds[r]])
        

        handles, labels = f3_ax1.get_legend_handles_labels()
        
        ax1_lims_up = [1.4,1.4,1.4,1.4,1.4,1.4,1.4,1.4]
        
        ax1_lims_low = [-1.3,-1.3,-1.3,-1.3,-1.3,-1.3,-1.3,-1.3]
        
        if r in [0,4]:

            f3_ax1.set_xlim(-1.,2)
            
            f3_ax1.set_xticks([0,1])
            
            f3_ax1.set_xticklabels([ '$R_{+}$', '$R$'], fontsize = 7)
            
        elif r == 2:
            f3_ax1.set_xlim(-1.,3)
            
            f3_ax1.set_xticks([0,1,2])
            
            f3_ax1.set_xticklabels(['$R_{+}$', '$R$', '$R_{-}$'], fontsize = 7)
            
        elif r in [1,3]:
            f3_ax1.set_xlim(-1.,2)
            
            f3_ax1.set_xticks([0, 1])
            
            f3_ax1.set_xticklabels([ '$R$', '$R_{-}$'], fontsize = 7)
        
        elif r ==5:
            f3_ax1.set_xlim(-1,1)
            
            f3_ax1.set_xticks([0])
            
            f3_ax1.set_xticklabels([ '$R$'], fontsize = 7)
        else:
            f3_ax1.set_xlim(-1,1)
            
            f3_ax1.set_xticks([0])
            
            f3_ax1.set_xticklabels(['$R_{+}$'], fontsize = 7)
        
        ax1_ticks_up = [1,1,1,1,1,1,1,1]
        
        ax1_ticks_low = [-1,-1,-1,-1,-1,-1,-1,-1]
        
        ax1_labels_up = ['1','','','','','', '','']
        
        ax1_labels_low = ['-1','','','','','','','']
        
        
        f3_ax1.set_ylim(ax1_lims_low[r],ax1_lims_up[r])
        
        f3_ax1.set_yticks([ax1_ticks_low[r] , 0, ax1_ticks_up[r]])
        
        f3_ax1.set_yticklabels([ax1_labels_low[r], ax1_labels_up[r] ],fontsize =6)
        
        if r==0 :
            f3_ax1.set_yticklabels([ax1_labels_low[r],'0', ax1_labels_up[r] ],fontsize =6)
        

        f3_ax1.axhline(y=0,linewidth=0.6, color='k', linestyle = '-')

        
        if r ==0 :
            f3_ax1.set_ylabel('Contribution to D$_{1980}$ ($\\gamma$)',  fontsize = 7, labelpad=+1)

        f3_ax1.set_title(' '+ region_abs[regions[r]], position = (0.5,0.83), fontsize = 7)
            
        if r ==0:
        
            f3_ax1.text(-0.6, 0.98, 'b', transform=f3_ax1.transAxes, 
                size=11, weight='bold')
            
            f3_ax1.text(-0.6, 2.2, 'a', transform=f3_ax1.transAxes, 
                size=11, weight='bold')
       
        r+=1


r =0

for i in range(1):
    #r =0
    
    z =0
    for j in range(8):
        
        if r in one:
            f3_ax1 = fig3.add_subplot(gs[5*(i+2):5*(i+2)+4,z:z+1])
            z = z+1
        elif r in two:
            f3_ax1 = fig3.add_subplot(gs[5*(i+2):5*(i+2)+4,z:z+2])
            z = z+2
        else:
            f3_ax1 = fig3.add_subplot(gs[5*(i+2):5*(i+2)+4,z:z+3])
            z = z+3
        #f3_ax2 = fig3.add_subplot(gs[5*i+2:5*i+4,j*5:(j*5)+4])



        data_attr_reg = DATA_ATTR_Full[DATA_ATTR_Full['Region'] == regions[r]]
        data_attr_reg_pos = DATA_ATTR[DATA_ATTR['Region'] == regions[r]+'_pos']
        data_attr_reg_neg = DATA_ATTR[DATA_ATTR['Region'] == regions[r]+'_neg']


        tele = data_attr_reg[['gammaAbs_ENSO_gmt_run','gammaAbs_PDO_gmt_run','gammaAbs_NAO_gmt_run',
                              'gammaAbs_GMT_gmt_run','gammaAbs_ENSOlag_gmt_run',
                              'gammaAbs_PDOlag_gmt_run','gammaAbs_NAOlag_gmt_run']]
        tele_pos =  data_attr_reg_pos[['gammaAbs_ENSO_gmt_run','gammaAbs_PDO_gmt_run',
                                       'gammaAbs_NAO_gmt_run', 'gammaAbs_GMT_gmt_run',
                                       'gammaAbs_ENSOlag_gmt_run','gammaAbs_PDOlag_gmt_run',
                                       'gammaAbs_NAOlag_gmt_run']]
        tele_neg =  data_attr_reg_neg[['gammaAbs_ENSO_gmt_run','gammaAbs_PDO_gmt_run',
                                       'gammaAbs_NAO_gmt_run', 'gammaAbs_GMT_gmt_run',
                                       'gammaAbs_ENSOlag_gmt_run','gammaAbs_PDOlag_gmt_run',
                                       'gammaAbs_NAOlag_gmt_run']]
        
        
        tele_pv = data_attr_reg[['pval_ENSO_gmt_run','pval_PDO_gmt_run','pval_NAO_gmt_run',
                                 'pval_GMT_gmt_run','pval_ENSOlag_gmt_run','pval_PDOlag_gmt_run',
                                 'pval_NAOlag_gmt_run']]
        tele_pos_pv =  data_attr_reg_pos[['pval_ENSO_gmt_run','pval_PDO_gmt_run','pval_NAO_gmt_run',
                                          'pval_GMT_gmt_run','pval_ENSOlag_gmt_run',
                                          'pval_PDOlag_gmt_run','pval_NAOlag_gmt_run']]
        tele_neg_pv =  data_attr_reg_neg[['pval_ENSO_gmt_run','pval_PDO_gmt_run','pval_NAO_gmt_run',
                                          'pval_GMT_gmt_run','pval_ENSOlag_gmt_run',
                                          'pval_PDOlag_gmt_run','pval_NAOlag_gmt_run']]
        

        tele_pv = [tele_pos_pv, tele_pv, tele_neg_pv]
        
        tele_cons = [tele_pos, tele, tele_neg]
        
        df_teles=pd.DataFrame(columns= ['limit','tele_pos', 'tele', 'tele_neg'])
                              
        tele_names = ['tele_pos', 'tele', 'tele_neg']
        
        df_teles['limit'] = ['ENSO_up', 'ENSO_bot','ENSO_pv','ENSO_lw',\
                  'PDO_up', 'PDO_bot','PDO_pv', 'PDO_lw','NAO_up', 'NAO_bot', 'NAO_pv', 'NAO_lw',\
                  'GMT_up', 'GMT_bot','GMT_pv', 'GMT_lw',\
                  'ENSOlag_up', 'ENSOlag_bot','ENSOlag_pv', 'ENSOlag_lw',\
                  'PDOlag_up', 'PDOlag_bot','PDOlag_pv', 'PDOlag_lw', 'NAOlag_up',
                  'NAOlag_bot', 'NAOlag_pv', 'NAOlag_lw']
        
        preds = ['ENSO','PDO','NAO', 'GMT', 'ENSOlag','PDOlag','NAOlag']
        
        for t,tel in enumerate(tele_cons):
            
            abs_tel = np.nan_to_num(np.abs(tel))
            sum_tel = np.sum(np.array(abs_tel))

            perc = np.nan_to_num(tel/sum_tel)
            
            bottom = np.array(perc)[np.less(perc,0)].sum(axis =0)
            
            
            for m in np.argsort(perc)[0]:
                
                df_teles.loc[df_teles['limit']==preds[m]+'_bot', tele_names[t]] = bottom
                
                df_teles.loc[df_teles['limit']==preds[m]+'_up', tele_names[t]] = np.abs(perc[0,m])
            
            
                bottom += np.abs(perc[0,m])

        t_iter = 0
        
        shortage=['']
        
        shor = 0
        
        for t,tel_pv in enumerate(tele_pv):
            
            if t in [3,6,9]:
                shor+=1
            
            
            for p in preds:
                
                if tel_pv['pval_{}_gmt_run'.format(p)].values[0]<0.1:
                    lw = 1.5
                    alpha = 'k'
                else:
                    alpha = 'w'
                    lw = 0.
                    
                df_teles.loc[df_teles['limit']==p+'_pv', tele_names[t]] = alpha
                df_teles.loc[df_teles['limit']==p+'_lw', tele_names[t]] = lw
        

         

        inds= [[0,1],[1,2],[0,1,2],[1,2], [0,1], [1], [0],[0]]
        
        if r in three:
        
            x = [0,1,2]
        
        elif r in two:
        
            x = [0,1]
        
        else:
        
            x = [0]
        
        colour_code = ['#5ab4ac', '#4575b4', '#d8b365', '#5ab4ac', '#4575b4', '#d8b365']
        
        df_teles_a1 = df_teles.iloc[:,0:4]
        
        
        f3_ax1.bar(np.array(x), np.array(df_teles_a1[df_teles_a1['limit']=='NAOlag_up'])[0,1:][inds[r]], 
        bottom = np.array(df_teles_a1[df_teles_a1['limit']=='NAOlag_bot'])[0,1:][inds[r]], color = 'darkgray',
        edgecolor = np.array(df_teles_a1[df_teles_a1['limit']=='NAOlag_pv'])[0,1:][inds[r]],
        linewidth = np.array(df_teles_a1[df_teles_a1['limit']=='NAOlag_lw'])[0,1:][inds[r]])
        
        f3_ax1.bar(np.array(x), np.array(df_teles_a1[df_teles_a1['limit']=='ENSO_up'])[0,1:][inds[r]], 
        bottom = np.array(df_teles_a1[df_teles_a1['limit']=='ENSO_bot'])[0,1:][inds[r]], color = 'steelblue',
        edgecolor=np.array(df_teles_a1[df_teles_a1['limit']=='ENSO_pv'])[0,1:][inds[r]],
        linewidth = np.array(df_teles_a1[df_teles_a1['limit']=='ENSO_lw'])[0,1:][inds[r]])
        
        f3_ax1.bar(np.array(x), np.array(df_teles_a1[df_teles_a1['limit']=='ENSOlag_up'])[0,1:][inds[r]], 
        bottom = np.array(df_teles_a1[df_teles_a1['limit']=='ENSOlag_bot'])[0,1:][inds[r]], color = 'steelblue',
        edgecolor = np.array(df_teles_a1[df_teles_a1['limit']=='ENSOlag_pv'])[0,1:][inds[r]],
        linewidth = np.array(df_teles_a1[df_teles_a1['limit']=='ENSOlag_lw'])[0,1:][inds[r]])
        
        f3_ax1.bar(np.array(x), np.array(df_teles_a1[df_teles_a1['limit']=='NAO_up'])[0,1:][inds[r]], 
        bottom = np.array(df_teles_a1[df_teles_a1['limit']=='NAO_bot'])[0,1:][inds[r]], color = 'darkgray',
        edgecolor=np.array(df_teles_a1[df_teles_a1['limit']=='NAO_pv'])[0,1:][inds[r]], label = 'NAO', 
        linewidth = np.array(df_teles_a1[df_teles_a1['limit']=='NAO_lw'])[0,1:][inds[r]])
        
        
        
        f3_ax1.bar(np.array(x), np.array(df_teles_a1[df_teles_a1['limit']=='PDO_up'])[0,1:][inds[r]], 
        bottom = np.array(df_teles_a1[df_teles_a1['limit']=='PDO_bot'])[0,1:][inds[r]], color = 'gold', label = 'PDO',
        edgecolor=np.array(df_teles_a1[df_teles_a1['limit']=='PDO_pv'])[0,1:][inds[r]],
        linewidth = np.array(df_teles_a1[df_teles_a1['limit']=='PDO_lw'])[0,1:][inds[r]])
        
        
        
        f3_ax1.bar(np.array(x), np.array(df_teles_a1[df_teles_a1['limit']=='GMT_up'])[0,1:][inds[r]], 
        bottom = np.array(df_teles_a1[df_teles_a1['limit']=='GMT_bot'])[0,1:][inds[r]], color = 'sandybrown',
        label ='GMT', edgecolor=np.array(df_teles_a1[df_teles_a1['limit']=='GMT_pv'])[0,1:][inds[r]], 
        linewidth = np.array(df_teles_a1[df_teles_a1['limit']=='GMT_lw'])[0,1:][inds[r]])


        f3_ax1.bar(np.array(x), np.array(df_teles_a1[df_teles_a1['limit']=='PDOlag_up'])[0,1:][inds[r]], 
        bottom = np.array(df_teles_a1[df_teles_a1['limit']=='PDOlag_bot'])[0,1:][inds[r]], color = 'gold',
        edgecolor = np.array(df_teles_a1[df_teles_a1['limit']=='PDOlag_pv'])[0,1:][inds[r]],
        linewidth = np.array(df_teles_a1[df_teles_a1['limit']=='PDOlag_lw'])[0,1:][inds[r]])

        if j ==4:
            
            y_0 = np.array(df_teles_a1[df_teles_a1['limit']=='NAO_bot'])[0,1:][0]
            
            x_0_0 = 6/30
            
            x_0_1 = 14/30 
            
            f3_ax1.axhline(y_0, x_0_0, x_0_1,linewidth=1.5, color='k', linestyle = '-')
        
        handles, labels = f3_ax1.get_legend_handles_labels()
        
        ax1_lims_up = [1.4,1.4,1.4,1.4,1.4,1.4,1.4,1.4]
        
        ax1_lims_low = [-1.3,-1.3,-1.3,-1.3,-1.3,-1.3,-1.3,-1.3]
        
        if r in [0,4]:

            f3_ax1.set_xlim(-1.,2)
            
            f3_ax1.set_xticks([0,1])
            
            f3_ax1.set_xticklabels([ '$R_{+}$', '$R$'], fontsize = 7)
            
        elif r == 2:
            f3_ax1.set_xlim(-1.,3)
            
            f3_ax1.set_xticks([0,1,2])
            
            f3_ax1.set_xticklabels(['$R_{+}$', '$R$', '$R_{-}$'], fontsize = 7)
            
        elif r in [1,3]:
            f3_ax1.set_xlim(-1.,2)
            
            f3_ax1.set_xticks([0, 1])
            
            f3_ax1.set_xticklabels([ '$R$', '$R_{-}$'], fontsize = 7)
        
        elif r ==5:
            f3_ax1.set_xlim(-1,1)
            
            f3_ax1.set_xticks([0])
            
            f3_ax1.set_xticklabels([ '$R$'], fontsize = 7)
        else:
            f3_ax1.set_xlim(-1,1)
            
            f3_ax1.set_xticks([0])
            
            f3_ax1.set_xticklabels(['$R_{+}$'], fontsize = 7)
            
        
        ax1_ticks_up = [1,1,1,1,1,1,1,1]
        
        ax1_ticks_low = [-1,-1,-1,-1,-1,-1,-1,-1]
        
        ax1_labels_up = ['1','','','','','', '','']
        #plt.legend()
#plt.savefig('/home/insauer/projects/NC_Submission/Data/Figures/Mainfigures/Figure5.png',bbox_inches = 'tight',dpi =600)

        ax1_labels_low = ['-1','','','','','','','']
        

        f3_ax1.set_ylim(ax1_lims_low[r],ax1_lims_up[r])
        
        f3_ax1.set_yticks([ax1_ticks_low[r] , 0, ax1_ticks_up[r]])
        
        f3_ax1.set_yticklabels([ax1_labels_low[r], ax1_labels_up[r] ],fontsize =6)
        
        if r==0 :
            f3_ax1.set_yticklabels([ax1_labels_low[r],'0', ax1_labels_up[r] ],fontsize =6)
        

        f3_ax1.axhline(y=0,linewidth=0.6, color='k', linestyle = '-')

        if r ==0 :
            f3_ax1.set_ylabel('Contribution to D$_{1980}$ ($\\gamma$)',  fontsize = 7, labelpad=+1)

        f3_ax1.set_title(' '+ region_abs[regions[r]], position = (0.5,0.83), fontsize = 7)

        if r ==0:

            f3_ax1.text(-0.6, 0.98, 'c', transform=f3_ax1.transAxes, 
                size=11, weight='bold')
        r+=1



        
enso_box = mpatches.Rectangle((0, 0), 1, 1, facecolor='steelblue')
pdo_box = mpatches.Rectangle((0, 0), 1, 1, facecolor='gold')
nao_box = mpatches.Rectangle((0, 0), 1, 1, facecolor='darkgray')
gmt_box = mpatches.Rectangle((0, 0), 1, 1, facecolor= 'sandybrown')
amo_box = mpatches.Rectangle((0, 0), 1, 1, facecolor='darkgreen')
sig_box = mpatches.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor ='k')
#unex_box = mpatches.Rectangle((0, 0), 1, 1, facecolor='white', hatch = '////')

labels=['ENSO','PDO','NAO','GMT','AMO','Significant predictor']
handles = [enso_box,pdo_box, nao_box,gmt_box, amo_box, sig_box]

f3_ax2 = fig3.add_subplot(gs[0:3,z-14:z])
          
f3_ax2.legend(handles,labels, frameon=True, fontsize = 7, loc = (0.72,-0.1), edgecolor = 'k')
f3_ax2.axis('off')


       

f3_ax3 = fig3.add_subplot(gs[0:4,0:9])
years = np.arange(1951,2011)
f3_ax3.set_xlim(1950,2010)   
f3_ax3.plot(years,teleDataNorm['AMO'], label='AMO', color = 'darkgreen')
f3_ax3.plot(years,teleDataNorm['ENSO']+4,  color = 'cornflowerblue')
f3_ax3.plot(years,teleDataNorm['PDO']+3, color = 'gold')
f3_ax3.plot(years,teleDataNorm['NAO']+2, color = 'darkgray')
f3_ax3.plot(years,teleDataNorm['GMT']+1, color =  'sandybrown')

f3_ax3.set_yticks([0,1,2,3,4,5])
f3_ax3.set_yticklabels(['','','','','',''])
        

f3_ax3.axhline(y=1,linewidth=0.3, color='k', linestyle = '-', alpha = 0.8)
f3_ax3.axhline(y=2,linewidth=0.3, color='k', linestyle = '-', alpha = 0.8)
f3_ax3.axhline(y=3,linewidth=0.3, color='k', linestyle = '-', alpha = 0.8)
f3_ax3.axhline(y=4,linewidth=0.3, color='k', linestyle = '-', alpha = 0.8)
f3_ax3.axvline(x=1971,linewidth=0.3, color='k', linestyle = '-', alpha = 0.8)
f3_ax3.axvspan(1971,2010, facecolor='gainsboro')
f3_ax3.set_ylabel('Normalized indices',  fontsize = 7, labelpad=+1)
f3_ax3.set_xlabel('Year',  fontsize = 7, labelpad=+2)

f3_ax3.tick_params(axis='both', labelsize=7)

plt.savefig('../../data/figures/Figure5.pdf',bbox_inches = 'tight', format = 'pdf')
