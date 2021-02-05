#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 22:35:58 2020

@author: insauer
"""

#!/usr/bin/env python
import numpy as np
import pandas as pd
import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import argparse
from climada.entity.exposures.gdp_asset import GDP2Asset
from climada.entity.impact_funcs.river_flood import flood_imp_func_set
from climada.hazard.river_flood import RiverFlood
from climada.util.constants import RIVER_FLOOD_REGIONS_CSV
import copy
from climada.engine import Impact


"""This script allows to reproduce the damage time series with the impact model CLIMADA
   for one combination of climate forcing and GHM. For each of the model combination (46)
   this script must be executed. Here, we call the script for each model combination
   by passing a forcing dataset and a GHM name with, schedule_run.py
   The output is a CSV file containig damage for each country in R, R+ and R- areas from
   1971-2010. Additionally, flooded area and volume is calculated.
"""
parser = argparse.ArgumentParser(
    description='run climada for different climate and runoff models')
parser.add_argument(
    '--RF_model', type=str, default='H08',
    help='runoff model')
parser.add_argument(
    '--CL_model', type=str, default='princeton',
    help='Climate forcing')

args = parser.parse_args()

# we only calculate damages for 'flopros' protection
PROT_STD = ['flopros']

# insert gridded gdp file here  https://doi.org/10.5880/pik.2017.007 'GCP_PPP-2005_1850-2100.nc'
# file needs to be interpolated to yearly data and upscaled to 2.5 arcmin

gdp_path = '../../../data/downloads/gridded_gdp.nc'

# this file is available on github
RF_PATH_FRC = '../../../data/hazard_settings/fldfrc_2yr_return_level.nc'


output = currentdir


# please provide the path for the discharge trend map here
dis_path = '../../../data/hazard_settings/basin_trends.nc'

# please set the directory containing the ISIMIP flood data here
flood_dir = '../../../data/downloads/isimip_flood/'

# climate forcing watch contains only data until 2001
if args.CL_model == 'watch':
    years = np.arange(1971, 2002)
else:
    years = np.arange(1971, 2011)


country_info = pd.read_csv(RIVER_FLOOD_REGIONS_CSV)
isos = country_info['ISO'].tolist()


cont_list = country_info['if_RF'].tolist()
l = (len(years) * (len(isos)-2))
continent_names = ['Africa', 'Asia', 'Europe', 'NorthAmerica', 'Oceania', 'SouthAmerica']


dataDF = pd.DataFrame(data={'Year': np.full(l, np.nan, dtype=int),
                            'Country': np.full(l, "", dtype=str),
                            'Region': np.full(l, "", dtype=str),
                            'D_CliExp_pos_raw': np.full(l, np.nan, dtype=float),
                            'D_CliExp_neg_raw': np.full(l, np.nan, dtype=float),
                            'D_CliExp_raw': np.full(l, np.nan, dtype=float),
                            'D_1980_pos_raw': np.full(l, np.nan, dtype=float),
                            'D_1980_neg_raw': np.full(l, np.nan, dtype=float),
                            'D_1980_raw': np.full(l, np.nan, dtype=float),
                            'D_2010_pos_raw': np.full(l, np.nan, dtype=float),
                            'D_2010_neg_raw': np.full(l, np.nan, dtype=float),
                            'D_2010_raw': np.full(l, np.nan, dtype=float)
                            })
# set JRC impact functions
if_set = flood_imp_func_set()

fail_lc = 0
line_counter = 0

# loop over all countries
for cnt_ind in range(len(isos)):
    country = [isos[cnt_ind]]

    if country[0] in ['GIB', 'MCO']:
        continue
    reg = country_info.loc[country_info['ISO'] == country[0], 'Reg_name'].values[0]

    # setting fixed exposures
    gdpa1980 = GDP2Asset()
    gdpa1980.set_countries(countries=country, ref_year=1980, path=gdp_path)
    gdpa2010 = GDP2Asset()
    gdpa2010.set_countries(countries=country, ref_year=2010, path=gdp_path)

    save_lc = line_counter
    # loop over protection standards
    for pro_std in range(len(PROT_STD)):
        line_counter = save_lc
        #  insert the files on flddph and fldfrc from http://doi.org/10.5281/zenodo.4446364
        dph_path = flood_dir + 'flddph_150arcsec_{}_{}_{}.nc'\
            .format(args.RF_model, args.CL_model, PROT_STD[pro_std])
        frc_path = flood_dir + 'fldfrc_150arcsec_{}_{}_{}.nc'\
            .format(args.RF_model, args.CL_model, PROT_STD[pro_std])

        if not os.path.exists(dph_path):
            print('{} path not found'.format(dph_path))
            break
        if not os.path.exists(frc_path):
            print('{} path not found'.format(frc_path))
            break

        # set flood hazard
        rf = RiverFlood()

        rf.set_from_nc(dph_path=dph_path, frc_path=frc_path,
                       countries=country, years=years, ISINatIDGrid=True)

        # set flood hazard with the 2yr return period area substraction

        rf2y = copy.copy(rf)

        rf2y.exclude_returnlevel(RF_PATH_FRC)

        # set flood hazard for subregions
        rf2y_pos = copy.copy(rf2y)
        rf2y_pos.exclude_trends(dis_path, 'pos')
        rf2y_neg = copy.copy(rf2y)
        rf2y_neg.exclude_trends(dis_path, 'neg')
        # loop over all years
        for year in range(len(years)):
            print('country_{}_year{}_protStd_{}'.format(country[0], str(years[year]), PROT_STD[pro_std]))
            ini_date = str(years[year]) + '-01-01'
            fin_date = str(years[year]) + '-12-31'
            dataDF.iloc[line_counter, 0] = years[year]
            dataDF.iloc[line_counter, 1] = country[0]
            dataDF.iloc[line_counter, 2] = reg
            # set variable exposure
            gdpa = GDP2Asset()
            gdpa.set_countries(countries=country, ref_year=years[year], path=gdp_path)
            # calculate damages for all combinations
            imp2y_fl_pos = Impact()
            imp2y_fl_pos.calc(gdpa, if_set, rf2y_pos.select(date=(ini_date, fin_date)))
            imp2y_fl_neg = Impact()
            imp2y_fl_neg.calc(gdpa, if_set, rf2y_neg.select(date=(ini_date, fin_date)))
            imp2y_fl = Impact()
            imp2y_fl.calc(gdpa, if_set, rf2y.select(date=(ini_date, fin_date)))

            imp2y_fl_1980_pos = Impact()
            imp2y_fl_1980_pos.calc(gdpa1980, if_set, rf2y_pos.select(date=(ini_date, fin_date)))
            imp2y_fl_1980_neg = Impact()
            imp2y_fl_1980_neg.calc(gdpa1980, if_set, rf2y_neg.select(date=(ini_date, fin_date)))
            imp2y_fl_1980 = Impact()
            imp2y_fl_1980.calc(gdpa1980, if_set, rf2y.select(date=(ini_date, fin_date)))

            imp2y_fl_2010_pos = Impact()
            imp2y_fl_2010_pos.calc(gdpa2010, if_set, rf2y_pos.select(date=(ini_date, fin_date)))
            imp2y_fl_2010_neg = Impact()
            imp2y_fl_2010_neg.calc(gdpa2010, if_set, rf2y_neg.select(date=(ini_date, fin_date)))
            imp2y_fl_2010 = Impact()
            imp2y_fl_2010.calc(gdpa2010, if_set, rf2y.select(date=(ini_date, fin_date)))

            # write dataframe (needs to be adjusted if more prot standards are considered)
            dataDF.iloc[line_counter, 3] = imp2y_fl_pos.at_event[0]
            dataDF.iloc[line_counter, 4] = imp2y_fl_neg.at_event[0]
            dataDF.iloc[line_counter, 5] = imp2y_fl.at_event[0]

            dataDF.iloc[line_counter, 6] = imp2y_fl_1980_pos.at_event[0]
            dataDF.iloc[line_counter, 7] = imp2y_fl_1980_neg.at_event[0]
            dataDF.iloc[line_counter, 8] = imp2y_fl_1980.at_event[0]

            dataDF.iloc[line_counter, 9] = imp2y_fl_2010_pos.at_event[0]
            dataDF.iloc[line_counter, 10] = imp2y_fl_2010_neg.at_event[0]
            dataDF.iloc[line_counter, 11] = imp2y_fl_2010.at_event[0]

            line_counter += 1

    # save output dataframe
    dataDF.to_csv('../../../data/damage_reco/damage_{}_{}.csv'.format(args.RF_model, args.CL_model))

