# -*- coding: utf-8 -*-
"""
Spyder Editor

This file aggregates multi-model flood damage output on country level 
to regional model medians. Additionally recorded damage is added.
"""
import numpy as np
import pandas as pd
import os


def aggregation_regions(x):
    """
    This function aggregates country-level damages and variables to
    regional level.
    Parameters
    ----------
    x : DataFrame
        country-level damages and other indicators for all model combinations

    Returns
    -------
    DataFrame
        regionally aggregated damages and other indicators
    """
    # Aggreagted damages is given as the sum over country-wise damages
    aggregated_model_damages = x['D_CliExp_raw'].sum()
    aggregated_model_damages_1980flospros = x['D_1980_raw'].sum()
    aggregated_model_damages_2010flospros = x['D_2010_raw'].sum()
    aggregated_observed_damages = (x['natcat_flood_damages_2005_CPI']).sum()

    return pd.Series([aggregated_model_damages,
                      aggregated_model_damages_1980flospros,
                      aggregated_model_damages_2010flospros,
                      aggregated_observed_damages],
                     index=['D_CliExp_raw',
                            'D_1980_raw',
                            'D_2010_raw',
                            'natcat_flood_damages_2005_CPI'])

def func_median(x):
    """
    This function aggregates the damages and other indicators from the
    different model runs to the model median and adds basic statistics such as
    the one-third and two-third quantiles.

    Parameters
    ----------
    x : DataFrame
        regionally aggregated damages and other indicators for all model
        combinations

    Returns
    -------
    DataFrame
         model medians of regionally aggregated damages and other indicators
    """
    # identify the median of the model data:
    median_model_damages = x['D_CliExp_raw'].median()  # =quantile(0.5)
    median_model_damages_1980flospros = x['D_1980_raw'].median()
    # =quantile(0.5)
    median_model_damages_2010flospros = x['D_2010_raw'].median()
    # =quantile(0.5)
    
    # the same for all
    median_observed_damages = (x['natcat_flood_damages_2005_CPI']).mean()

    one_third_quantile_model_damages = x['D_CliExp_raw'].quantile(0.3)
    # 30
    two_third_quantile_model_damages = x['D_CliExp_raw'].quantile(0.7)  # 70

    return pd.Series([median_model_damages,
                      median_model_damages_1980flospros,
                      median_model_damages_2010flospros,
                      median_observed_damages,
                      one_third_quantile_model_damages,
                      two_third_quantile_model_damages],
                     index=['D_CliExp_raw',
                            'D_1980_raw',
                            'D_2010_raw',
                            'natcat_flood_damages_2005_CPI',
                            'D_CliExp_raw_onethird_quantile', 
                            'D_CliExp_raw_twothird_quantile'])


def add_GDP_NatCat(megaDataFrame, years, gdp_resc):
    """
    This function inserts national annual variables in the MegaDataFrame
    containing all the data. Damages are all converted to capital stock by
    applying a corresponding annual national conversion factor.
    Inserted are:
        recorded damages (NatCat Munich Re)

    Parameters
    ----------
    megaDataFrame : DataFrame
        big data set containing all the data from all model runs
    years : int array
        years to be considered
    gdp_resc : bool
        gdp to capital stock conversion

    Returns
    -------
    DataFrame
         model medians of regionally aggregated damages and other indicators
    """

    # provide dataset with observational data these need to be requested from Munich Re
    # the DataSet needs to be treated with the adjustment for 2005 PPP as done in 
    # /code/scripts_reconstruction/natcat_damages/flood_damage_conversion.ipynb
    natcat = pd.read_excel('../../../data/downloads/1980-2016_Masterdatei_NatCat_worldwide_no-pw_2005conversions_PPP.xlsx', index_col=0)
    countries = pd.read_csv('../../../data/supporting_data/final_country_list.csv')
    
    # asset rescaling to correct for the ssp transition and to convert GDP to capital stock
    # datasets can be generated with a separate code discribed in the readme.txt
    
    # here we need the files in /data/exposure_rescaling
    resc_factors = pd.read_csv('../../../data/exposure_rescaling/resc_ssp_transition.csv')

    cap_factors = pd.read_csv('../../../data/exposure_rescaling/totalwealth_capital_stock_rescaling.csv')

    countries = list(set(megaDataFrame['Country']).intersection(countries.iloc[:, 0]))
    natcat_short = natcat.iloc[:,[4,7,8,9,13,12,11,21,20,25,29,33,-17,-3,-2,-1]] # selection of NatCat columns, containing all rows

    cols = ['year', 'event', 'type', 'subtype', 'ISO', 'country', 'Continent',
            'tot_loss', 'ins_loss', 'tot_loss_GDP', 'tot_loss_GCP',
            'tot_loss_CPI', 'Fatalities', 'CPI_con', 'GDP_con', 'Reg_name']
    natcat_short.columns = cols
    natcat_hydro = natcat_short.loc[natcat_short['subtype'] == 'gf:General flood']
    megaDataFrame['natcat_flood_damages_2005_CPI'] = np.nan


    for country in countries:
        
        resc_fac_cnt_yr_1980 = resc_factors.loc[resc_factors['ISO'] == country, str(1980)].sum()
        capst_fac_cnt_yr_1980 = cap_factors.loc[cap_factors['ISO'] == country, str(1980)].sum()
        resc_fac_cnt_yr_2010 = resc_factors.loc[resc_factors['ISO'] == country, str(2010)].sum()
        capst_fac_cnt_yr_2010 = cap_factors.loc[cap_factors['ISO'] == country, str(2010)].sum()


        for year in years:
            print(str(year) + ' ' + country)
            resc_fac_cnt_yr = resc_factors.loc[resc_factors['ISO'] == country, str(year)].sum()
            capst_fac_cnt_yr = cap_factors.loc[cap_factors['ISO'] == country, str(year)].sum()
            
            # this lines need to be added if damage generation was done with CLIMADA, if damages 
            # from region_damages_modelmedian_R.csv from the Supplementary Data is used
            if gdp_resc:
                megaDataFrame.loc[(megaDataFrame['Country'] == country) &
                                  (megaDataFrame['Year'] == year),
                                  'D_CliExp_raw'] *= resc_fac_cnt_yr * capst_fac_cnt_yr
    
                megaDataFrame.loc[(megaDataFrame['Country'] == country) &
                                  (megaDataFrame['Year'] == year),
                                  'D_1980_raw'] *= resc_fac_cnt_yr_1980 * capst_fac_cnt_yr_1980
    
                megaDataFrame.loc[(megaDataFrame['Country'] == country) &
                                  (megaDataFrame['Year'] == year),
                                  'D_2010_raw'] *= resc_fac_cnt_yr_2010 * capst_fac_cnt_yr_2010

            if year > 1979:
                tmp_natcat_05_cpi = natcat_hydro.loc[(natcat_hydro['year']==year) & (natcat_hydro['ISO']==country), 'CPI_con'].sum()

                megaDataFrame.loc[(megaDataFrame['Country'] == country) &
                                  (megaDataFrame['Year'] == year),
                                  'natcat_flood_damages_2005_CPI'] = tmp_natcat_05_cpi*1.0e6

            else:
                megaDataFrame.loc[(megaDataFrame['Country'] == country) &
                                  (megaDataFrame['Year'] == year),
                                  'natcat_flood_damages_2005_CPI'] = np.nan

    return megaDataFrame


def region_aggregation(cols, dataFrame):
    """
    This function is a wrapper for the aggregation and selects the columns to
    be aggregated to regional level.

    Parameters
    ----------
    out_cols : string list
        Columns to be aggregated

    Returns
    -------
    DataFrame
         regionally aggregated damages and other indicators regions
    """
    data_region = dataFrame.groupby(['Year', 'GHM', 'clim_forc', 'Region'])\
                                    [cols].apply(aggregation_regions)\
                                    .reset_index()  # groupby year and model

    return data_region


def model_aggregation(out_cols, dataFrame, years, select_model):
    """
    This function is a wrapper for the multi-model aggregation and provides
    the model median for each region of all variables.

    Parameters
    ----------
    out_cols : string list
        Columns to be aggregated

    Returns
    -------
    DataFrame
         regionally aggregated model medians
    """

    if select_model:

        dataFrame = dataFrame[dataFrame['GHM'] == select_model]

    data_models = dataFrame[(dataFrame['Year'] <= np.max(years)) &
                            (dataFrame['Year'] >= np.min(years))]
    # Get the median for model and datasets
    data_models = data_models.groupby(['Year', 'Region'])\
                              [out_cols].apply(func_median).reset_index()

    return data_models


def assemble_data_frame(path, sort, years):
    """
    This function gathers all the data from all model_runs and
    provides one big data set containing all model runs

    Parameters
    ----------
    path : string
        Path to directory on cluster where all runs are stored
    sort : string list
        columns used for sorting
    years : int array

    Returns
    -------
    DataFrame
         full data
    """
    megaDataFrame = pd.DataFrame()
    list_of_model_output = os.listdir(path)

    for i, model_output in enumerate(list_of_model_output):  # loop over all model output 

        [n1, n2, n3, n4, ghm, clim_forc, n7, n8, n9] = model_output.split('_') 
#        if cl_model == 'watch':
#            continue

        print('Loading ' + model_output)
        temp = pd.read_csv(path+model_output)
        temp['GHM'] = ghm
        temp['clim_forc'] = clim_forc
        temp = temp[temp['Year'] >= 1971]
        megaDataFrame = megaDataFrame.append(temp, ignore_index=True)

    megaDataFrame = megaDataFrame.sort_values(by=sort)

    return megaDataFrame


def aggregate_new_region(dataFrame):
    """
    This function combines world regions to bigger regions

    Parameters
    ----------
    dataFrame : DataFrame
        DataFrame containig smaller regions

    Returns
    -------
    DataFrame
         DataFrame with manipulated regions
    """

    dataFrame.loc[dataFrame['Country'] == 'RUS',
                  'Region'] = 'CAS'
    
    dataFrame.loc[dataFrame['Region'] == 'CHN',
                  'Region'] = 'EAS'

    dataFrame.loc[(dataFrame['Region'] == 'CAR') |
                  (dataFrame['Region'] == 'LAS') |
                  (dataFrame['Region'] == 'LAN'),
                  'Region'] = 'LAM'

    dataFrame.loc[(dataFrame['Region'] == 'NAF') |
                  (dataFrame['Region'] == 'ARA'), 'Region'] = 'NAF'

    dataFrame.loc[(dataFrame['Region'] == 'SSA') |
                  (dataFrame['Region'] == 'SAF'), 'Region'] = 'SSA'

    dataFrame.loc[(dataFrame['Region'] == 'EUR') |
                  (dataFrame['Region'] == 'EUA'), 'Region'] = 'EUR'

    dataFrame.loc[(dataFrame['Region'] == 'SWA') |
                  (dataFrame['Region'] == 'SEA'),
                  'Region'] = 'SEA'

    dataFrame.loc[(dataFrame['Region'] == 'PIS1') |
                  (dataFrame['Region'] == 'PIS2') |
                  (dataFrame['Region'] == 'AUS'),
                  'Region'] = 'OCE'

    return dataFrame


def get_rm(col_names, dataFrame, rm_window):
    """
    This function calculates running means for selected columns.
    Not applied in the paper!

    Parameters
    ----------
    col_names : string list
        selected columns
    dataFrame : DataFrame
        full dataset
    rm_window : int
         window size

    Returns
    -------
    DataFrame
         DataFrame containing the running means for selected columns
    """
    for col in col_names:
        dataFrame[col] = dataFrame[col].replace(
                                            [-np.inf, np.inf],
                                            [np.nan, np.nan])
        dataFrame[col] = runmean(dataFrame[col], rm_window)

    return dataFrame


sort = ['Year', 'Country']
years = np.arange(1971, 2012)

""" If damages where previously calculated with CLIMADA please uncomment the following instructions.
"""

#  Path where data from damage assessment calculated with CLIMADA is stored (output) from all models

#path = '../../../data/damage_reco/'

#  Building one big data set with all the data from all model runs

#assDataFrame = assemble_data_frame(path, sort, years)

#  New regional aggregation (change region name)

#assDataFrame = aggregate_new_region(assDataFrame)

""" If country level damages are used from country_damages_multimodel_R.csv from the
    Supplementary Data, start the script here and add the correct path.
"""
assDataFrame = pd.read_csv('../../../data/supplementary_data/country_damages_multimodel_R.csv')

# Columns used for region aggregation
in_cols = ['D_CliExp_raw',  # D_CliExp_raw
           'D_1980_raw',  # D_1980_raw
           'D_2010_raw',  # D_2010_raw
           'natcat_flood_damages_2005_CPI']

""" If damages where previously calculated with CLIMADA set gdp_resc = False, if supplementary data
    is used set gdp_resc = True.
"""
assDataFrame = add_GDP_NatCat(assDataFrame, years, gdp_resc = False)

#  aggregate all the country based data to regions
regDataFrame = region_aggregation(in_cols, assDataFrame)

#  building model median
modDataFrame = model_aggregation(in_cols, regDataFrame, years, None)

modDataFrame.to_csv('../../../data/reconstruction/model_median_regions.csv', index=False)
