#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:32:07 2020

@author: insauer
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pyts.decomposition import SingularSpectrumAnalysis
import pymannkendall as mk
from scipy import stats


def rel_time_attr_MK(dataFrame71, disch):
    """
    Theil-Sen-Slope estimation and Mann-Kendall-Test to estimate the
    contribution of each driver!

    Parameters
    ----------
    dataFrame71 : time series
        Time series.
    disch : string
        discharge group

    Returns
    -------
    regH : List MK-output
        Sen_slope and MK-test result with uncertainty range of hazard
        (with 1980 fixed exposure)(TS_Haz) 1980-2010
    regHE : List MK-output
        Sen_slope and MK-test result with uncertainty range of TS_HazExp
        1980-2010
    regF : List MK-output
        Sen_slope and MK-test result with uncertainty range of TS_Full
        1980-2010.
    regH7 : List MK-output
        Sen_slope and MK-test result with uncertainty range of hazard
        (with 1980 fixed exposure)(TS_Haz) 1971-2010
    regH107 : List MK-output
        Sen_slope and MK-test result with uncertainty range of hazard
        (with 2010 fixed exposure)(TS_Haz) 1971-2010
    regH10 : List MK-output
        Sen_slope and MK-test result with uncertainty range of hazard
        (with 2010 fixed exposure)(TS_Haz) 1980-2010
    regE : List MK-output
        Sen_slope and MK-test result with uncertainty range of exposure
        difference function (TS_HazExp - TS_Haz) 1980-2010 (not used)
    regE7 : List MK-output
        Sen_slope and MK-test result with uncertainty range of exposure
        difference function (TS_HazExp - TS_Haz) 1971-2010 (not used)
    regV : List MK-output
        Sen_slope and MK-test result with uncertainty range of vulnerability
        difference function (TS_full - TS_Haz_Exp)(not used)
    regI : List MK-output
        Sen_slope and MK-test result with uncertainty range of modeled damges
        (including vulnerability)
    regN : List MK-output
        Sen_slope and MK-test result with uncertainty range of observed damages

    """

    dataFrame = dataFrame71[dataFrame71['Year'] > 1979]

    regLHazExp = mk.original_test(dataFrame['D_CliExp_norm_for_trend_{}'.format(disch)],
                                  alpha=0.1)
    slopeLHazExp = stats.theilslopes(dataFrame['D_CliExp_norm_for_trend_{}'.format(disch)],
                                     alpha=0.1)

    regHE = [regLHazExp.slope, regLHazExp.p, slopeLHazExp[2], slopeLHazExp[3]]

    regLFull = mk.original_test(dataFrame['D_Full_{}'.format(disch)],
                                alpha=0.1)

    slopeLFull = stats.theilslopes(dataFrame['D_Full_{}'.format(disch)],
                                   alpha=0.1)

    regF = [regLFull.slope, regLFull.p, slopeLFull[2], slopeLFull[3]]

    regHaz = mk.original_test(dataFrame['D_1980_norm_for_trend_{}'.format(disch)],
                              alpha=0.1)

    slopeHaz = stats.theilslopes(dataFrame['D_1980_norm_for_trend_{}'.format(disch)],
                                 alpha=0.1)

    regH = [regHaz.slope, regHaz.p, slopeHaz[2], slopeHaz[3]]

    regHaz7 = mk.original_test(dataFrame71['D_1980_norm_for_trend_{}'.format(disch)], alpha=0.1)

    slopeHaz7 = stats.theilslopes(dataFrame71['D_1980_norm_for_trend_{}'.format(disch)],
                                  alpha=0.1)

    regH7 = [regHaz7.slope, regHaz7.p, slopeHaz7[2], slopeHaz7[3]]

    regHaz107 = mk.original_test(dataFrame71['D_2010_norm_for_trend_{}'.format(disch)],
                                 alpha=0.1)

    slopeHaz107 = stats.theilslopes(dataFrame71['D_2010_norm_for_trend_{}'.format(disch)],
                                    alpha=0.1)

    regH107 = [regHaz107.slope, regHaz107.p, slopeHaz107[2], slopeHaz107[3]]

    regHaz10 = mk.original_test(dataFrame['D_2010_norm_for_trend_{}'.format(disch)],
                                alpha=0.1)

    slopeHaz10 = stats.theilslopes(dataFrame['D_2010_norm_for_trend_{}'.format(disch)],
                                   alpha=0.1)

    regH10 = [regHaz10.slope, regHaz10.p, slopeHaz10[2], slopeHaz10[3]]

    regE = mk.original_test(dataFrame['D_CliExp_norm_for_trend_{}'.format(disch)], alpha=0.1)
    regE7 = mk.original_test(dataFrame['D_CliExp_norm_for_trend_{}'.format(disch)], alpha=0.1)

    regV = mk.original_test(dataFrame['D_Full_{}'.format(disch)], alpha=0.1)

    regI = mk.original_test(dataFrame['D_Full_{}'.format(disch)],
                            alpha=0.1)
    regI = regF

    regNat = mk.original_test(dataFrame['natcat_damages_2005_CPI_{}'.format(disch)],
                              alpha=0.1)

    slopeNat = stats.theilslopes(dataFrame['natcat_damages_2005_CPI_{}'.format(disch)],
                                 alpha=0.1)

    regN = [slopeNat[0], regNat.p, slopeNat[2], slopeNat[3]]

    return regH, regHE, regF, regH7, regH107, regH10, regE, regE7, regV, regI, regN


def normalise(region, dataFrame):
    """
    Normalisation of time series. Normalisation for trends estimation
    normalises everything to total observed damage. Normalisation for plotting
    shifts time series to the same starting point in 1980.
    ----------
    region : string
        Abbrevation of region
    dataFrame : DataFrame
        Time series

    Returns
    -------
    dataFrame : DataFrame
        Time series + normalised time series

    """

    obs_adj_neg = dataFrame.loc[dataFrame['Year'] > 1979, 'natcat_damages_2005_CPI_neg'].mean() / \
        dataFrame.loc[dataFrame['Year'] > 1979, 'D_Full_neg_raw'].mean()
    obs_adj_pos = dataFrame.loc[dataFrame['Year'] > 1979, 'natcat_damages_2005_CPI_pos'].mean() / \
        dataFrame.loc[dataFrame['Year'] > 1979, 'D_Full_pos_raw'].mean()

    dataFrame['D_Full_neg'] = dataFrame['D_Full_neg_raw'] * obs_adj_neg
    dataFrame['D_Full_pos'] = dataFrame['D_Full_pos_raw'] * obs_adj_pos

    # normalisation for plotting
    offsetExpNeg = dataFrame.loc[dataFrame['Year'] == 1980, 'D_Full_neg'].sum() / \
        dataFrame.loc[dataFrame['Year'] == 1980, 'D_CliExp_neg_raw'].sum()
    dataFrame['D_CliExp_neg'] = dataFrame['D_CliExp_neg_raw'] * offsetExpNeg

    offsetExpPos = dataFrame.loc[dataFrame['Year'] == 1980, 'D_Full_pos'].sum() / \
        dataFrame.loc[dataFrame['Year'] == 1980, 'D_CliExp_pos_raw'].sum()
    dataFrame['D_CliExp_pos'] = dataFrame['D_CliExp_pos_raw'] * offsetExpPos

    offsetHazNeg = dataFrame.loc[dataFrame['Year'] == 1980,
                                 'D_Full_neg'].sum() / \
        dataFrame.loc[dataFrame['Year'] == 1980, 'D_1980_neg_raw'].sum()
    dataFrame['D_1980_neg'] = dataFrame['D_1980_neg_raw'] * offsetHazNeg

    offsetHazPos = dataFrame.loc[dataFrame['Year'] == 1980, 'D_Full_pos'].sum() / \
        dataFrame.loc[dataFrame['Year'] == 1980, 'D_1980_pos_raw'].sum()
    dataFrame['D_1980_pos'] = dataFrame['D_1980_pos_raw'] * offsetHazPos

    dataFrame['D_2010_pos'] = dataFrame['D_2010_pos_raw']*offsetExpPos
    dataFrame['D_2010_neg'] = dataFrame['D_2010_neg_raw']*offsetExpNeg

    # modelspread for plotting

    dataFrame['D_Full_1thrd_neg'] = dataFrame['D_Full_1thrd_neg_raw'] * obs_adj_neg

    dataFrame['D_Full_1thrd_pos'] = dataFrame['D_Full_1thrd_pos_raw'] * obs_adj_pos

    dataFrame['D_Full_2thrd_neg'] = dataFrame['D_Full_2thrd_neg_raw'] * obs_adj_neg

    dataFrame['D_Full_2thrd_pos'] = dataFrame['D_Full_2thrd_pos_raw'] * obs_adj_pos



    # normalisation for trend estimation

    trendNormExpNeg = dataFrame.loc[dataFrame['Year'] >= 1980, 'D_Full_neg'].mean() / \
        dataFrame.loc[dataFrame['Year'] >= 1980, 'D_CliExp_neg_raw'].mean()
    dataFrame['D_CliExp_norm_for_trend_neg'] = dataFrame['D_CliExp_neg_raw'] * trendNormExpNeg

    trendNormExpPos = dataFrame.loc[dataFrame['Year'] >= 1980, 'D_Full_pos'].mean() / \
        dataFrame.loc[dataFrame['Year'] >= 1980, 'D_CliExp_pos_raw'].mean()
    dataFrame['D_CliExp_norm_for_trend_pos'] = dataFrame['D_CliExp_pos_raw'] * trendNormExpPos

    trendNormHazNeg = dataFrame.loc[dataFrame['Year'] >= 1980, 'D_Full_neg'].mean() / \
        dataFrame.loc[dataFrame['Year'] >= 1980, 'D_1980_neg_raw'].mean()
    dataFrame['D_1980_norm_for_trend_neg'] = dataFrame['D_1980_neg_raw'] * trendNormHazNeg

    trendNormHazPos = dataFrame.loc[dataFrame['Year'] >= 1980, 'D_Full_pos'].mean() / \
        dataFrame.loc[dataFrame['Year'] >= 1980, 'D_1980_pos_raw'].mean()

    dataFrame['D_1980_norm_for_trend_pos'] = dataFrame['D_1980_pos_raw'] * trendNormHazPos

    trendNormHaz10Neg = dataFrame.loc[dataFrame['Year'] >= 1980, 'D_Full_neg'].mean() / \
        dataFrame.loc[dataFrame['Year'] >= 1980, 'D_2010_neg_raw'].mean()
    dataFrame['D_2010_norm_for_trend_neg'] = dataFrame['D_2010_neg_raw'] * trendNormHaz10Neg

    trendNormHaz10Pos = dataFrame.loc[dataFrame['Year'] >= 1980, 'D_Full_pos'].mean() / \
        dataFrame.loc[dataFrame['Year'] >= 1980, 'D_2010_pos_raw'].mean()
    dataFrame['D_2010_norm_for_trend_pos'] = dataFrame['D_2010_pos_raw'] * trendNormHaz10Pos

    return dataFrame


def prep_table_timeMK(region, data, regLHaz, regLHazExp, regLFull, regH7,
                      regH107, regH10, regE, regE7, regV, regI, regN, dis):
    """
    Prepare output table for attribution done with Mann-Kendall and Theil-Sen slope

    Parameters
    ----------
    region : string
        Abbrevation of region
    dat : DataFrame
        Time series
    regLHaz : List MK-output
        Sen_slope and MK-test result with uncertainty range of hazard
        (with 1980 fixed exposure)(TS_Haz) 1980-2010
    regLHazExp :  List MK-output
        Sen_slope and MK-test result with uncertainty range of TS_HazExp
        1980-2010
    regLFull : List MK-output
        Sen_slope and MK-test result with uncertainty range of TS_Full
        1980-2010..
    regH7 : List MK-output
        Sen_slope and MK-test result with uncertainty range of hazard
        (with 1980 fixed exposure)(TS_Haz) 1971-2010
    regH107 : List MK-output
        Sen_slope and MK-test result with uncertainty range of hazard
        (with 2010 fixed exposure)(TS_Haz) 1971-2010
    regH10 : List MK-output
        Sen_slope and MK-test result with uncertainty range of hazard
        (with 2010 fixed exposure)(TS_Haz) 1980-2010
    regE : List MK-output
        Sen_slope and MK-test result with uncertainty range of exposure
        difference function (TS_HazExp - TS_Haz) 1980-2010 (not used)
    regE7 : List MK-output
        Sen_slope and MK-test result with uncertainty range of exposure
        difference function (TS_HazExp - TS_Haz) 1971-2010 (not used)
    regV : List MK-output
        Sen_slope and MK-test result with uncertainty range of vulnerability
        difference function (TS_full - TS_Haz_Exp)(not used)
    regI : List MK-output
        Sen_slope and MK-test result with uncertainty range of modeled damges
        (including vulnerability)
    regN : List MK-output
        Sen_slope and MK-test result with uncertainty range of observed damages
    dis : string
        discharge group

    Returns
    -------
    table1 : DataFrame
        final output for one region

    """

    dam_norm = np.nanmean(data.loc[(data['Year'] > 1979) & (data['Year'] < 1991),
                        'D_Full_{}'.format(dis)])
    nat_norm = np.nanmean(data.loc[(data['Year'] > 1979) & (data['Year'] < 1996),
                        'natcat_damages_2005_CPI_{}'.format(dis)])

    nat_2010 = data.loc[(data['Year'] == 2010), 'natcat_damages_2005_CPI_{}'.format(dis)].mean()

    cH_norm = regLHaz[0]*100/nat_norm

    cH_bot = (cH_norm - regLHaz[2]*100/nat_norm)

    cH_up = (regLHaz[3]*100/nat_norm) - cH_norm

    cH7_normCL = regH7[0]*100/nat_norm

    cH7up_normCL = (regH7[3]*100/nat_norm) - cH7_normCL

    cH7bot_normCL = cH7_normCL - regH7[2]*100/nat_norm

    cH_normCL = regLHaz[0]*100/nat_norm

    cH10_normCL = regH10[0]*100/nat_norm

    cH10up_normCL = (regH10[3]*100/nat_norm)-cH10_normCL

    cH10bot_normCL = (cH10_normCL - regH10[2]*100/nat_norm)

    cH107_normCL = regH107[0]*100/nat_norm

    cH107up_normCL = (regH107[3]*100/nat_norm) - cH107_normCL

    cH107bot_normCL = (cH107_normCL - regH107[2]*100/nat_norm)

    # cH_norm = (regH10.params[0]*100)/dam_norm
    cH7_norm = regH7[0]  # *100/nat_norm
    cE_norm = (regLHazExp[0]-regLHaz[0])*100/nat_norm
    cE10_norm = (regLHazExp[0]-regH10[0]) * 100 / nat_norm
    cV_norm = (regLFull[0]-regLHazExp[0])*100/nat_norm

    cI_norm = regLFull[0]*100/nat_norm
    cN_norm = regN[0]*100/nat_norm


    table1 = pd.DataFrame({'Region': region+'_'+dis,
                           'C_1980_80up': cH_up,
                           'C_1980_80bot': cH_bot,
                           'C_1980_80': cH_normCL,
                           'C_2010_80': cH10_normCL,
                           'C_2010_80up': cH10up_normCL,
                           'C_2010_80bot': cH10bot_normCL,
                           'C_1980_71': cH7_normCL,
                           'C_1980_71up': cH7up_normCL,
                           'C_1980_71bot': cH7bot_normCL,
                           'C_2010_71up': cH107up_normCL,
                           'C_2010_71bot': cH107bot_normCL,
                           'C_2010_71': cH107_normCL,
                           'E_exposure': cE_norm,
                           'E_exposure_2010': cE10_norm,
                           'V_vulnerability': cV_norm,
                           'M_damage_modeled': cI_norm,
                           'N_damage_observed': cN_norm,
                           'p_val_C_1980_80': regLHaz[1],
                           'p_val_C_2010_80': regH10[1],
                           'p_val_C_1980_71': regH7[1],
                           'p_val_C_2010_71': regH107[1],
                           'p_val_M_damage_modeled': regI[1],
                           'p_val_N_damage_observed': regN[1],
                           'delta_2010_since80': (regH10[0]*31/nat_norm)*100,
                           'delta_2010_since71': (regH107[0]*40/nat_norm)*100,

                           },
                          index=[0])
    return table1


def attr_regr(dataFrame):

    attrTable = pd.DataFrame()
    normData = pd.DataFrame()

    for i, test_region in enumerate(test_regions):

        DATA_region = dataFrame[(dataFrame['Region'] == test_region) &
                                (dataFrame['Year'] < 2011) & (dataFrame['Year'] > 1970)]
        DATA_region = DATA_region.reset_index()

        DATA_region = normalise(test_region, DATA_region)

        regLHazPos, regLHazExpPos, regLFullPos, regH7Pos, regH107Pos,\
            regH10Pos, regEPos, regE7Pos, regVPos, regIPos, regNPos =\
            rel_time_attr_MK(DATA_region, 'pos')
        regLHazNeg, regLHazExpNeg, regLFullNeg, regH7Neg, regH107Neg,\
            regH10Neg, regENeg, regE7Neg, regVNeg, regINeg, regNNeg =\
            rel_time_attr_MK(DATA_region, 'neg')
        attrRegPos = prep_table_timeMK(test_region, DATA_region, regLHazPos,
                                       regLHazExpPos, regLFullPos, regH7Pos,
                                       regH107Pos, regH10Pos, regEPos, regE7Pos,
                                       regVPos, regIPos, regNPos, 'pos')
        attrRegNeg = prep_table_timeMK(test_region, DATA_region, regLHazNeg,
                                       regLHazExpNeg, regLFullNeg, regH7Neg,
                                       regH107Neg, regH10Neg, regENeg, regE7Neg,
                                       regVNeg, regINeg, regNNeg, 'neg')

        attrTable = attrTable.append(attrRegPos, ignore_index=True)

        attrTable = attrTable.append(attrRegNeg, ignore_index=True)
        normData = normData.append(DATA_region, ignore_index=True)

    return normData, attrTable


DATA = pd.read_csv('../../../data/reconstruction/vulnerability_adjustment_TimeSeries_subregions.csv')


region_names = {'NAM': 'North America',
                'LAM': 'Central America',
                'EUR': 'Europe',
                'NAF': 'North Africa + Middle East',
                'SSA': 'SSA + Southern Africa',
                'CAS': 'Central Asia + Eastern Europe',
                'SEA': 'Southern Asia + South-East Asia',
                'EAS': 'Eastern Asia',
                'OCE': 'Oceania',
                'GLB': 'Global'}

test_regions = list(region_names)



normData, attrTable = attr_regr(DATA)

attrTable.to_csv('../../../data/reconstruction/attribution_MetaData_subregions.csv', index=False)

normData.to_csv('../../../data/reconstruction/attribution_TimeSeries_subregions.csv', index=False)