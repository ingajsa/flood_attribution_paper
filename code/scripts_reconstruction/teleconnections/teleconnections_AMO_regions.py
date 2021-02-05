#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 19:34:03 2020

@author: insauer

This script test damage time series on the influence of teleconnections and
GMT. Tested are the teleconnections with ENSO, NAO, PDO and the effect of AMO.
This script is for entire regions.

"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import itertools
import pymannkendall as mk
from scipy.stats import shapiro
import numpy.ma as ma
import matplotlib.pyplot as plt

DATA_TS = pd.read_csv('../../../data/reconstruction/attribution_TimeSeries_regions.csv')
DATA_ATTR = pd.read_csv('../../../data/reconstruction/attribution_MetaData_regions.csv')


teleData = pd.read_csv('../../../data/climate_oscillations/norm_oscillations_lag.csv')

normClim = True

link_fnc_list = [sm.families.links.log(), sm.families.links.identity(),
                 sm.families.links.inverse_power()]

teleData80 = teleData.loc[teleData['Year'] >= 1980]
telecon80 = teleData80[['ENSO', 'ENSOlag', 'AMO', 'AMOlag', 'PDO', 'PDOlag', 'NAO', 'NAOlag']]

teleData = teleData.loc[teleData['Year'] >= 1971]
telecon = teleData[['ENSO', 'ENSOlag', 'AMO', 'AMOlag', 'PDO', 'PDOlag', 'NAO', 'NAOlag']]

region_names = {'GLB': 'Global (GLB)',
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

predictors = ['ENSO', 'ENSOlag', 'AMO', 'PDO', 'PDOlag', 'NAO', 'NAOlag']

test_regions = list(region_names)


def get_pearson(pred, climdat):
    """
    pearson correlation of model predicted data and damage time series

    Parameters
    ----------
    pred : GLM
        model
    climdat : np.array
        damage time series

    Returns
    -------
    float
        Pearson correlation coefficient

    """
    a = ma.masked_invalid(climdat)
    b = ma.masked_invalid(pred.predict())
    msk = (~a.mask & ~b.mask)
    corrcoef = ma.corrcoef(a[msk], b[msk])

    # corrcoef = stats.spearmanr(a[msk], b[msk])

    return corrcoef[0, 1]


def looCV(clim, predic, fnc):
    """This function calculates the the leave-one-out-Cross-Validation

    Parameters
    ----------
    clim : damage time series
        DESCRIPTION.
    predic : DataFrame
        DESCRIPTION.
    fnc : LinkFunctionObject
        link function

    Returns
    -------
    out of sample error

    """

    err = 0
    for lo_index in range(len(clim)):

        clim_mask = np.ma.array(clim, mask=False)
        clim_mask.mask[lo_index] = True
        clim_lo = clim_mask.compressed()

        predic_lo = predic.reset_index().drop(lo_index, axis=0).drop('index', 1)

        model_res = sm.GLM(clim_lo, predic_lo,
                           family=sm.families.Gamma(fnc)).fit(maxiter=5000, scale=1.)

        value_pred = model_res.predict(predic).iloc[lo_index]

        err = err + (clim[lo_index] - value_pred)**2

    return err/len(clim)


def pred_double(comb):

    pred_names = ['AMO', 'ENSO', 'NAO', 'PDO']

    for p in pred_names:
        if p in comb:
            if p + '_lag' in comb:
                return True

    return False

def find_best_model(climateDat, telecon):
    """
    Wrapper function to select the best model. Function provides all possible
    combination of predictors and a constant and evaluates the model applying
    the LooCV. It selects the model with the smallest out-of-sample-error.

    Parameters
    ----------
    climateDat : np.array
        damage time series
    telecon : DataFrame
        teleconnections and GMT

    Returns
    -------
    best_model: GLMObject
        full GLM object of the best model
    best_model_indices[0]: int
        x-index of the best model (indicates combination of predictors)
    best_model_indices[1]: int
        y-index of the best model (indicates link function)
    iter_max: int
        number iterations needed for convergence of the best model
    pearson_corr: float
        pearson correlation of model and data
    best_loo: float
        out-of-sample-error of the best model
    looICs_lf: np.array
        all out-off-sample errors
    """
    max_i = 5000

    if test_region == 'OCE':

        climateDat = np.nan_to_num(climateDat)

    models_lf = []
    deviances_lf = []
    chi2_lf = []
    looICs_lf = []
    iter_max = 0
    comb_list_lf = []

    for link_fnc in link_fnc_list:
        models = []
        deviances = []
        chi2 = []
        looICs = []
        comb_list = []
         
        for n_preds in range(0, 5):
            for comb in list(itertools.combinations(predictors, n_preds)):

                print(list(comb))
                if pred_double(comb):
                    print('skip combination')
                    continue
                data_exog = sm.add_constant(telecon[list(comb)])
                try:

                    model_result = sm.GLM(climateDat, data_exog,
                                          family=sm.families.Gamma(link_fnc)).fit(maxiter=max_i,
                                                                                  scale=1.)
                    looIC = looCV(climateDat, data_exog, link_fnc)

                    models.append(model_result)
                    looICs.append(looIC)
                    deviances.append(model_result.aic)
                    chi2.append(model_result.pearson_chi2)
                    comb_list.append(comb)

                except ValueError:

                    models.append(sm.GLM(data_exog, np.ones(len(climateDat))))
                    deviances.append(1e10)
                    looICs.append(1e10)
                    chi2.append(1e10)
                    comb_list.append(comb)
                if model_result.fit_history['iteration'] == max_i:
                    iter_max += 1

        models_lf.append(models)
        looICs_lf.append(looICs)
        deviances_lf.append(deviances)
        chi2_lf.append(chi2)
        comb_list_lf.append(comb_list)

    best_model_indices = np.array(np.unravel_index(np.argmin(np.array(looICs_lf), axis=None),
                                                   np.array(looICs_lf).shape))

    best_model = models_lf[best_model_indices[0]][best_model_indices[1]]

    best_loo = looICs_lf[best_model_indices[0]][best_model_indices[1]]

    pearson_corr = get_pearson(best_model, climateDat)

    return best_model, best_model_indices[0], best_model_indices[1],\
        iter_max, pearson_corr, best_loo, looICs_lf, comb_list_lf


def test_residuals(model, timeperiod,reg):
    """
    Test for a residual trend, applying a Mann-Kendall-test

    Parameters
    ----------
    model : GLMObject
        Best model
    timeperiod : np.array
        considered years (not used here)

    Returns
    -------
    float
        slope in residuals
    float
        p-value

    """
    res_trend = mk.original_test(model.resid_response, alpha=0.1)
    

    return res_trend.slope, res_trend.p

def test_autocorrelation(time_series):
    """
    Test for autocorrelation

    Parameters
    ----------
    time-series

    Returns
    -------
    float
        tau

    """
    auto = mk.original_test(time_series, alpha=0.1)

    return auto.Tau


def extract_model_coefs_short(region, model, link):
    """
    Reads the coefficients and p-values for each predictor of the best model
    and saves data in a csv file. To achieve comparability between coefficients of
    models with different link functions the partial devaritive in a centric
    development point is calculated.

    Parameters
    ----------
    region : string
        region abbreviation
    model : GLMObject
        best model (1971-2010, 1980 fixed exposure)
    link : int
        index of link function (1971-2010, 1980 fixed exposure)

    """
    mods = [model]
    coefs = ['ENSO', 'ENSOlag', 'AMO', 'PDO', 'PDOlag', 'NAO', 'NAOlag']
    lnks = [link]
    dev_index = [20, 20, 15, 15]

    for m, mod in enumerate(mods):

        coeff_sum = 0

        for c, coef in enumerate(coefs):
            try:
                DATA_ATTR.loc[DATA_ATTR['Region'] == region,
                              'alpha_{}_amo_run'.format(coef)] = mod.params[coef]
                coef_deriv = link_fnc_list[lnks[m]].inverse_deriv(teleData[coef])\
                    * mod.params[coef]
                DATA_ATTR.loc[DATA_ATTR['Region'] == region,
                              'gammaAbs_{}_amo_run'.format(coef)] = np.array(coef_deriv)[dev_index[m]]
                DATA_ATTR.loc[DATA_ATTR['Region'] == region,
                              'pval_{}_amo_run'.format(coef)] = mod.pvalues[coef]
                coeff_sum += mod.params[coef]
            except KeyError:
                DATA_ATTR.loc[DATA_ATTR['Region'] == region,
                              'alpha_{}_amo_run'.format(coef)] = np.nan
                DATA_ATTR.loc[DATA_ATTR['Region'] == region,
                              'gammaAbs_{}_amo_run'.format(coef)] = np.nan
                DATA_ATTR.loc[DATA_ATTR['Region'] == region,
                              'pval_{}_amo_run'.format(coef)] = np.nan

    DATA_ATTR.to_csv('../../../data/reconstruction/ENSO_AMO_PDO_NAO_regions.csv')


def unexplainable_Trends(pval, slope, test_region, change, sign):
    """
    Check for the presence of an unexplainable trend after adjustment for
    teleconnections

    Parameters
    ----------
    pval : float
        p-value of residual trend
    slope : float
        slope of residual trend
    test_region : string
        region
    change : slope
        slope of trend in damages
    sign : float
        significance of slope in damages

    Returns
    -------
    bool

    """
    if pval > 0.1:
        return False

    haz_slope = DATA_ATTR.loc[DATA_ATTR['Region'] == test_region, change].sum()

    if (haz_slope < 0) and (slope > 0):
        return False

    if (haz_slope > 0) and (slope < 0):
        return False

    return True


for i, test_region in enumerate(test_regions):

    print(test_region)
    DATA_region = DATA_TS[(DATA_TS['Region'] == test_region) &
                          (DATA_TS['Year'] < 2011) & (DATA_TS['Year'] > 1970)]

    DATA_region80 = DATA_TS[(DATA_TS['Region'] == test_region) &
                            (DATA_TS['Year'] < 2011) & (DATA_TS['Year'] > 1979)]


    climateData = np.array(DATA_region['Norm_ImpFix_2y_offset'])
    auto_corr = test_autocorrelation(climateData)


    if normClim is True:

        climateData = climateData/np.nanmax(climateData)


    t, shap_log = shapiro(np.log(climateData))

    t, shap_norm = shapiro(climateData)

    best_model, y, x, maxi, pearson_corr, best_loo, loos, combs = find_best_model(climateData, telecon)

    
    comb_df = pd.DataFrame(combs)
    comb_df = comb_df.T
    

    loo_df = pd.DataFrame(loos)
    loo_df = loo_df.T
    loo_df.columns = ['log', 'identity', 'inverse-power']
    loo_df['combination'] = comb_df.iloc[:, 0]
    loo_df.to_csv('../../../data/reconstruction/LooIC_ENSO_AMO_PDO_NAO_{}.csv'.format(test_region))

    extract_model_coefs_short(test_region,  best_model, y)

    coef, pval = test_residuals(best_model, np.arange(1971, 2011), test_region)


    unexT = unexplainable_Trends(pval, coef, test_region, 'C_1980_71', 'p_val_C_1980_71')


    DATA_ATTR.loc[DATA_ATTR['Region'] == test_region, 'pval_residual_trend_amo_run'] = pval


    DATA_ATTR.loc[DATA_ATTR['Region'] == test_region, 'residual_trend_amo_run'] = coef


    DATA_ATTR.loc[DATA_ATTR['Region'] == test_region, 'Unexplained Trend'] = unexT


    DATA_ATTR.loc[DATA_ATTR['Region'] == test_region, 'R2_D_1980_bm_amo_run'] = pearson_corr


    DATA_ATTR.loc[DATA_ATTR['Region'] == test_region, 'oose_bm_amo_run'] = best_loo

    DATA_ATTR.to_csv('../../../data/reconstruction/ENSO_AMO_PDO_NAO_regions.csv')

DATA_ATTR.to_csv('../../../data/reconstruction/ENSO_AMO_PDO_NAO_regions.csv')
