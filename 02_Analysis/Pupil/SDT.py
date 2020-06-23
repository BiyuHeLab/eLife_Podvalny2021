#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:05:03 2019

@author: podvae01
"""
import sys
sys.path.append('../../')
from os import path
import HLTP_pupil
from HLTP_pupil import MEG_pro_dir, subjects
import numpy as np
import scipy
from scipy.optimize import curve_fit
import scipy.stats
from scipy.stats import sem, zscore
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.eval_measures import bic, aic
from statsmodels.stats.multitest import multipletests

bhv_vars = ['HR', 'FAR', 'c', 'd', 'p_corr', 'catRT', 'recRT']
n_roi = 7
def sdt_from_df(df):
    n_rec_real = sum(df[df.real_img == True].recognition == 1)
    n_real = len(df[(df.real_img == True) & (df.recognition != 0)])
    n_rec_scr = sum(df[df.real_img == False].recognition == 1)
    n_scr = len(df[(df.real_img == False) & (df.recognition != 0)])
    p_correct = df.correct.values.mean()   
    catRT =  df.catRT.values.mean()
    recRT =  df.recRT.values.mean()
    HR, FAR, d, c = get_sdt_msr(n_rec_real, n_real, n_rec_scr, n_scr)
    return HR, FAR, d, c, p_correct, catRT, recRT
        
def get_sdt_msr(n_rec_signal, n_signal, n_rec_noise, n_noise):
    Z = scipy.stats.norm.ppf
    if (n_noise == 0): FAR = np.nan
    else: FAR = max( float(n_rec_noise) / n_noise, 1. / (2 * n_noise) )
    if  n_signal == 0: HR = np.nan
    else: HR = min( float(n_rec_signal) / n_signal, 1 - (1. / (2 * n_signal) ) )
    d = Z(HR)- Z(FAR)
    c = -(Z(HR) + Z(FAR))/2.
    
    # return nans instead of infs
    if np.abs(d) == np.inf: d = np.nan
    if np.abs(c) == np.inf: c = np.nan
    return HR, FAR, d, c

def fill_dict_w_bhv(HR, FAR, d, c, p_correct, catRT, recRT, subject, group, IV):
    bhv_group_val = {}
    bhv_group_val['HR'] = [HR]; 
    bhv_group_val['FAR'] = [FAR]; 
    bhv_group_val['d'] = [d]; 
    bhv_group_val['c'] = [c];
    bhv_group_val['p_corr'] = [p_correct] 
    bhv_group_val['catRT'] = [catRT]
    bhv_group_val['recRT'] = [recRT]
    bhv_group_val['subject'] = subject; 
    bhv_group_val['group'] = [group]
    bhv_group_val['IV'] = IV
    return bhv_group_val

def get_SDT_by_IV(bhv_df, IV):
    ''' calculate Behavioral SDT variables in pupil groups '''
    group_percentile = np.arange(0., 100., 20);
    dfs = []
    for subject in subjects:
        # TODO: update the df w non-digitized pupil for consistency & remove 
        # the  if statement below
        if IV != 'pupil_size_pre':
            pwr = bhv_df.loc[bhv_df.subject == subject, IV].values
            p_group = np.digitize(pwr, 
                                  np.percentile(pwr, group_percentile))
        else:
            p_group = bhv_df[bhv_df.subject == subject][IV]
            
        for group in np.unique(p_group):

            group_df = bhv_df[bhv_df.subject == subject].loc[p_group == group]

            HR, FAR, d, c, p_correct, catRT, recRT = sdt_from_df(group_df)
            bhv_group_val = fill_dict_w_bhv(HR, FAR, d, c, 
                                            p_correct, catRT, recRT,
                                            subject, group, IV) 

            dfs.append(pd.DataFrame(bhv_group_val));
    
    sdt_df = pd.concat(dfs)
    sdt_df.to_pickle(HLTP_pupil.MEG_pro_dir +
                         '/results/sdt_by_' + IV + '.pkl')
    return sdt_df


def stats_for_SDT(sdt_df, IV, savetag):
    """
    Predict behavior from any independent variable (group) in the dataframe

    Parameters
    ----------
    sdt_df : pandas dataframe
         dataframe with columns indicating behavioral variables and independent
         variables groups such as pupil, residual power
         
    IV : STRING
        Independent Variable, IV = "pupil_group"# independent variable
        
    savetag: tag describing the saved models, beyond the IV, indicating ROI or
        frequency band, for example. 

    Returns
    -------
    None.

    """
    # fit a model for each behavioral variable (i.e., dependent variable)
    for bhv_var in bhv_vars:
        # quadratic
        mdf_Q = smf.mixedlm(bhv_var + " ~ np.power(group, 2) + group", sdt_df.dropna(), 
                        groups = sdt_df.dropna()["subject"]).fit()
        mdf_Q.bic = bic(mdf_Q.llf, mdf_Q.nobs, mdf_Q.df_modelwc)
        mdf_Q.save(HLTP_pupil.MEG_pro_dir +
                        '/results/mixedlmQ_' + IV + savetag + '_' + bhv_var +'.pkl')
        #print(mdf_Q.summary())
        # linear
        mdf_L = smf.mixedlm(bhv_var + " ~ group", sdt_df.dropna(),
                         groups = sdt_df.dropna()["subject"]).fit()
        mdf_L.bic = bic(mdf_L.llf, mdf_L.nobs, mdf_L.df_modelwc)
        mdf_L.save(HLTP_pupil.MEG_pro_dir +
                         '/results/mixedlmL_' + IV + savetag + '_' + bhv_var
                         +  '.pkl')
        print('saving to ' + IV + savetag + '_' + bhv_var)
        print(mdf_L.summary())
          
def add_subject_residual_power(bhv_df):
    """
    Add a column of residual power after regressing out the pupil-linked power

    Parameters
    ----------
    bhv_df : TYPE
        Dataframe including single trial behavior, pupil size and power.

    Returns
    -------
    bhv_df : TYPE
        DESCRIPTION.

    """

    for roi in range(n_roi):
        for fband, _ in HLTP_pupil.freq_bands.items():
            DV = fband + str(roi)
            bhv_df[ DV + 'res'] = 0
            for subject in HLTP_pupil.subjects:                            
                subj_df = bhv_df[bhv_df.subject == subject]
                mdf_Q = smf.mixedlm(DV + " ~ np.power(pupil, 2) + pupil", 
                        subj_df.dropna(), groups = subj_df.dropna()["subject"]
                        ).fit(method='powell')
                bhv_df.loc[
                    bhv_df.subject == subject,  DV + 'res'] = mdf_Q.resid.values
    return bhv_df
        
def correct_pvals_by_roi(savetag):
    """
    correct the p-values across ROIs

    Returns
    -------
    None.

    """
    for bhv_var in bhv_vars:
        table_par_e = np.zeros( (7, 5) )
        table_pvals = np.zeros( (7, 5) )
        for bi, band in enumerate(HLTP_pupil.freq_bands.keys()):
            roi_pval = []
            for roi in range(7):
                IV =  band + str(roi)
                save_name = IV + savetag + '_' + bhv_var
                mdf_L = pd.read_pickle(
                    HLTP_pupil.MEG_pro_dir + '/results/mixedlmL_' + save_name
                                +'.pkl')
                table_par_e[roi, bi] = mdf_L.params[1]
                roi_pval.append(mdf_L.pvalues[1])
            table_pvals[:, bi] = multipletests(roi_pval, method = 'fdr_bh')[1]
        print(bhv_var, table_pvals < 0.05)
        HLTP_pupil.save([table_par_e, table_pvals], HLTP_pupil.MEG_pro_dir + 
                        '/results/' + savetag + 'pe_pval_' + bhv_var +'.pkl')
    return        
            
# Run the functions
df_name = 'all_subj_bhv_w_pupil_power'# dataframe with power 
bhv_df = pd.read_pickle(HLTP_pupil.MEG_pro_dir +
                          '/results/' + df_name + '.pkl')

# analysis of behavior according to prestimulus pupil size
sdt_df = get_SDT_by_IV(bhv_df, IV = "pupil");  
model_name = stats_for_SDT(sdt_df, IV = "pupil", savetag = '') 
# analysis of behavior according to prestimulus residual pupil
# TODO: combine with code in Independent_Pupil_to_Behavior

# analysis of behavior according to prestimulus power
for roi in range(n_roi):
    for fband in HLTP_pupil.freq_bands.keys():
        IV = fband + str(roi)
        sdt_df = get_SDT_by_IV(bhv_df, IV)
        stats_for_SDT(sdt_df, IV, savetag = '')

correct_pvals_by_roi(savetag = '')


# analysis of behavior according to prestimulus residual power
bhv_df = add_subject_residual_power(bhv_df)
for roi in range(n_roi): 
    for fband in HLTP_pupil.freq_bands.keys():
        IV = fband + str(roi) + 'res'
        sdt_df = get_SDT_by_IV(bhv_df, IV)
        stats_for_SDT(sdt_df, IV, savetag = '')
correct_pvals_by_roi(savetag = 'res')
        
# TODO: goto the place in code where I write th below df and fix its name 
# until then combine the two dataframes:
# df = pd.read_pickle(HLTP_pupil.MEG_pro_dir + 
#                                 '/results/all_subj_bhv_df_w_roi_pwr.pkl') 
# df = df[ ~((df.index == 288) & (df.subject == 'BJB'))]
# bhv_df['pupil'] = df.pupil_size_pre.values
# for fband, _ in HLTP_pupil.freq_bands.items():
#         for roi in range(7):
#             bhv_df[fband + str(roi)] = df[fband + str(roi)].values
# b = 'task_prestim'

# for subject in HLTP_pupil.subjects:            
#     pupil_states = HLTP_pupil.load(
#                             HLTP_pupil.result_dir + '/pupil_states_' + 
#                                        b + subject + '.pkl')
#     bhv_df.loc[bhv_df.subject == subject,  "pupil"] = pupil_states.mean_pupil       
# bhv_df.to_pickle(HLTP_pupil.MEG_pro_dir +
#                              '/results/' + df_name + '.pkl')