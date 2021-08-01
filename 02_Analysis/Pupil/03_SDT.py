#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:05:03 2019

@author: podvae01
"""
import sys
sys.path.append('../../')
import HLTP_pupil
from HLTP_pupil import MEG_pro_dir, subjects
import numpy as np
import scipy
import scipy.stats
from scipy.stats import sem, zscore
from scipy.stats.distributions import chi2
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from rpy2.robjects.packages import importr
from pymer4.models import Lmer
performance = importr('performance')
piecewiseSEM = importr('piecewiseSEM')

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
    group_percentile = np.arange(0., 100., 20)
    dfs = []
    for subject in subjects:
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
    sdt_df.to_pickle(HLTP_pupil.result_dir +
                         '/sdt_by_' + IV + '.pkl')
    return sdt_df

def stats_for_SDT(sdt_df, IV, savetag = '', _reml = False):
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
        sdt_df.group = zscore(sdt_df.group)
        sdt_df['group_sqr'] = np.power(sdt_df.group, 2)
        try:
            mdf_Q = smf.mixedlm(bhv_var + " ~ np.power(group, 2) + group",
                            sdt_df.dropna(), 
                        groups = sdt_df.dropna()["subject"],
                        re_formula = "~ np.power(group, 2) + group"
                        ).fit(reml = _reml, method = 'Powell')

        except: # default fit method
            mdf_Q = smf.mixedlm(bhv_var + " ~ np.power(group, 2) + group",
                            sdt_df.dropna(),
                        groups = sdt_df.dropna()["subject"],
                        re_formula = "~ np.power(group, 2) + group" #keep only lin re
                        ).fit(reml = _reml)

        model = Lmer(bhv_var + ' ~ 1 + group_sqr + group + (1 + group_sqr + group|subject)',
                     data=sdt_df.dropna())
        model.fit(REML=False, method='Powell')

        r2 = piecewiseSEM.rsquared(model.model_obj)
        mdf_Q.marginal_r2 = r2.Marginal[0]
        mdf_Q.conditional_r2 = r2.Conditional[0]
        mdf_Q.save(HLTP_pupil.result_dir +
                        '/mixedlmQ_' + IV + savetag + '_' + bhv_var +'.pkl')

        # linear
        try:
            mdf_L = smf.mixedlm(bhv_var + " ~ group", sdt_df.dropna(),
                             groups = sdt_df.dropna()["subject"],
                            re_formula =  "~ group").fit(reml = _reml, method = 'Powell')

        except: # go to default fit method
            mdf_L = smf.mixedlm(bhv_var + " ~ group", sdt_df.dropna(),
                                groups=sdt_df.dropna()["subject"],
                            re_formula =  "~ group").fit(reml=_reml)

        model = Lmer(bhv_var + ' ~ 1 + group + (1 + group|subject)',
                     data=sdt_df.dropna())
        model.fit(REML=False, method='Powell')

        r2 = piecewiseSEM.rsquared(model.model_obj)
        mdf_L.marginal_r2 = r2.Marginal[0]
        mdf_L.conditional_r2 = r2.Conditional[0]
        mdf_L.save(HLTP_pupil.result_dir +
                   '/mixedlmL_' + IV + savetag + '_' + bhv_var
                   + '.pkl')
        print('saving to ' + HLTP_pupil.result_dir +
                       '/mixedlmL_' + IV + savetag + '_' + bhv_var
                       + '.pkl')
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
                subj_df["constant"] = 1  
                mdf_Q = smf.ols(DV + " ~ np.power(pupil, 2) + pupil + constant", 
                        subj_df.dropna()).fit()
                mdf_L = smf.ols(DV + " ~ pupil + constant",
                        subj_df.dropna()).fit()
                if mdf_Q.aic < mdf_L.aic:
                    res_val = mdf_Q.resid.values
                else:
                    res_val = mdf_L.resid.values
                bhv_df.loc[
                    bhv_df.subject == subject,  DV + 'res'] = res_val
    return bhv_df
        
def correct_pvals_by_roi(savetag):
    """
    correct the p-values across ROIs

    Returns
    -------
    None.

    """
    for bhv_var in bhv_vars:
        table_par_e = np.zeros( (7, 5) ) + np.inf
        table_pvals = np.ones( (7, 5) ) + np.inf
        table_pvals_corr = np.ones( (7, 5) ) +  np.inf
        for bi, band in enumerate(HLTP_pupil.freq_bands.keys()):
            for roi in range(7):
                IV =  band + str(roi)
                save_name = IV + savetag + '_' + bhv_var
                mdf_L = pd.read_pickle(
                    HLTP_pupil.result_dir + '/mixedlmL_' + save_name
                                +'.pkl')
                if mdf_L.converged:
                    table_par_e[roi, bi] = mdf_L.params[1]
                    table_pvals[roi, bi] = mdf_L.pvalues[1]
            table_pvals_corr[:, bi] = multipletests(table_pvals[:, bi], method = 'fdr_bh')[1]
        #print(multipletests(roi_pval, method = 'fdr_bh')[1] < 0.05)
        print(bhv_var, table_pvals < 0.05)
        HLTP_pupil.save([table_par_e, table_pvals, table_pvals_corr], HLTP_pupil.result_dir +
                        '/' + savetag + 'pe_pval_' + bhv_var +'.pkl')
    #check the quadratic models
    print("Quadratic p-vals")
    for bhv_var in bhv_vars:
        table_par_e = np.zeros( (7, 5) )
        table_pvals = np.zeros( (7, 5) )
        for bi, band in enumerate(HLTP_pupil.freq_bands.keys()):
            roi_pval = []
            for roi in range(7):
                IV =  band + str(roi)
                save_name = IV + savetag + '_' + bhv_var
                mdf_L = pd.read_pickle(
                    HLTP_pupil.result_dir + '/mixedlmL_' + save_name
                                +'.pkl')
                mdf_Q = pd.read_pickle(
                    HLTP_pupil.result_dir + '/mixedlmQ_' + save_name
                                +'.pkl')
                #print("Q-conv:", mdf_Q.converged, "L-conv:", mdf_L.converged)
                if mdf_Q.aic < mdf_L.aic:
                    print("Q is better")
                    print("Q-conv:", mdf_Q.converged, "Sig:", mdf_Q.pvalues[1], " ", save_name)
                    print("L-conv:", mdf_L.converged, "Sig:", mdf_L.pvalues[1], " ", save_name)
                #else:
                #    print("L is better")
                table_par_e[roi, bi] = mdf_L.params[1]
                roi_pval.append(mdf_L.pvalues[1])
            table_pvals[:, bi] = multipletests(roi_pval, method = 'fdr_bh')[1]
        print(bhv_var, table_pvals < 0.05)
        HLTP_pupil.save([table_par_e, table_pvals], HLTP_pupil.result_dir +
                        '/' + savetag + '_Q_pe_pval_' + bhv_var +'.pkl')
    return

def compare_L_and_Q_models(bhv_vars, IV = "pupil", savetag = ''):
    for bhv_var in bhv_vars:
        mdf_Q = HLTP_pupil.load(HLTP_pupil.result_dir +
                        '/mixedlmQ_' + IV + savetag + '_' + bhv_var +'.pkl')
        mdf_L = HLTP_pupil.load(HLTP_pupil.result_dir +
                        '/mixedlmL_' + IV + savetag + '_' + bhv_var +'.pkl')

        LR = 2 * ( mdf_Q.llf - mdf_L.llf)
        DOF_diff = mdf_Q.params.shape[0] - mdf_L.params.shape[0]
        p = chi2.sf(LR, DOF_diff)
        print(bhv_var + "pval:", p)

def test_blink_effects_on_SDT():
    df_name = 'all_subj_bhv_df_w_pupil_power'  # prepare this with 4_DICS_roi_analysis
    # df_name = 'all_subj_bhv_df_w_pupil'
    bhv_df = pd.read_pickle(HLTP_pupil.result_dir +
                            '/' + df_name + '.pkl')
    dfs = []
    for subject in HLTP_pupil.subjects:
        # load the array indicating whether there has occurred at least one blink
        # during a pre-stimulus interval
        blinks = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir
                       + '/' + subject +  '/blinks_for_prestim_epochs.pkl')
        for blink in [True, False]:
            HR, FAR, d, c, p_correct, catRT, recRT = sdt_from_df(
                bhv_df[bhv_df.subject == subject].loc[blinks == blink])
            df = pd.DataFrame({"subject": subject, "blink":blink,
                                 "HR":[HR], "FAR":[FAR], "d":[d], "c":[c],
                                 "p_correct":[p_correct], "catRT":[catRT] })
            dfs.append(df)
    adf = pd.concat(dfs)

def check_bic_of_res_power_models():
    for bhv_var in bhv_vars:
        for roi in range(n_roi):
            for fband in HLTP_pupil.freq_bands.keys():
                bhv_var
                IV = fband + str(roi) + 'res'
                mdf_Q = HLTP_pupil.load(HLTP_pupil.result_dir +
                            '/mixedlmQ_' + IV  + '_' + bhv_var + '.pkl')
                mdf_L = HLTP_pupil.load(HLTP_pupil.result_dir +
                            '/mixedlmL_' + IV  + '_' + bhv_var + '.pkl')
                if mdf_Q.bic < mdf_L.bic:
                    print(bhv_var, str(roi), fband, mdf_Q.params[1], mdf_Q.pvalues[1])

bhv_vars = ['HR', 'FAR', 'c', 'd', 'p_corr', 'catRT', 'recRT']
n_roi = 7
# Run the functions
df_name = 'all_subj_bhv_df_w_pupil_power'# prepare this with 4_DICS_roi_analysis
#df_name = 'all_subj_bhv_df_w_pupil'
bhv_df = pd.read_pickle(HLTP_pupil.result_dir +
                          '/' + df_name + '.pkl')

# analysis of behavior according to prestimulus pupil size
sdt_df = get_SDT_by_IV(bhv_df, IV = "pupil")
model_name = stats_for_SDT(sdt_df, IV = "pupil", savetag = '')

# analysis of behavior according to prestimulus residual power
df = pd.read_pickle(HLTP_pupil.result_dir +
                    '/roi_pwr_and_pupiltask_prestim.pkl')
df = add_subject_residual_power(df)
#bhv_df = add_subject_residual_power(bhv_df)
for roi in range(n_roi): 
    for fband in HLTP_pupil.freq_bands.keys():
        IV = fband + str(roi) + 'res'
        bhv_df[IV] = df[IV]
        sdt_df = get_SDT_by_IV(bhv_df, IV)
        stats_for_SDT(sdt_df, IV, savetag = '', _reml = False)
correct_pvals_by_roi(savetag = 'res')



# analysis of behavior according to prestimulus residual pupil - REMOVED
#sdt_df = pd.read_pickle(HLTP_pupil.result_dir +
#                         '/sdt_by_pupil_resid_df_3.pkl')

#model_name = stats_for_SDT(sdt_df, IV = "pupil", savetag = 'res3_')

# analysis of behavior according to prestimulus power - NOT USED
#for roi in range(n_roi):
#    for fband in HLTP_pupil.freq_bands.keys():
#        IV = fband + str(roi)
#        sdt_df = get_SDT_by_IV(bhv_df, IV)
#        stats_for_SDT(sdt_df, IV, savetag = '')
#correct_pvals_by_roi(savetag = '')

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