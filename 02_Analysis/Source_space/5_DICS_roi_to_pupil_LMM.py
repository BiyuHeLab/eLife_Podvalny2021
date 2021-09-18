#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:06:29 2020

Fit models of spectral power in resting state networks as a function of pupil size.
The fit is done with statsmodels, but we had to use R through pymer to be able to calculate R2
After running this script figure 3 can be plotted and results summary table printed:
To plot figure 3: 04_Visualization/Figure3.py
To print the results table: 03_Reporting/Print_Pupil_Brain_LMM_results.py

@author: podvae01
"""
import numpy as np
import HLTP_pupil
from HLTP_pupil import MEG_pro_dir
import pandas as pd
from rpy2.robjects.packages import importr
from pymer4.models import Lmer
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import zscore
import os
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
from scipy.stats.distributions import chi2
performance = importr('performance')
piecewiseSEM = importr('piecewiseSEM')

method = 'dics'
def pupil_roi_power_df():
    # prepare dataframes for all conditions including  task and rest
    for b in ['rest01', 'rest02', 'task_prestim']:#, 
        dfs = []
        for subject in HLTP_pupil.subjects:
            # load pupil
            fname = HLTP_pupil.result_dir + '/pupil_states_' + b + subject + \
                '.pkl'
            if not os.path.exists(fname): continue
            pupil = HLTP_pupil.load(fname)
            subj_df = pd.DataFrame({"subject" :np.repeat(subject, len(pupil)), 
                           "pupil" :pupil.mean_pupil})
            # load mean roi trials
            sub_pro_dir = MEG_pro_dir + '/' + subject
            # this is full frequency power data in all 2-sec intervals
            roi_data = HLTP_pupil.load(sub_pro_dir + 
                                       '/roi_single_epoch_dics_power_' + b + '_ds')
            for roi in range(7):
                for fband, frange in HLTP_pupil.freq_bands.items():
                    subj_df[fband + str(roi)] = (np.log(roi_data[roi, 
                        (frange[0] - 1) : (frange[1] - 1), :]).mean(axis = 0))
                    # center the data
                    subj_df[fband + str(roi)] = subj_df[fband + str(roi)] - np.mean(subj_df[fband + str(roi)])

            dfs.append(subj_df)  
        df = pd.concat(dfs)
        # save df
        df.to_pickle(HLTP_pupil.result_dir +
                            '/roi_pwr_and_pupil_blocknorm_' + b + '.pkl')
    # combine the two rest runs
    rest1_df = pd.read_pickle(HLTP_pupil.result_dir +
                            '/roi_pwr_and_pupil_blocknorm_rest01.pkl')
    rest2_df = pd.read_pickle(HLTP_pupil.result_dir +
                            '/roi_pwr_and_pupil_blocknorm_rest02.pkl')

    pd.concat([rest1_df, rest2_df]).to_pickle(HLTP_pupil.result_dir +
                            '/roi_pwr_and_pupil_blocknorm_rest.pkl')

def fitLM_for_roi_pwr():
    '''fit linear mixed effect models, power by pupil'''
    n_rois = 7
    for b in ['task_prestim', 'rest']:
        df = pd.read_pickle(HLTP_pupil.result_dir +
                            '/roi_pwr_and_pupil_blocknorm_' + b + '.pkl')
        df.pupil = zscore(df.pupil) # so we have standardized betas
        df["pupil_sqr"] = np.power(df.pupil, 2)

        # fit the model for each roi and frequency band
        for fband in HLTP_pupil.freq_bands.keys():
            for roi in range(n_rois):
                # fit using standard likelihood so AIC/BIC is comparable
                fit_formula = " ~ pupil_sqr + pupil"
                re_fit_formula = fit_formula + " + 0"
                lmm_Q = smf.mixedlm(fband + str(roi) + fit_formula,
                        df.dropna(), groups = df.dropna()["subject"],
                        re_formula = re_fit_formula)
                mdf_Q = lmm_Q.fit(method = 'Powell', reml = False)
                if not mdf_Q.converged or np.any(np.isnan(mdf_Q.bse)):
                    print('the Q model for roi' + str(roi) + fband + b +
                          ' did not converge, consider implementing' +
                          ' dropping random factors procedure here')
                    #return 1


                model = Lmer(fband + str(roi) + ' ~ 1 + pupil_sqr + pupil + (0 + pupil_sqr + pupil|subject)',
                             data = df.dropna())
                model.fit(REML=False, method='Powell')

                r2 = piecewiseSEM.rsquared(model.model_obj)
                mdf_Q.marginal_r2 = r2.Marginal[0]
                mdf_Q.conditional_r2 = r2.Conditional[0]

                fit_formula = " ~ pupil"; re_fit_formula = fit_formula + " + 0"
                lmm_L = smf.mixedlm(fband + str(roi) + " ~ pupil",
                                df.dropna(), groups = df.dropna()["subject"],
                                re_formula = re_fit_formula)
                mdf_L = lmm_L.fit(method = 'Powell', reml = False)
                if not mdf_L.converged or np.any(np.isnan(mdf_L.bse)):
                    print('the L model for roi' + str(roi) + fband + b +
                          ' didnt converge, ' +
                          ' dropping random factors')

                    #return 1
                # fit with pymer to get the r2
                model = Lmer(fband + str(roi)  + ' ~ pupil + 1 + (0 + pupil|subject)', data = df.dropna())
                model.fit(REML=False, method='Powell')
                r2 = piecewiseSEM.rsquared(model.model_obj)
                mdf_L.marginal_r2 = r2.Marginal[0]
                mdf_L.conditional_r2 = r2.Conditional[0]

                mdf_L.save(HLTP_pupil.result_dir + 
                           '/LM_linear_full_rand_stats_' + str(roi)
                           + fband + b + '.pkl')
                    
                mdf_Q.save(HLTP_pupil.result_dir + 
                           '/LM_quadratic_full_rand_stats_' + str(roi)
                           + fband + b + '.pkl')





                # re-fit the best model with REML 
                # if mdf_Q.bic > mdf_L.bic:
                #     try:
                #         mdf_L = lmm_L.fit(method = 'powell', reml = True)
                #         mdf_L.save(HLTP_pupil.result_dir +
                #            '/bestLM_linear_full_rand_stats_' + str(roi)
                #            + fband + b + '.pkl')
                #     except:
                #         print('L reml failed for roi' + str(roi) + fband + b)
                #         pass
                # else:
                #     try:
                #         mdf_Q = lmm_Q.fit(method = 'powell', reml = True)
                #         mdf_Q.save(HLTP_pupil.result_dir +
                #            '/bestLM_full_rand_stats_' + str(roi)
                #            + fband + b + '.pkl')
                #     except:
                #         print('Q reml failed for roi' + str(roi) + fband + b)
                #         pass

    return 0

def summarize_LMM_results():  
    n_rois = 7
    for b in ['task_prestim', 'rest']:
        res = {}
        for fband in HLTP_pupil.freq_bands.keys():
            # initialize
            res[fband + 'inter'] = np.zeros(n_rois) + np.nan
            res[fband + '_LRT_pval'] = np.zeros(n_rois) + np.nan
            for term in ['L', 'Q']:
                res[fband + term] = np.zeros(n_rois) + np.nan
                res[fband + term + 'pval'] = np.zeros(n_rois) + np.nan
                res[fband + term + 'err'] = np.zeros(n_rois) + np.nan
                res[fband + term + 'bic'] = np.zeros(n_rois) + np.nan
                res[fband + term + 'conv'] = np.zeros(n_rois) + np.nan
                res[fband + term + 'marginal_r2'] = np.zeros(n_rois) + np.nan
                res[fband + term + 'conditional_r2'] = np.zeros(n_rois) + np.nan
            
            for roi in range(n_rois):
                mdf_Q = pd.read_pickle(HLTP_pupil.result_dir + 
                           '/LM_quadratic_full_rand_stats_' + str(roi)
                           + fband + b + '.pkl')  
                mdf_L = pd.read_pickle(HLTP_pupil.result_dir + 
                           '/LM_linear_full_rand_stats_' + str(roi)
                           + fband + b + '.pkl')
                LR = 2 * (mdf_Q.llf - mdf_L.llf)
                DOF_diff = mdf_Q.df_modelwc - mdf_L.df_modelwc
                p = chi2.sf(LR, DOF_diff)

                res[fband + '_LRT_pval'][roi] = p
                res[fband + 'Qbic'][roi] = mdf_Q.bic
                res[fband + 'Lbic'][roi] = mdf_L.bic
                res[fband + 'Qconv'][roi] = mdf_Q.converged
                res[fband + 'Lconv'][roi] = mdf_L.converged

                if (mdf_Q.bic < mdf_L.bic) & (mdf_Q.pvalues[1] < 0.05):
                    # quadratic model is better according to bic and Q parameter is significant
                    #try:
                    #    mdf_Q = pd.read_pickle(HLTP_pupil.result_dir +
                    #       '/bestLM_full_rand_stats_' + str(roi)
                    #       + fband + b + '.pkl')
                    #except:
                    #    print("best model didn't fit w REML, use the nonREML fit")
                    res[fband + 'inter'][roi] = mdf_Q.fe_params[0]
                    res[fband + 'Q'][roi] = mdf_Q.fe_params[1]
                    res[fband + 'L'][roi] = mdf_Q.fe_params[2]
                    res[fband + 'Qerr'][roi] = mdf_Q.bse[1]
                    res[fband + 'Lerr'][roi] = mdf_Q.bse[2]
                    res[fband + 'Qpval'][roi] = mdf_Q.pvalues[1]
                    res[fband + 'Lpval'][roi] = mdf_Q.pvalues[2]
                    res[fband + 'Qmarginal_r2'][roi] = mdf_Q.marginal_r2
                    res[fband + 'Qconditional_r2'][roi] = mdf_Q.conditional_r2


                else: # linear model is better
                    #try:
                    #    mdf_L = pd.read_pickle(HLTP_pupil.result_dir +
                    #       '/bestLM_linear_full_rand_stats_' + str(roi)
                    #       + fband + b + '.pkl')
                    #except:
                    #    print("best model didn't fit w REML, use the nonREML fit")
                    res[fband + 'inter'][roi] = mdf_L.fe_params[0]
                    res[fband + 'L'][roi] = mdf_L.fe_params[1]
                    res[fband + 'Lerr'][roi] = mdf_L.bse[1]
                    res[fband + 'Lpval'][roi] = mdf_L.pvalues[1]
                    res[fband + 'Lmarginal_r2'][roi] = mdf_L.marginal_r2
                    res[fband + 'Lconditional_r2'][roi] = mdf_L.conditional_r2

        # correct for multiple comparisons across rois
        for fband in HLTP_pupil.freq_bands.keys():
            not_nan = ~np.isnan(res[fband + 'Qpval'])
            res[fband + 'Qpval_corrected'] = res[fband + 'Qpval']
            if sum(not_nan) > 1:
                res[fband + 'Qpval_corrected'][not_nan] = multipletests(
                    res[fband + 'Qpval'][not_nan], method = 'fdr_bh')[1]

            not_nan = ~np.isnan(res[fband + 'Lpval'])
            res[fband + 'Lpval_corrected'] = res[fband + 'Lpval']
            if sum(not_nan) > 1:
                res[fband + 'Lpval_corrected'][not_nan] = multipletests(
                    res[fband + 'Lpval'][not_nan], method = 'fdr_bh')[1]
        print(res)
        pd.DataFrame(res).to_pickle(
                HLTP_pupil.result_dir + '/LM_betas_full_random_' + b + '.pkl')
        
pupil_roi_power_df()
fitLM_for_roi_pwr()
summarize_LMM_results()

