#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:39:45 2020

Print linear model results tables for the models of pupil size as a function 
of spectral power

@author: podvae01
"""
import pandas as pd
import HLTP_pupil
from statsmodels.tools.eval_measures import bic, aic

n_rois = 7
kwords = ['exp', 'fband', 'roi', 'Q_pe', 'Q_pval', 'Q_std', 
          'L_pe', 'L_pval', 'L_std', 'aic', 'bic']
all_df = []
for b in ['task_prestim', 'rest']:
    for fband in HLTP_pupil.freq_bands.keys():
        for roi in range(n_rois):
            mdf_Q = pd.read_pickle(HLTP_pupil.result_dir + '/LM_stats_'
                                   + str(roi) + fband + b + '.pkl')

            LM_res_dict = { i : [] for i in kwords }
            LM_res_dict['exp']    = b
            LM_res_dict['fband']  = fband
            LM_res_dict['roi']    = [roi]
            LM_res_dict['Q_pe']   = "{:.2e}".format(mdf_Q.params[1])
            LM_res_dict['Q_pval'] = "{:.2e}".format(round(mdf_Q.pvalues[1], 3))
            LM_res_dict['Q_std']  = "{:.2e}".format(mdf_Q.bse[1])
            LM_res_dict['L_pe']   = "{:.2e}".format(mdf_Q.params[2])
            LM_res_dict['L_pval'] = "{:.2e}".format(mdf_Q.pvalues[2])
            LM_res_dict['L_std']  = "{:.2e}".format(mdf_Q.bse[2])
            LM_res_dict['aic']    = "{:.2e}".format(aic(
                mdf_Q.llf, mdf_Q.nobs, mdf_Q.df_modelwc))
            LM_res_dict['bic']    = "{:.2e}".format(bic(
                mdf_Q.llf, mdf_Q.nobs, mdf_Q.df_modelwc))
            
            all_df.append(pd.DataFrame.from_dict(LM_res_dict.copy(), 
                                                 orient = 'columns'))
            
(pd.concat(all_df) ).to_csv(HLTP_pupil.result_dir + '/LM_stats_file.csv')