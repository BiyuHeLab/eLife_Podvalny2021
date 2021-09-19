#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:24:42 2020

@author: podvae01
"""
import HLTP_pupil
import pandas as pd
import numpy as np
from statsmodels.tools.eval_measures import bic, aic

vtype = 'pupil_'

def prep_stats_dfs(vtype):
    Qall_df = []; Lall_df = []

    kwords = ['BHV', 'Q_pe', 'Q_pval', 'Q_std',
              'L_pe', 'L_pval', 'L_std', 'aic', 'bic']
    for bhv_var in ['HR', 'FAR', 'c', 'd', 'p_corr', 'catRT']:
            mdf_Q = pd.read_pickle(HLTP_pupil.result_dir +
                                 '/mixedlmQ_' + vtype + bhv_var +'.pkl')

            mdf_L = HLTP_pupil.load(HLTP_pupil.result_dir +
                                 '/mixedlmL_' + vtype + bhv_var +'.pkl')
            Qres_dict = { i : [] for i in kwords }

            Qres_dict['BHV'] = [bhv_var]
            Qres_dict['Q_pe']   = "{:.3f}".format(round(mdf_Q.params[1], 3))
            Qres_dict['Q_pval'] = "{:.3f}".format(round(mdf_Q.pvalues[1], 3))
            Qres_dict['Q_std']  = "{:.3f}".format(round(mdf_Q.bse[1], 3))
            Qres_dict['L_pe']   = "{:.3f}".format(round(mdf_Q.params[2], 3))
            Qres_dict['L_pval'] = "{:.3e}".format(round(mdf_Q.pvalues[2], 3))
            Qres_dict['L_std']  = "{:.3f}".format(round(mdf_Q.bse[2], 3))
            Qres_dict['aic']    = "{:.3f}".format(round(mdf_Q.aic, 3))
            Qres_dict['bic']    = "{:.3f}".format(round(mdf_Q.bic, 3))
            Qres_dict['marg_r2'] = "{:.3f}".format(round(mdf_Q.marginal_r2, 3))
            Qres_dict['cond_r2'] = "{:.3f}".format(round(mdf_Q.conditional_r2, 3))
            Qall_df.append(pd.DataFrame.from_dict(Qres_dict.copy(),
                                                     orient = 'columns'))



            Lres_dict = { i : [np.NaN] for i in kwords }

            Lres_dict['BHV'] = [bhv_var]
            Lres_dict['L_pe']   = "{:.3f}".format(round(mdf_L.params[1], 3))
            Lres_dict['L_pval'] = "{:.3f}".format(round(mdf_L.pvalues[1], 3))
            Lres_dict['L_std']  = "{:.3f}".format(round(mdf_L.bse[1], 3))
            Lres_dict['aic']    = "{:.3f}".format(round(mdf_L.aic, 3))
            Lres_dict['bic']    = "{:.3f}".format(round(mdf_L.bic, 3))
            Lres_dict['marg_r2'] = "{:.3f}".format(round(mdf_L.marginal_r2, 3))
            Lres_dict['cond_r2'] = "{:.3f}".format(round(mdf_L.conditional_r2, 3))
            Lall_df.append(pd.DataFrame.from_dict(Lres_dict.copy(),
                                                     orient = 'columns'))
    return Qall_df, Lall_df

# print pupil-linked behavior to file    
vtype = 'pupil_'
Qall_df, Lall_df = prep_stats_dfs(vtype)
(pd.concat(Lall_df) ).to_csv(HLTP_pupil.result_dir +
                                 '/Pupil_L_stats_' + vtype + '.csv')
(pd.concat(Qall_df) ).to_csv(HLTP_pupil.result_dir +
                                 '/Pupil_Q_stats_' + vtype + '.csv')

#print power-linked behavior to file
n_roi = 7
Lall_df = []; Qall_df = []
for roi in range(n_roi):
    for fband in HLTP_pupil.freq_bands.keys():
        IV = fband + str(roi) + 'res_' # res indicates residual
        Q_list, L_list = prep_stats_dfs(IV)
        Ldf = pd.concat(L_list); Ldf.index = [fband + str(roi)] * len(Ldf)
        Qdf = pd.concat(Q_list); Qdf.index = [fband + str(roi)] * len(Qdf)
        Lall_df.append(Ldf); Qall_df.append(Qdf)
        
(pd.concat(Lall_df) ).to_csv(HLTP_pupil.result_dir + 
                                 '/Res_Power_L_stats_file_' + vtype + '.csv')  
(pd.concat(Qall_df) ).to_csv(HLTP_pupil.result_dir + 
                                 '/Res_Power_Q_stats_file_' + vtype + '.csv')        
        