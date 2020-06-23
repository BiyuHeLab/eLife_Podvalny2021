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

Qall_df = []; Lall_df = []
vtype = 'pupil_'
kwords = ['BHV', 'Q_pe', 'Q_pval', 'Q_std', 
          'L_pe', 'L_pval', 'L_std', 'aic', 'bic']
for bhv_var in ['HR', 'FAR', 'c', 'd', 'p_corr', 'catRT']:
        mdf_Q = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir +
                             '/results/mixedlmQ_' + vtype + bhv_var +'.pkl')
        print(mdf_Q.pvalues, mdf_Q.params)
        mdf_L = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir +
                             '/results/mixedlmL_' + vtype + bhv_var +'.pkl')
        print(mdf_L.pvalues, mdf_L.params)
        Qres_dict = { i : [] for i in kwords }

        Qres_dict['BHV'] = [bhv_var]
        Qres_dict['Q_pe']   = "{:.3f}".format(mdf_Q.params[1])
        Qres_dict['Q_pval'] = "{:.3f}".format(round(mdf_Q.pvalues[1], 3))
        Qres_dict['Q_std']  = "{:.3f}".format(mdf_Q.bse[1])
        Qres_dict['L_pe']   = "{:.3f}".format(mdf_Q.params[2])
        Qres_dict['L_pval'] = "{:.3f}".format(mdf_Q.pvalues[2])
        Qres_dict['L_std']  = "{:.3f}".format(mdf_Q.bse[2])
        Qres_dict['aic']    = "{:.3f}".format(aic(
                mdf_Q.llf, mdf_Q.nobs, mdf_Q.df_modelwc))
        Qres_dict['bic']    = "{:.3f}".format(bic(
                mdf_Q.llf, mdf_Q.nobs, mdf_Q.df_modelwc))
        Qall_df.append(pd.DataFrame.from_dict(Qres_dict.copy(), 
                                                 orient = 'columns'))
        Lres_dict = { i : [np.NaN] for i in kwords }

        Lres_dict['BHV'] = [bhv_var]
        Lres_dict['L_pe']   = "{:.3f}".format(mdf_L.params[1])
        Lres_dict['L_pval'] = "{:.3f}".format(mdf_L.pvalues[1])
        Lres_dict['L_std']  = "{:.3f}".format(mdf_L.bse[1])
        Lres_dict['aic']    = "{:.3f}".format(aic(
                mdf_L.llf, mdf_L.nobs, mdf_L.df_modelwc))
        Lres_dict['bic']    = "{:.3f}".format(bic(
                mdf_L.llf, mdf_L.nobs, mdf_L.df_modelwc))
        Lall_df.append(pd.DataFrame.from_dict(Lres_dict.copy(), 
                                                 orient = 'columns'))
        
(pd.concat(Lall_df) ).to_csv(HLTP_pupil.result_dir + '/Pupil_L_stats_file.csv')  
(pd.concat(Qall_df) ).to_csv(HLTP_pupil.result_dir + '/Pupil_Q_stats_file.csv')          