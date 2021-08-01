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

n_rois = 7
kwords = ['exp', 'fband', 'roi', 'Q_pe', 'Q_pval', 'Q_std', 
          'L_pe', 'L_pval', 'L_std', 'L_bic', 'Q_bic', 'L_marg_r2', 'Q_marg_r2', 'L_cond_r2', 'Q_cond_r2']
all_df = []
idx = 0
RSNs = ['Vis','SM','DAN','VAN','Lim','FPN', 'DMN']
for b in ['task_prestim', 'rest']:
    res = pd.read_pickle(HLTP_pupil.result_dir + '/LM_betas_full_random_' + b + '.pkl')

    for fband in HLTP_pupil.freq_bands.keys():
        for roi in range(n_rois):
            idx += 1
            LM_res_dict = { i : [] for i in kwords }
            #LM_res_dict['index']    = idx
            LM_res_dict['exp']    = b
            LM_res_dict['fband']  = fband
            LM_res_dict['roi']    = RSNs[roi]
            LM_res_dict['Q_pe']   = "{:.3f}".format(round(res[fband + 'Q'][roi], 3))
            LM_res_dict['Q_pval'] = "{:.3f}".format(round(res[fband + 'Qpval'][roi], 3))
            LM_res_dict['Q_std']  = "{:.3f}".format(round(res[fband + 'Qerr'][roi], 3))
            LM_res_dict['L_pe']   = "{:.3f}".format(round(res[fband + 'L'][roi], 3))
            LM_res_dict['L_pval'] = "{:.3f}".format(round(res[fband + 'Lpval'][roi], 3))
            LM_res_dict['L_std']  = "{:.3f}".format(round(res[fband + 'Lerr'][roi], 3))

            LM_res_dict['L_bic']    = "{:.1f}".format(round(res[fband + 'Lbic'][roi], 1))
            LM_res_dict['Q_bic']    = "{:.1f}".format(round(res[fband + 'Qbic'][roi], 1))

            LM_res_dict['L_marg_r2'] = "{:.3f}".format(round(res[fband + 'Lmarginal_r2'][roi], 3))
            LM_res_dict['L_cond_r2'] = "{:.3f}".format(round(res[fband + 'Lconditional_r2'][roi], 3))
            LM_res_dict['Q_marg_r2'] = "{:.3f}".format(round(res[fband + 'Qmarginal_r2'][roi], 3))
            LM_res_dict['Q_cond_r2'] = "{:.3f}".format(round(res[fband + 'Qconditional_r2'][roi], 3))

            if res[fband + 'Qbic'][roi] < res[fband + 'Lbic'][roi]:
                LM_res_dict['LRT_pval'] = "{:.2e}".format(res[fband + '_LRT_pval'][roi])
            all_df.append(pd.DataFrame(LM_res_dict.copy(), index = [idx]))
            
(pd.concat(all_df) ).to_csv(HLTP_pupil.result_dir + '/LM_stats_file.csv')
