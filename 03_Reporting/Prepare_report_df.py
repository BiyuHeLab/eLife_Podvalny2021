#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 18:25:51 2020

I want this folder to contain source data for main figures, reported
as dataframes or csv data

@author: podvae01
"""
import HLTP_pupil
import pandas as pd
import numpy as np
from scipy.stats import zscore
RSNs = ['Vis','SM','DAN','VAN','Lim','FPN', 'DMN']

def prepare_fig1_csv():
    # for figure 1F, to generate the results use 01_pupil_timecourse_analysis
    data_dict = {}
    for s, subject in enumerate(HLTP_pupil.subjects):
        [f, Pxx_den] = HLTP_pupil.load(HLTP_pupil.result_dir + '/Pupil_PSD_' +
                                       subject + '.pkl')
        data_dict['Frequency'] = f
        data_dict['subject_' + subject] = Pxx_den

    pd.DataFrame(data_dict).to_csv(HLTP_pupil.result_dir + '/Source_CSV_files/fig1F.csv')

def prepare_fig3B_csv():
    b  = 'task_prestim'
    df = pd.read_pickle(HLTP_pupil.result_dir +
                            '/roi_pwr_and_pupil_blocknorm_' + b + '.pkl')
    df.pupil = zscore(df.pupil)
    df.index.name = 'trial'
    for rn in range(7):
        for freq_band in HLTP_pupil.freq_bands.keys():
            df = df.rename(columns = {freq_band + str(rn) : freq_band + '_' + RSNs[rn]})

    df.to_csv(HLTP_pupil.result_dir + '/Source_CSV_files/fig3B.csv')

def prepare_supp_fig1_csv():
    b  = 'rest'
    df = pd.read_pickle(HLTP_pupil.result_dir +
                            '/roi_pwr_and_pupil_blocknorm_' + b + '.pkl')
    df.pupil = zscore(df.pupil)
    df.index.name = 'trial'
    for rn in range(7):
        for freq_band in HLTP_pupil.freq_bands.keys():
            df = df.rename(columns = {freq_band + str(rn) : freq_band + '_' + RSNs[rn]})
    df.to_csv(HLTP_pupil.result_dir + '/Source_CSV_files/supp_fig1.csv')

def prepare_fig4CD():
    df = pd.read_pickle(HLTP_pupil.result_dir +
                            '/sdt_by_pupil.pkl')
    df.to_csv(HLTP_pupil.result_dir + '/Source_CSV_files/fig4CD.csv')

def prepare_fig4E():
    scores, perm_scores = HLTP_pupil.load(HLTP_pupil.result_dir +
                                          '/cat_decod_score.pkl')
    df = pd.DataFrame({'subjectN':np.indices(scores.shape)[2].flatten(),
          'pupil_group':np.indices(scores.shape)[0].flatten(),
          'time_point':np.indices(scores.shape)[1].flatten(),
          'score':scores.flatten()})

    df.to_csv(HLTP_pupil.result_dir +
                                '/Source_CSV_files/fig4E.csv')

def prepare_fig5A():
    scores, coefs, mean_perm, pop_pval = HLTP_pupil.load(
        HLTP_pupil.result_dir +'/predict_rec_from_all')
    pd.DataFrame(scores).to_csv(HLTP_pupil.result_dir +
                                '/Source_CSV_files/fig5A.csv')

def prepare_fig5B():
    dic = {}
    labels = np.zeros( (7, 5) ).astype(str)
    for r_i, RSN in enumerate(RSNs):
        for p_i, pwr in enumerate(HLTP_pupil.freq_bands.keys()):
            labels[r_i, p_i] = pwr + '_' + RSN
    dic['Frequency_RNS'] = labels.flatten()
    for bhv_var in ['HR', 'FAR', 'c', 'd', 'p_corr', 'catRT']:
        [table_par_e, table_pvals, table_pvals_corr] = HLTP_pupil.load(
            HLTP_pupil.result_dir +
                        '/respe_pval_' + bhv_var +'.pkl')
        dic[bhv_var] = table_par_e.flatten()

    pd.DataFrame(dic).to_csv(HLTP_pupil.result_dir +
                                '/Source_CSV_files/fig5B.csv')

def prepare_fig6():
    bhv_result = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir + '/pupil_result' +
                                '/fast_event_bhv.pkl')
    bhv_result.pop('m_pupilcon', None); bhv_result.pop('m_pupildil', None)
    pd.DataFrame(bhv_result).to_csv(HLTP_pupil.result_dir +
                                '/Source_CSV_files/fig6E.csv')











# report the result of the effect of residual power on behavior

bhv_vars = ['HR', 'FAR', 'c', 'd', 'p_corr', 'catRT']
RSNs = ['Vis','SM','DAN','VAN','Lim','FPN', 'DMN']
kwords = ['fband', 'roi', 'Coef', 'pval', 'std',
          'bic']
n_roi = 7
savetag = 'res'
all_df = []
idx = 0
for bhv_var in bhv_vars:
    for bi, band in enumerate(HLTP_pupil.freq_bands.keys()):
        for roi in range(7):
            LM_res_dict = { i : [] for i in kwords }
            idx += 1

            IV =  band + str(roi)
            save_name = IV + savetag + '_' + bhv_var
            mdf = pd.read_pickle(
                HLTP_pupil.result_dir + '/mixedlmL_' + save_name +'.pkl')

            LM_res_dict['fband']  = band
            LM_res_dict['roi']    = RSNs[roi]
            LM_res_dict['BHV'] = [bhv_var]
            LM_res_dict['Coef']  = "{:.3f}".format(mdf.params[2])
            LM_res_dict['pval'] = "{:.3f}".format(mdf.pvalues[2])
            LM_res_dict['std']  = "{:.3f}".format(mdf.bse[2])
            LM_res_dict['bic']  = "{:.3f}".format(mdf.bic)
            all_df.append(pd.DataFrame(LM_res_dict.copy(), index = [idx]))

(pd.concat(all_df) ).to_csv(HLTP_pupil.result_dir + '/LM_respower_bhv_stats_file.csv')
