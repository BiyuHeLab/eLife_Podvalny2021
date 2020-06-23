#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:06:29 2020

@author: podvae01
"""
import mne
import numpy as np
import HLTP_pupil
from HLTP_pupil import subjects, freq_bands, MEG_pro_dir, FS_dir
from os import path  
from mne.beamformer import make_dics, apply_dics_csd
from mne.time_frequency import csd_multitaper
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import zscore 
import os
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
            roi_data = HLTP_pupil.load(sub_pro_dir + 
                                       '/roi_single_epoch_dics_power_' + 
                                   b + '_ds')
            for roi in range(7):# for each roi and each frequency band
                for fband, frange in HLTP_pupil.freq_bands.items():
                    subj_df[fband + str(roi)] = (np.log(roi_data[roi, 
                        (frange[0] - 1) : (frange[1] - 1), :]).mean(axis = 0))
            dfs.append(subj_df)  
        df = pd.concat(dfs)
        # save df
        df.to_pickle(MEG_pro_dir + 
                            '/results/roi_pwr_and_pupil' + b + '.pkl')
    # combine the two rest runs
    rest1_df = pd.read_pickle(MEG_pro_dir + 
                            '/results/roi_pwr_and_pupilrest01.pkl')
    rest2_df = pd.read_pickle(MEG_pro_dir + 
                            '/results/roi_pwr_and_pupilrest02.pkl')
    pd.concat([rest1_df, rest2_df]).to_pickle(MEG_pro_dir + 
                            '/results/roi_pwr_and_pupilrest.pkl')
# fit linear mixed effect models to   
def fitLM_for_roi_pwr():
    n_rois = 7
    for b in ['task_prestim', 'rest']:
        
        df = pd.read_pickle(MEG_pro_dir + 
                            '/results/roi_pwr_and_pupil' + b + '.pkl')
        # initialize the results dictionary
        res = {}; 
        for fband in HLTP_pupil.freq_bands.keys():
            res[fband + 'inter'] = np.zeros(n_rois);
            for term in ['L', 'Q']:
                res[fband + term] = np.zeros(n_rois);
                res[fband + term + 'pval'] = np.zeros(n_rois)
                res[fband + term + 'err'] = np.zeros(n_rois)
        # fit a model for each roi and frequency band    
        for fband in HLTP_pupil.freq_bands.keys():
            for roi in range(n_rois):
                mdf_Q = smf.mixedlm(fband + str(roi) + 
                                    " ~ np.power(pupil, 2) + pupil", 
                            df.dropna(), groups = df.dropna()["subject"]
                            ).fit(method='powell')
                mdf_Q.save(HLTP_pupil.result_dir + '/LM_stats_' + str(roi)
                           + fband + b + '.pkl')
                
                print(mdf_Q.summary())
                
                res[fband + 'inter'][roi] = mdf_Q.params[0].copy();
                res[fband + 'Q'][roi] = mdf_Q.params[1].copy();
                res[fband + 'L'][roi] = mdf_Q.params[2].copy();
                res[fband + 'Qerr'][roi] = mdf_Q.bse[1].copy();
                res[fband + 'Lerr'][roi] = mdf_Q.bse[2].copy();
                res[fband + 'Qpval'][roi] = mdf_Q.pvalues[1].copy();
                res[fband + 'Lpval'][roi] = mdf_Q.pvalues[2].copy();
        # correct for multiple comparisons across sensors
        for fband in HLTP_pupil.freq_bands.keys():
            res[fband + 'Qpval_corrected'] = multipletests(res[fband + 'Qpval'], 
               method = 'fdr_bh')[1]
            res[fband + 'Lpval_corrected'] = multipletests(res[fband + 'Lpval'], 
               method = 'fdr_bh')[1]
        pd.DataFrame(res).to_pickle(
                HLTP_pupil.result_dir + '/LM_betas_' + b + '.pkl')
        
x
    
       