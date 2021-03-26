#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:29:59 2020

For each trial we see if the stim was presented during dilation or
during constriction

@author: podvae01
"""
import HLTP_pupil
from HLTP_pupil import MEG_pro_dir, subjects
import numpy as np
import scipy
import scipy.stats
from scipy.stats import sem
import pandas as pd
import mne
from os import path
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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

bhv_df = pd.read_pickle(HLTP_pupil.result_dir + '/all_subj_bhv_df.pkl')
#df_name = 'all_subj_bhv_w_pupil_power'# dataframe with prestimulus power
#bhv_df = pd.read_pickle(HLTP_pupil.result_dir + '/' + df_name + '.pkl')
bhv_vars = ['HR', 'FAR', 'c', 'd', 'p_correct', 'catRT', 'm_pupil']
b = 'task_prestim'
bhv_result = {}
for state  in ['con', 'dil']:
    for slope in ['']:
        for v in bhv_vars:
            bhv_result[slope + v + state] = []

for sub_id, subject  in enumerate(subjects):
    #load epochs
    filename = MEG_pro_dir + '/' + subject + '/' + b + '_ds' + '-epo.fif'
    if not path.exists(filename): print('No such file'); continue
    epochs = mne.read_epochs(filename)
    #identify pupil dilaiton or constriction
    #note that the filter here is on prestim data - no risk of phasic leak 
    epochs.filter(0, 5, picks = HLTP_pupil.pupil_chan)
    pupil_data = epochs.get_data()[:, HLTP_pupil.pupil_chan, :]
    n_epochs = pupil_data.shape[0]
    slopes = np.zeros(n_epochs); R2 = np.zeros(n_epochs)
    for e in range(n_epochs):
        X = pupil_data[e, -np.int(0.1 * HLTP_pupil.resamp_fs):]
        model = LinearRegression().fit(np.arange(len(X)).reshape(-1, 1), X)
        R2[e] = model.score(np.arange(len(X)).reshape(-1, 1), X)
        slopes[e] = model.coef_
    # calculate 
    slopes[R2 < 0.1] = np.nan
    #trials = {'fast':[], 'slow':[], 'con':[], 'dil':[]}
    trials = {'con':[], 'dil':[]}
    qc = np.quantile(slopes[~np.isnan(slopes) & (slopes < 0)], 0.5)
    qd = np.quantile(slopes[~np.isnan(slopes) & (slopes > 0)], 0.5)
    trials['con'] = np.where(slopes < qc)[0]
    trials['dil'] = np.where(slopes > qd)[0]
    for state  in ['con', 'dil']:
        
        tr  = trials[state]
        #for idx, tr  in enumerate([slow_trials, fast_trials]):
        sub_df = bhv_df[bhv_df.subject == subject].loc[tr]
        HR, FAR, d, c, p_correct, catRT, _ = sdt_from_df(sub_df)
        bhv_result['HR' + state].append(HR)
        bhv_result['FAR' + state].append(FAR)
        bhv_result['c' + state].append(c)
        bhv_result['d' + state].append(d)
        bhv_result['p_correct' + state].append(p_correct)
        bhv_result['catRT' + state].append(catRT)
        bhv_result['m_pupil' + state].append(
                pupil_data[tr, :].mean(axis = 0))

HLTP_pupil.save(bhv_result, HLTP_pupil.MEG_pro_dir + '/pupil_result' +
                                '/fast_event_bhv.pkl')

