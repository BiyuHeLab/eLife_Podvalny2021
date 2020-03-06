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
from scipy.optimize import curve_fit
import scipy.stats
from scipy.stats import sem
import pandas as pd
import mne
from os import path
from sklearn.linear_model import LinearRegression

def sdt_from_df(df):
    n_rec_real = sum(df[df.real_img == True].recognition == 1)
    n_real = len(df[(df.real_img == True) & (df.recognition != 0)])
    n_rec_scr = sum(df[df.real_img == False].recognition == 1)
    n_scr = len(df[(df.real_img == False) & (df.recognition != 0)])
        
    HR, FAR, d, c = get_sdt_msr(n_rec_real, n_real, n_rec_scr, n_scr)
    return HR, FAR, d, c
        
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

bhv_df = pd.read_pickle(MEG_pro_dir + '/results/all_subj_bhv_df.pkl')
b = 'task_prestim'
bhv_result = {}
for state  in ['con', 'dil']:
    for slope in ['']:
        for v in ['HR', 'FAR', 'c', 'd', 'm_pupil']:
            bhv_result[slope + v + state] = []

for sub_id, subject  in enumerate(subjects):
    #load epochs
    filename = MEG_pro_dir + '/' + subject + '/' + b + '_ds' + '-epo.fif'
    if not path.exists(filename): print('No such file'); continue
    epochs = mne.read_epochs(filename)
    #identify pupil dilaiton or constriction
    epochs.filter(0, 5, picks = HLTP_pupil.pupil_chan)
    pupil_data = epochs.get_data()[:, HLTP_pupil.pupil_chan, :]
    n_epochs = pupil_data.shape[0]; 
    slopes = np.zeros(n_epochs); R2 = np.zeros(n_epochs)
    for e in range(n_epochs):
        X = pupil_data[e, -np.int(0.15 * HLTP_pupil.resamp_fs):]
        model = LinearRegression().fit(np.arange(len(X)).reshape(-1, 1), X)
        R2[e] = model.score(np.arange(len(X)).reshape(-1, 1), X)
        slopes[e] = model.coef_
    # calculate 
    #slopes[R2 < 0.1] = np.nan
    #trials = {'fast':[], 'slow':[], 'con':[], 'dil':[]}
    trials = {'con':[], 'dil':[]}
    trials['con'] = np.where(slopes < 0)[0];
    trials['dil'] = np.where(slopes > 0)[0]
    for state  in ['con', 'dil']:
        
        tr  = trials[state]
        #for idx, tr  in enumerate([slow_trials, fast_trials]):
        sub_df = bhv_df[bhv_df.subject == subject].loc[tr]
        HR, FAR, d, c = sdt_from_df(sub_df)
        bhv_result['HR' + state].append(HR)
        bhv_result['FAR' + state].append(FAR)
        bhv_result['c' + state].append(c)
        bhv_result['d' + state].append(d)
        bhv_result['m_pupil' + state].append(
                pupil_data[tr, :].mean(axis = 0))

    
    
    speed = np.zeros( slopes.shape )
    speed[~np.isnan(slopes)] = np.digitize(np.abs(slopes[~np.isnan(slopes)]), 
                        np.percentile(np.abs(slopes[~np.isnan(slopes)]), [0., 50.]))
    trials['fast'] = np.where(speed == 2)[0]; 
    trials['slow'] = np.where(speed == 1)[0]
    for state  in ['con', 'dil']:
        for slope in ['fast', 'slow']:
            tr  = np.intersect1d( trials[state], trials[slope])
            #for idx, tr  in enumerate([slow_trials, fast_trials]):
            sub_df = bhv_df[bhv_df.subject == subject].loc[tr]
            HR, FAR, d, c = sdt_from_df(sub_df)
            bhv_result[slope + 'HR' + state].append(HR)
            bhv_result[slope + 'FAR' + state].append(FAR)
            bhv_result[slope + 'c' + state].append(c)
            bhv_result[slope + 'd' + state].append(d)
            bhv_result[slope + 'm_pupil' + state].append(
                    pupil_data[tr, :].mean(axis = 0))
  
for v in ['HR', 'FAR', 'c', 'd']:
    for slope in ['']:
        fig, ax = plt.subplots(1, 1, figsize = (1.5,2.5))    
        d = np.array(bhv_result[slope + v + 'con']) \
            - np.array(bhv_result[slope + v + 'dil'])
        plt.bar('con', np.array(bhv_result[slope + v + 'con']).mean(),
                facecolor = 'c', alpha = 0.5)   
        plt.bar('dil', np.array(bhv_result[slope + v + 'dil']).mean(),
                facecolor = 'r', alpha = 0.5)   
        plt.ylabel(v); #plt.ylim([0., .4])
        print(slope, v, scipy.stats.ttest_1samp( d, 0))
        fig.savefig(figures_dir + '/bhv_convsdil_' + b + slope + v +
                    '.png', dpi = 800, bbox_inches = 'tight', transparent = True)
    
    fig, ax = plt.subplots(1, 1, figsize = (2.5,2.5))    
    colors = {'dil':'r', 'con':'c'}

    for state in ['con', 'dil']:
        
        m = np.mean(bhv_result['m_pupil' + state], axis = 0)
        e = np.std(bhv_result['m_pupil' + state], 
                   axis = 0) / np.sqrt(24)
        ax.fill_between(epochs.times, m + e, m-e, color = colors[state],
                        alpha = 0.5, edgecolor='none', linewidth=0, 
                        label = slope + ' ' + state)
    plt.xlim([-1., 0.]);   
    plt.ylim([-.3, .2])   
    plt.xlabel('time (s)'); plt.ylabel('pupil size (s.d.)')         
    fig.savefig(figures_dir + '/pre_stim_pupil_' + b + slope + 
                    '.png', dpi = 800, bbox_inches = 'tight', transparent = True)
                
np.mean(bhv_result['slowdcon'])
np.mean(bhv_result['slowddil'])
#pupil_data = zscore(pupil_data, axis = 1)
fig, ax = plt.subplots(1, 1, figsize = (2.5,2.5))    
colors = {'fastdil':'r', 'fastcon':'b', 'slowcon':'c', 'slowdil':'m'}
for slope in ['fast', 'slow']:
        
plt.legend(position = 'outside')




plt.plot(epochs.times,
        (pupil_data[con_trials].T).mean(axis = 1), 
        'r', alpha = .8);
plt.plot(epochs.times,
        (pupil_data[dil_trials].T).mean(axis = 1), 
        'c', alpha = .8);


plt.plot(epochs.times,
        (pupil_data[np.intersect1d( con_trials, slow_trials)].T).mean(axis = 1), 
        'r:', alpha = .8);
plt.plot(epochs.times,
        (pupil_data[np.intersect1d( dil_trials, slow_trials)].T).mean(axis = 1), 
        'c:', alpha = .8);
#plt.xlim([-.5, 0]); plt.ylim([-6, 6])

plt.plot(epochs.times,
        (pupil_data[np.intersect1d( con_trials, fast_trials)].T).mean(axis = 1), 
        'r', alpha = .8);
plt.plot(epochs.times,
        (pupil_data[np.intersect1d( dil_trials, fast_trials)].T).mean(axis = 1), 
        'c', alpha = .8);
plt.xlim([-.5, 0]); #plt.ylim([-4, 4])
