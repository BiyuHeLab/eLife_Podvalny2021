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
df_name = 'all_subj_bhv_w_pupil_power'# dataframe with prestimulus power 
bhv_df = pd.read_pickle(HLTP_pupil.result_dir + '/' + df_name + '.pkl')

b = 'task_prestim'
bhv_result = {}
for state  in ['con', 'dil']:
    for slope in ['']:
        for v in ['HR', 'FAR', 'c', 'd', 'p_correct', 'catRT', 'm_pupil']:
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
    n_epochs = pupil_data.shape[0];
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
    trials['con'] = np.where(slopes < qc)[0];
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

y_lims = {'HR':[-.2, 1], 'FAR':[-.2, 1], 'c':[-.5, 2.], 'd':[-.5, 2.5],
          'p_correct':[0.2, 0.8], 'catRT':[0.6, 1.6]}
for v in ['HR', 'FAR', 'c', 'd', 'p_correct', 'catRT']:
    data = [np.array(bhv_result[ v + 'con']), 
         np.array(bhv_result[ v + 'dil'])]
    print(scipy.stats.wilcoxon(data[0], data[1]))
    fig, ax = plt.subplots(1, 1, figsize = (0.8, 1.5))
    
    ax.spines['left'].set_position(('outward', 10))
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 15))
    ax.xaxis.set_ticks_position('bottom')
    box1 = plt.boxplot(data, positions = [0,1], patch_artist = True, 
                       widths = 0.8,showfliers=False,
             boxprops=None,    showbox=None,     whis = 0, showcaps = False)
    
    box1['boxes'][0].set( facecolor = 'c', lw=0, zorder=0, alpha =0.1)
    box1['boxes'][1].set( facecolor = 'r', lw=0, zorder=0, alpha =0.1)
    
    box1['medians'][0].set( color = 'c', lw=2, zorder=20)
    box1['medians'][1].set( color =  'r', lw=2, zorder=20)
    plt.plot([0, 1], data, 
             color = [.5, .5, .5], lw = 0.5);
    plt.plot([0], [data[0]], 'o', 
             markerfacecolor = [.9, .9, .9], color = 'c', 
             alpha = 1.);    
    plt.plot([1], [data[1]], 'o',
             markerfacecolor = [.9, .9, .9], color = 'r', alpha = 1.);          
    plt.locator_params(axis='y', nbins=6)
    plt.ylim(y_lims[v]);plt.xlim([-.4, 1.4]);

    plt.ylabel(v); 
    fig.savefig(figures_dir + '/bhv_convsdil_' + v +
                '.png', dpi = 800, bbox_inches = 'tight', transparent = True)
    
    # fig, ax = plt.subplots(1, 1, figsize = (2.5,2.5))    
    # colors = {'dil':'r', 'con':'c'}

    # for state in ['con', 'dil']:        
    #     m = np.mean(bhv_result['m_pupil' + state], axis = 0)
    #     e = np.std(bhv_result['m_pupil' + state], 
    #                axis = 0) / np.sqrt(24)
    #     ax.fill_between(epochs.times, m + e, m-e, color = colors[state],
    #                     alpha = 0.5, edgecolor='none', linewidth=0, 
    #                     label = slope + ' ' + state)
    # plt.xlim([-1., 0.]);   
    # plt.ylim([-.3, .2])   
    # plt.xlabel('time (s)'); plt.ylabel('pupil size (s.d.)')         
    # fig.savefig(figures_dir + '/pre_stim_pupil_' + b + slope + 
    #                 '.png', dpi = 800, bbox_inches = 'tight', transparent = True)
                
# np.mean(bhv_result['slowdcon'])
# np.mean(bhv_result['slowddil'])
# #pupil_data = zscore(pupil_data, axis = 1)
# fig, ax = plt.subplots(1, 1, figsize = (2.5,2.5))    
# colors = {'fastdil':'r', 'fastcon':'b', 'slowcon':'c', 'slowdil':'m'}
# for slope in ['fast', 'slow']:
        
# plt.legend(position = 'outside')




# plt.plot(epochs.times,
#         (pupil_data[con_trials].T).mean(axis = 1), 
#         'r', alpha = .8);
# plt.plot(epochs.times,
#         (pupil_data[dil_trials].T).mean(axis = 1), 
#         'c', alpha = .8);


# plt.plot(epochs.times,
#         (pupil_data[np.intersect1d( con_trials, slow_trials)].T).mean(axis = 1), 
#         'r:', alpha = .8);
# plt.plot(epochs.times,
#         (pupil_data[np.intersect1d( dil_trials, slow_trials)].T).mean(axis = 1), 
#         'c:', alpha = .8);
# #plt.xlim([-.5, 0]); plt.ylim([-6, 6])

# plt.plot(epochs.times,
#         (pupil_data[np.intersect1d( con_trials, fast_trials)].T).mean(axis = 1), 
#         'r', alpha = .8);
# plt.plot(epochs.times,
#         (pupil_data[np.intersect1d( dil_trials, fast_trials)].T).mean(axis = 1), 
#         'c', alpha = .8);
# plt.xlim([-.5, 0]); #plt.ylim([-4, 4])
