#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 08:14:29 2020

In this script I analyze chnages in PSD accoridng to pupil size in 2-sec win of
rest and pre-stimulus 

@author: podvae01
"""

from os import path
import HLTP_pupil
from HLTP_pupil import MEG_pro_dir, subjects
import numpy as np
import scipy
import scipy.stats
from scipy.stats import zscore
#from scipy.stats import sem
import mne
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.eval_measures import bic, aic
from mne.stats import permutation_cluster_test
from statsmodels.stats.multitest import multipletests

def get_epochs_and_pupil(sub_pro_dir, b, subject, res = 20):
    epoch_fname = sub_pro_dir + '/' + b + '_ds-epo.fif'
        
    if not path.exists(epoch_fname):
        print('No such file'); return np.NaN, np.NaN;
        
    epochs = mne.read_epochs(epoch_fname, preload = True )
    
    pupil_states = HLTP_pupil.load(HLTP_pupil.result_dir + '/pupil_states_' + 
                            b + subject + '.pkl')
    group_percentile = np.arange(0., 100., res);
    perc = np.percentile(pupil_states.mean_pupil, group_percentile)
    p_group = np.digitize(pupil_states.mean_pupil, perc)
    pupil_group = p_group
    
    epochs.events[:, 2] = pupil_group
    return epochs, pupil_states.mean_pupil.values

def save_sensor_PSD_w_pupil(res):
    
    for b in ['task_prestim', 'rest01', 'rest02']:
        for s, subject in enumerate(subjects):
            sub_pro_dir = MEG_pro_dir + '/' + subject
            
            epochs, pupil = get_epochs_and_pupil(sub_pro_dir, b, subject, res) 
            if not (type(epochs) == mne.epochs.EpochsFIF):  continue
        
            # remove linear treand from each epoch        
            epochs._data = scipy.signal.detrend(epochs.get_data(), axis = -1)
    
            # power spectrum
            psd, freq = mne.time_frequency.psd_welch(epochs, n_fft = 2**9)
            HLTP_pupil.save([psd, freq, pupil], HLTP_pupil.result_dir + 
                        '/PSD' + b + subject +'.pkl') 

def prepare_PSD_fband_pupil_df(b): 
    dfs = []  
    for s, subject in enumerate(subjects):
        fname = HLTP_pupil.result_dir + '/PSD' + b + subject +'.pkl'
        if not path.exists(fname):
            print('No such file'); continue
        psd, freq, pupil = HLTP_pupil.load(fname) 
        
        df = pd.DataFrame({"subject" :np.repeat(subject, len(pupil)), 
                           "pupil"   :pupil})
        # calculate mean power in freq bands:
        n_chan = psd.shape[1]
        for band, frange in HLTP_pupil.freq_bands.items():                          
            band_pwr = psd[:, :, (freq > frange[0]) & (freq <= frange[1])
                    ].mean(axis = -1)
            for chan in range(n_chan):
                df[band + str(chan)] = band_pwr[:, chan]
        dfs.append(df)   
    return pd.concat(dfs)
           
def fitLM_PSD_w_pupil():
    '''fit a model Power ~ F(pupil) for each sensor, plotted in fig 2A '''
    for b in ['task_prestim', 'rest']:
        if b == 'rest':
            df1 = prepare_PSD_fband_pupil_df('rest01')
            df2 = prepare_PSD_fband_pupil_df('rest02')
            df = pd.concat([df1, df2])
        else:
            df = prepare_PSD_fband_pupil_df(b)
        n_chan = 272
        
        # initialize the results dictionary
        res = {}
        for fband in HLTP_pupil.freq_bands.keys():
            res[fband + 'inter'] = np.zeros(n_chan);
            for term in ['L', 'Q']:
                res[fband + term] = np.zeros(n_chan);
                res[fband + term + 'pval'] = np.zeros(n_chan)
        # fit a model for each sensor    
        for fband in HLTP_pupil.freq_bands.keys():
            for chan in range(n_chan):
                df[fband + str(chan)] = (np.log(df[fband + str(chan)]))
                mdf_Q = smf.mixedlm(fband + str(chan) + 
                                    " ~ np.power(pupil, 2) + pupil", 
                            df.dropna(), groups = df.dropna()["subject"]
                            ).fit(method='powell')
                res[fband + 'inter'][chan] = mdf_Q.params[0].copy();
                res[fband + 'Q'][chan] = mdf_Q.params[1].copy();
                res[fband + 'L'][chan] = mdf_Q.params[2].copy();
                res[fband + 'Qpval'][chan] = mdf_Q.pvalues[1].copy();
                res[fband + 'Lpval'][chan] = mdf_Q.pvalues[2].copy();
        # correct for multiple comparisons across sensors
        for fband in HLTP_pupil.freq_bands.keys():
            res[fband + 'Qpval_corrected'] = multipletests(res[fband + 'Qpval'], 
               method = 'fdr_bh')[1]
            res[fband + 'Lpval_corrected'] = multipletests(res[fband + 'Lpval'], 
               method = 'fdr_bh')[1]  
        pd.DataFrame(res).to_pickle(
                HLTP_pupil.result_dir + '/LM_betas_' + b + '.pkl')
        

           
def save_sensor_PSD_by_pupil(res):
    n_subj = len(subjects)
    
    for b in ['task_prestim', 'rest01', 'rest02']:
        mean_meg = []
        for s, subject in enumerate(subjects):
            sub_pro_dir = MEG_pro_dir + '/' + subject
            
            epochs, _ = get_epochs_and_pupil(sub_pro_dir, b, subject, res) 
            if not (type(epochs) == mne.epochs.EpochsFIF):  continue
        
            # remove linear treand from each epoch        
            epochs._data = scipy.signal.detrend(epochs.get_data(), axis = -1)
    
            # power spectrum
            psd, freq = mne.time_frequency.psd_welch(epochs, n_fft = 2**9)
             
            # average according to pupil groups
            pgroups = np.unique(epochs.events[:, 2])
            mean_meg.append(np.array(
                    [psd[epochs.events[:, 2] == p, :, :].mean(axis = 0)
                    for p in pgroups]))
       
        n_subj = len(mean_meg)
        mean_meg = np.array(mean_meg)
        mean_meg[:, :, :, (freq >= 58) & (freq <= 62)] = np.NaN
        for s in range(n_subj):
            mean_meg[s, :,:,:] = mean_meg[s, :,:,:
                ] / mean_meg[s, :,:,:].mean(axis = 0)
        
        mean_meg = np.log(mean_meg)
        HLTP_pupil.save([mean_meg, freq], HLTP_pupil.result_dir + 
                        '/mean_meg_by_pupil_state' + b + str(res) +'.pkl')

def update_bhv_df_w_PSD():
    #---- prepare behavioral data frame with updated power bin---------------------
    bhv_df = pd.read_pickle(HLTP_pupil.MEG_pro_dir + 
                            '/results/all_subj_bhv_df.pkl')
    # add column for each frequency band:
    for band, _ in HLTP_pupil.freq_bands.items():
        bhv_df[band] = np.NaN   
        
    b = 'task_prestim'
    
    for s, subject in enumerate(subjects):
        sub_pro_dir = MEG_pro_dir + '/' + subject
        
        epochs, _ = get_epochs_and_pupil(sub_pro_dir, b, subject) 
        if not (type(epochs) == mne.epochs.EpochsFIF):  continue
    
        # remove linear treand from each epoch        
        epochs._data = scipy.signal.detrend(epochs.get_data(), axis=-1)
    
        # power spectrum
        psd, freq = mne.time_frequency.psd_welch(epochs, n_fft = 2**9)

        if subject == 'BJB':# one last trial is missing from meg for this s
                bhv_df = bhv_df[ ~((bhv_df.index == 288) & 
                                   (bhv_df.subject == subject))]
        # calculate mean power in freq bands:
        for band, frange in HLTP_pupil.freq_bands.items():
            _, mask = HLTP_pupil.load(HLTP_pupil.result_dir + 
                            '/pupil_states_ANOVA_' + band + b + '.pkl') 
            band_pwr = psd[:, :, (freq > frange[0]) & (freq <= frange[1])
                ].mean(axis = -1).mean(axis = -1)#[:, mask]
            bhv_df.loc[bhv_df.subject == subject, band ] = band_pwr
    bhv_df.to_pickle(HLTP_pupil.MEG_pro_dir +
                         '/results/all_subj_bhv_df_w_sensor_pwr.pkl')
def prep_rest_df_w_PSD():
    #---- prepare resting data frame with updated power bin---------------------
    dfs = []
    for b in ['rest01', 'rest02']:
    
        for s, subject in enumerate(subjects):
            sub_pro_dir = MEG_pro_dir + '/' + subject
            
            epochs, pupil = get_epochs_and_pupil(sub_pro_dir, b, subject) 

            if not (type(epochs) == mne.epochs.EpochsFIF):  continue
        
            # remove linear treand from each epoch        
            epochs._data = scipy.signal.detrend(epochs.get_data(), axis=-1)
        
            # power spectrum
            psd, freq = mne.time_frequency.psd_welch(epochs, n_fft = 2**9)
    
            df = pd.DataFrame({"subject" :np.repeat(subject, len(pupil)), 
                           "pupil"   :pupil})
            # calculate mean power in freq bands:
            for band, frange in HLTP_pupil.freq_bands.items(): 
                _, mask = HLTP_pupil.load(HLTP_pupil.result_dir + 
                            '/pupil_states_ANOVA_' + band + 'rest' + '.pkl') 
                band_pwr = psd[:, :, (freq > frange[0]) & (freq <= frange[1])
                    ].mean(axis = -1)[:, mask].mean(axis = -1)
                df[band] = band_pwr
            dfs.append(df)    
    pd.concat(dfs).to_pickle(HLTP_pupil.MEG_pro_dir +
                         '/results/all_subj_rest_df_w_sensor_pwr.pkl')    

def combine_rest_runs(res):
    ''' combine two rest runs '''
    mean_meg1, freq = HLTP_pupil.load(HLTP_pupil.result_dir + 
                        '/mean_meg_by_pupil_staterest01' + str(res) + '.pkl')
            
    mean_meg2, freq = HLTP_pupil.load(HLTP_pupil.result_dir + 
                        '/mean_meg_by_pupil_staterest02' + str(res) + '.pkl')
    s2 = 0
    for s, subject in enumerate(subjects):
        sub_pro_dir = MEG_pro_dir + '/' + subject
        epoch_fname = sub_pro_dir + '/rest02_ds-epo.fif'
        
        if path.exists(epoch_fname): # some sobjects didn't finish run 2
            mean_meg1[s, :,:,:] += mean_meg2[s2, :,:,:]
            mean_meg1[s, :,:,:] /= 2
            s2 += 1
    
    HLTP_pupil.save([mean_meg1, freq], HLTP_pupil.result_dir + 
                        '/mean_meg_by_pupil_staterest' + str(res) + '.pkl')


def fitLM_pupil_group_power(res):
    mean_meg, freq = HLTP_pupil.load(HLTP_pupil.result_dir + 
                        '/mean_meg_by_pupil_statetask_prestim' + str(res) + '.pkl')
    n_subj = mean_meg.shape[0]; n_pup = mean_meg.shape[1];
    # If I used noramlized power here, shouldn't do mixed effects
    for fband, frange in HLTP_pupil.freq_bands.items():
        mean_meg_fband = np.nanmean(mean_meg.mean(axis = 2)[:, :, 
                   (freq > frange[0]) & (freq < frange[1])], axis = -1)
        df = pd.DataFrame({"subject" :np.repeat(np.arange(n_subj), n_pup), 
                           "power"   :(mean_meg_fband.flatten()), 
                           "pupil"   :np.tile(np.arange(1, n_pup + 1), n_subj)})
        df.subject = df.subject.astype(str)
        mdf_Q = smf.mixedlm("power ~ np.power(pupil, 2) + pupil", 
                    df.dropna(), groups = df.dropna()["subject"]).fit(method='powell')
        mdf_Q.bic = bic(mdf_Q.llf, mdf_Q.nobs, mdf_Q.df_modelwc)
        mdf_Q.save(HLTP_pupil.MEG_pro_dir + '/results/mixedlmQ_pupil_' 
                       + fband + str(res) +'.pkl')
        mdf_L = smf.mixedlm("power ~ pupil", 
                    df.dropna(), groups = df.dropna()["subject"]).fit(
                            method='powell')
        mdf_L.bic = bic(mdf_L.llf, mdf_L.nobs, mdf_L.df_modelwc)
        mdf_L.save(HLTP_pupil.MEG_pro_dir + '/results/mixedlmL_pupil_' 
                       + fband + str(res) +'.pkl')
        print(fband, mdf_Q.pvalues, mdf_L.pvalues)
        
def fitLM_pupil_power_bhv():
    bhv_df = pd.read_pickle(HLTP_pupil.MEG_pro_dir +
                         '/results/all_subj_bhv_df_w_sensor_pwr.pkl')
    b = 'task_prestim'
    for fband, frange in HLTP_pupil.freq_bands.items():
        dfs = []
        for subject in HLTP_pupil.subjects:
            # get mean power spectrum
            pupil_states = HLTP_pupil.load(
                    HLTP_pupil.result_dir + '/pupil_states_' + 
                                b + subject + '.pkl')
            dfs.append(pd.DataFrame({"subject" :subject, 
                               "power"   :np.log(bhv_df[bhv_df.subject == subject][fband]), 
                               "pupil"   :pupil_states.mean_pupil.values}))
        df = pd.concat(dfs)
        df.power = zscore(df.power)
        mdf_Q = smf.mixedlm("power ~ np.power(pupil, 2) + pupil", 
                        df.dropna(), groups = df.dropna()["subject"]).fit(method='powell')
        mdf_Q.bic = bic(mdf_Q.llf, mdf_Q.nobs, mdf_Q.df_modelwc)
        mdf_Q.save(HLTP_pupil.MEG_pro_dir + '/results/mixedlmQ_full_pupil_bhv_' 
                           + fband +'.pkl')
        mdf_L = smf.mixedlm("power ~ pupil", 
                        df.dropna(), groups = df.dropna()["subject"]).fit(
                                method='powell')
        mdf_L.bic = bic(mdf_L.llf, mdf_L.nobs, mdf_L.df_modelwc)
        mdf_L.save(HLTP_pupil.MEG_pro_dir + '/results/mixedlmL_full_pupil_bhv_' 
                           + fband +'.pkl') 
        print(fband, mdf_Q.pvalues, mdf_L.pvalues)
        
def fitLM_pupil_power_rest():
    df = pd.read_pickle(HLTP_pupil.MEG_pro_dir +
                         '/results/all_subj_rest_df_w_sensor_pwr.pkl')        

    for fband, frange in HLTP_pupil.freq_bands.items():
        df["power"] = zscore(np.log(df[fband]))
        mdf_Q = smf.mixedlm("power ~ np.power(pupil, 2) + pupil", 
                        df.dropna(), groups = df.dropna()["subject"]).fit(method='powell')
        mdf_Q.bic = bic(mdf_Q.llf, mdf_Q.nobs, mdf_Q.df_modelwc)
        mdf_Q.save(HLTP_pupil.MEG_pro_dir + '/results/mixedlmQ_full_pupil_rest_' 
                           + fband +'.pkl')
        mdf_L = smf.mixedlm("power ~ pupil", 
                        df.dropna(), groups = df.dropna()["subject"]).fit(
                                method='powell')
        mdf_L.bic = bic(mdf_L.llf, mdf_L.nobs, mdf_L.df_modelwc)
        mdf_L.save(HLTP_pupil.MEG_pro_dir + '/results/mixedlmL_full_pupil_rest_' 
                           + fband +'.pkl') 
        print(fband, mdf_Q.pvalues, mdf_L.pvalues)
        
def fitLM_pupil_power_all_sensors():        
    return
        
# run functions        
save_sensor_PSD_by_pupil(10) # resolution of pupil binning
combine_rest_runs(10)
update_bhv_df_w_PSD()
fitLM_pupil_power(10)

# not used in paper:
def ANOVA_PSD_by_pupil_notused():
    '''  ANOVA with pupil size as a factor - currently not used'''

    factor_levels = [5]
    effects = 'A'
    p_accept = 0.05
    
    def stat_fun(*args):
            return mne.stats.f_mway_rm(np.swapaxes(np.array(args), 0, 1),
                   factor_levels, effects = effects, return_pvals = False)[0]
            
    for b in ['task_prestim', 'rest']:
        mean_meg, freq = HLTP_pupil.load(HLTP_pupil.result_dir + 
                        '/mean_meg_by_pupil_state' + b + '.pkl')
    
        n_subj = mean_meg.shape[0]  
        f_thresh = mne.stats.f_threshold_mway_rm(n_subj - 1, 
                                                 factor_levels, effects, 0.01)
        con, pos = HLTP_pupil.get_connectivity()
    
        for band, freq_range in HLTP_pupil.freq_bands.items():
         
            band_mean = np.nanmean(mean_meg[:, :, :, 
                        (freq > freq_range[0]) & (freq <= freq_range[1])], 
                        axis = -1)
            
            T_obs, clusters, cluster_p_values, H0 = \
                    mne.stats.permutation_cluster_test(
                            np.swapaxes(band_mean, 0, 1), 
                            connectivity = con.astype('int'),
                            n_jobs = 24, tail = 0, 
                            threshold = f_thresh, stat_fun=stat_fun, 
                            n_permutations = 10000)
                    
            mask = np.zeros(T_obs.shape, dtype=bool)
            good_cluster_inds = np.where( cluster_p_values < p_accept)[0]
            if len(good_cluster_inds) > 0:
                for g in good_cluster_inds:
                    mask[clusters[g]] = True
                    
            HLTP_pupil.save([T_obs, mask], HLTP_pupil.result_dir + 
                            '/pupil_states_ANOVA_' + band + b + '.pkl')
